"""手腕 3D + 掌参考系 → 机械臂位置跟随遥操 (核心遥操模块)。

纯函数部分: 深度反投影、掌参考系、俯仰/滚转角、delta→速度。
WristTracker 位置跟随范式 (参考 teleop_gesture_toolbox TeleoperationByDrawing):
  手位移 → 目标末端位置 → P 位置环 → 速度命令; 松开 H 重锚定 (走哪停哪)。
"""
import math
from typing import Optional, Tuple

import numpy as np

from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.handeye_calib import apply_rotation

# MediaPipe 21 点索引
_WRIST = 0
_MCP_INDEX = 5
_MCP_MIDDLE = 9
_MCP_PINKY = 17

K = Tuple[float, float, float, float]  # (fx, fy, cx, cy)


# ── 深度反投影 ──────────────────────────────────────────────

def backproject(u: float, v: float, depth_mm: float, K: K) -> np.ndarray:
    """像素 (u,v) + 深度(mm) → 相机系 3D (mm)。"""
    fx, fy, cx, cy = K
    Z = float(depth_mm)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def median_depth_at(depth: np.ndarray, u: float, v: float,
                    patch: int = 7) -> float:
    """wrist 邻域中值深度(mm)。越界/全 0 返回 nan。"""
    h, w = depth.shape
    r = patch // 2
    u0, u1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    v0, v1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    patch_d = depth[v0:v1, u0:u1]
    patch_d = patch_d[patch_d > 0]
    if patch_d.size == 0:
        return float("nan")
    return float(np.median(patch_d))


def build_palm_pts(hand, depth: Optional[np.ndarray],
                   K: Optional[K]) -> Optional[np.ndarray]:
    """从 HandResult + 对齐深度反投影出所需关键点 (21,3) 相机系 mm。

    任一所需关键点深度无效时返回 None。hand.landmarks 为归一化 Landmark
    (x,y∈[0,1]), 用 depth.shape 换算像素。
    """
    if depth is None or K is None:
        return None
    h, w = depth.shape
    lm = hand.landmarks
    pts = np.zeros((21, 3))
    for i in (_WRIST, _MCP_INDEX, _MCP_MIDDLE, _MCP_PINKY):
        u = lm[i].x * w
        v = lm[i].y * h
        if not (0 <= u < w and 0 <= v < h):
            return None
        z = median_depth_at(depth, u, v)
        if not math.isfinite(z):
            return None
        pts[i] = backproject(u, v, z, K)
    return pts


# ── 掌参考系 ────────────────────────────────────────────────

def palm_basis(pts21: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, n, lat) 单位向量。f: wrist→中指MCP; n: 掌法线; lat: cross(f,n)。"""
    f = pts21[_MCP_MIDDLE] - pts21[_WRIST]
    f = f / (np.linalg.norm(f) + 1e-9)
    across = pts21[_MCP_PINKY] - pts21[_MCP_INDEX]
    n = np.cross(across, f)
    n = n / (np.linalg.norm(n) + 1e-9)
    lat = np.cross(f, n)
    return f, n, lat


# ── 旋转角 (相对参考) ───────────────────────────────────────

def pitch_angle(f_base: np.ndarray) -> float:
    """f 相对基座系水平面的俯仰角 (rad), f 向上为正。"""
    return math.asin(max(-1.0, min(1.0, float(f_base[2]))))


def _proj_perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    v = v - (v @ axis) * axis
    return v / (np.linalg.norm(v) + 1e-9)


def roll_angle(n_base: np.ndarray, f_base: np.ndarray,
               n_ref: np.ndarray, f_ref: np.ndarray) -> float:
    """掌滚转角 (rad): n 绕 f 轴相对参考的有向转角 (旋前为正)。"""
    a = _proj_perp(n_ref, f_ref)
    b = _proj_perp(n_base, f_ref)
    return math.atan2(np.dot(np.cross(a, b), f_ref), np.dot(a, b))


# ── delta → 速度 ────────────────────────────────────────────

def delta_to_velocity(delta: float, gain: float, deadzone: float,
                      max_vel: float = 1.0) -> float:
    """delta → [-1,1] 速度: 死区内 0, 线性增益, 饱和 max_vel。"""
    d = float(delta)
    if abs(d) < deadzone:
        return 0.0
    v = (d - math.copysign(deadzone, d)) * gain
    return float(np.clip(v, -max_vel, max_vel))


# ── WristTracker: 位置跟随遥操 (手位移 → 目标 → P 位置环 → 速度) ──

class WristTracker:
    """位置跟随遥操: 手位移 → 目标末端位置 → P 位置环 → 速度命令.

    update() 需每帧喂入末端反馈 ee_mm(基座系mm) 与关节反馈 j4/j5_current(度).
    按住 H 时手位移决定目标; 松开时 capture() 重锚定 (走哪停哪).
    手滚转 → J4 主旋转, 手俯仰 → J5 (J6 不再手控).
    """

    def __init__(self, R: np.ndarray,
                 scale_pos: float = 1.0,          # 手位移mm → 末端目标mm
                 scale_ang: float = 1.0,          # 手滚转/俯仰deg → J4/J5目标deg
                 k_pos: float = 0.01,             # 位置环增益 (1/mm): 100mm误差→~0.9满速
                 k_ang: float = 0.02,             # 角度环增益 (1/deg): 50°误差→~0.9满速
                 deadzone_pos_mm: float = 8.0,    # 位置死区 (防末端抖动)
                 deadzone_ang_deg: float = 3.0,   # 角度死区
                 j5_range=(0.0, 90.0), j4_range=(-180.0, 180.0),   # J5/J4 目标钳制范围
                 dt: float = 1.0 / 30.0,
                 min_cutoff: float = 2.0, beta: float = 0.02):
        self.R = np.asarray(R, float)
        self.scale_pos, self.scale_ang = scale_pos, scale_ang
        self.k_pos, self.k_ang = k_pos, k_ang
        self.deadzone_pos_mm, self.deadzone_ang_deg = deadzone_pos_mm, deadzone_ang_deg
        self.j5_range, self.j4_range = j5_range, j4_range
        self.dt = dt
        # 手参考 + 臂锚点
        self._ref_pts = None
        self._ref_f = self._ref_n = None
        self._has_ref = False
        self._pos_filt = OneEuroFilter(3, min_cutoff=min_cutoff, beta=beta)
        self._anchor_ee = np.zeros(3)     # 基座系 mm
        self._anchor_j5 = 0.0
        self._anchor_j4 = 0.0
        # 深度时域中值 (Z 轴鲁棒性, 借鉴 toolbox temporal filter)
        self._depth_buf = []
        self.DEPTH_BUF_N = 5
        # 诊断 (HUD 依赖)
        self.last_delta_base = np.zeros(3)
        self.last_target_ee = np.zeros(3)
        self.last_target_j5 = 0.0
        self.last_target_j4 = 0.0
        self.last_roll_deg = 0.0
        self.last_pitch_deg = 0.0

    def capture(self, pts21_cam, ee_mm, j5_deg, j4_deg) -> None:
        """捕获手参考 + 臂锚点 (clutch 按下/松开时调用). pts=None 只清手参考."""
        if pts21_cam is not None:
            self._ref_pts = np.asarray(pts21_cam, float)
            f, n, _ = palm_basis(self._ref_pts)
            self._ref_f = apply_rotation(self.R, np.array([f]))[0]
            self._ref_n = apply_rotation(self.R, np.array([n]))[0]
            self._pos_filt.reset()
            self._has_ref = True
        else:
            self._has_ref = False
        if ee_mm is not None:
            self._anchor_ee = np.asarray(ee_mm, float)
        self._anchor_j5 = float(j5_deg)
        self._anchor_j4 = float(j4_deg)
        self._depth_buf.clear()
        self.last_delta_base = np.zeros(3)
        self.last_target_ee = np.array(self._anchor_ee)
        self.last_target_j5 = self._anchor_j5
        self.last_target_j4 = self._anchor_j4
        self.last_roll_deg = 0.0
        self.last_pitch_deg = 0.0

    def update(self, pts21_cam, ee_mm, j5_deg, j4_deg=0.0):
        """位置跟随: 返回 (vx,vy,vz,j4_cmd,j5_cmd) ∈ [-1,1]. 无手/无参考 → 全0."""
        if pts21_cam is None or not self._has_ref:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        pts = np.asarray(pts21_cam, float)

        # 深度时域中值: wrist 深度(Z)滚动中值, 去飞点
        if self._depth_buf or pts[0][2] > 0:
            self._depth_buf.append(float(pts[0][2]))
            if len(self._depth_buf) > self.DEPTH_BUF_N:
                self._depth_buf.pop(0)
        if self._depth_buf:
            pts = pts.copy()
            pts[0][2] = float(np.median(self._depth_buf))

        # 位置目标 = 锚点 + 手位移·scale
        wrist = apply_rotation(self.R, np.array([pts[0]]))[0]
        wrist = self._pos_filt(wrist)
        ref_w = apply_rotation(self.R, np.array([self._ref_pts[0]]))[0]
        delta = wrist - ref_w
        self.last_delta_base = delta
        target_ee = self._anchor_ee + delta * self.scale_pos
        self.last_target_ee = target_ee

        # 位置环: error → 速度 (P + 死区 + 饱和)
        # ee 反馈不可用(真机固件无 get_ee)时退回锚点 → error=手位移·scale (差分模式降级)
        if ee_mm is None:
            ee_fb = self._anchor_ee
        else:
            ee_fb = np.asarray(ee_mm, float)
        err = target_ee - ee_fb
        vx = delta_to_velocity(err[0], self.k_pos, self.deadzone_pos_mm)
        vy = delta_to_velocity(err[1], self.k_pos, self.deadzone_pos_mm)
        vz = delta_to_velocity(err[2], self.k_pos, self.deadzone_pos_mm)

        # 姿态目标: 滚转→J4 主旋转, 俯仰→J5
        f, n, _ = palm_basis(pts)
        f_base = apply_rotation(self.R, np.array([f]))[0]
        n_base = apply_rotation(self.R, np.array([n]))[0]
        pitch_deg = math.degrees(pitch_angle(f_base))
        roll_deg = math.degrees(roll_angle(n_base, f_base, self._ref_n, self._ref_f))
        self.last_pitch_deg, self.last_roll_deg = pitch_deg, roll_deg
        ref_pitch = math.degrees(pitch_angle(self._ref_f))
        target_j4 = float(np.clip(self._anchor_j4 + roll_deg * self.scale_ang,
                                  *self.j4_range))
        target_j5 = float(np.clip(self._anchor_j5 + (pitch_deg - ref_pitch) * self.scale_ang,
                                  *self.j5_range))
        self.last_target_j4, self.last_target_j5 = target_j4, target_j5

        # 角度环
        j4_cmd = delta_to_velocity(target_j4 - float(j4_deg), self.k_ang, self.deadzone_ang_deg)
        j5_cmd = delta_to_velocity(target_j5 - float(j5_deg), self.k_ang, self.deadzone_ang_deg)

        return (vx, vy, vz, j4_cmd, j5_cmd)

    def no_hand(self):
        self._pos_filt.reset()
        self.last_roll_deg = 0.0
        self.last_pitch_deg = 0.0
        return (0.0, 0.0, 0.0, 0.0, 0.0)
