"""手腕 3D + 掌参考系 → 机械臂差分速度 (核心遥操模块)。

纯函数部分: 深度反投影、掌参考系、俯仰/滚转角、delta→速度。
Task 8 追加 WristTracker 类 (动态参考 + 滤波 + J5/J6 钳制)。
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


# ── WristTracker: 动态参考 + 滤波 + 速度生成 + J5/J6 钳制 ──

class WristTracker:
    """把相机系 3D 手关键点流 → (vx,vy,vz,j5,j6)∈[-1,1] 差分速度。

    用法: capture_reference() 在离合器按下/松开时锚定参考; 之后每次
    update() 输出相对参考的速度。无手调用 no_hand() 清零。
    """

    def __init__(self, R: np.ndarray,
                 gain_pos: float = 0.008,          # 1/mm (±140mm→满速)
                 gain_pitch: float = 0.02,         # 1/deg (50°→0.8)
                 gain_roll: float = 0.02,          # 1/deg
                 deadzone_pos_mm: float = 15.0,
                 deadzone_ang_deg: float = 5.0,
                 j5_rate_deg_s: float = 45.0,
                 j6_rate_deg_s: float = 180.0,
                 j5_range=(0.0, 90.0), j6_range=(0.0, 360.0),
                 dt: float = 1.0 / 30.0,
                 min_cutoff: float = 2.0,          # Hz (2.0: 更跟手, 滞后减半)
                 beta: float = 0.02):
        self.R = np.asarray(R, float)
        self.gain_pos = gain_pos
        self.gain_pitch = gain_pitch
        self.gain_roll = gain_roll
        self.deadzone_pos_mm = deadzone_pos_mm
        self.deadzone_ang_deg = deadzone_ang_deg
        self.j5_rate_deg_s = j5_rate_deg_s
        self.j6_rate_deg_s = j6_rate_deg_s
        self.j5_range = j5_range
        self.j6_range = j6_range
        self.dt = dt
        # 状态
        self._ref_pts = None          # 参考 21 点 (相机系 mm)
        self._ref_f = None
        self._ref_n = None
        self._pos_filt = OneEuroFilter(3, min_cutoff=min_cutoff, beta=beta)
        self.j5_pos_deg = 0.0
        self.j6_pos_deg = 0.0
        self._has_ref = False
        self.last_delta_base = None   # 最近一次 update() 的基座系 delta (mm); 无手/无参考为 None

    # ── 参考 ──────────────────────────────────────────────

    def capture_reference(self, pts21_cam: Optional[np.ndarray]) -> None:
        if pts21_cam is None:
            self._has_ref = False
            return
        self._ref_pts = np.asarray(pts21_cam, float)
        f, n, _ = palm_basis(self._ref_pts)
        self._ref_f = apply_rotation(self.R, np.array([f]))[0]
        self._ref_n = apply_rotation(self.R, np.array([n]))[0]
        self._pos_filt.reset()
        self._has_ref = True

    def sync_j5j6(self, deg_j5: float, deg_j6: float) -> None:
        """从 get_state 同步 J5/J6 实际角度 (remote_enable 软复位后调用)."""
        self.j5_pos_deg = float(np.clip(deg_j5, *self.j5_range))
        self.j6_pos_deg = float(np.clip(deg_j6, *self.j6_range))

    # ── 主更新 ────────────────────────────────────────────

    def update(self, pts21_cam: Optional[np.ndarray]):
        """返回 (vx,vy,vz,j5_cmd,j6_cmd) ∈ [-1,1]。无手/无参考 → 全 0。"""
        if pts21_cam is None or not self._has_ref:
            self.last_delta_base = None
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        pts = np.asarray(pts21_cam, float)

        # 位置 delta (基座系) + 平滑
        wrist = apply_rotation(self.R, np.array([pts[_WRIST]]))[0]
        wrist = self._pos_filt(wrist)
        ref_w = apply_rotation(self.R, np.array([self._ref_pts[_WRIST]]))[0]
        delta = wrist - ref_w
        self.last_delta_base = delta
        vx = delta_to_velocity(delta[0], self.gain_pos, self.deadzone_pos_mm)
        vy = delta_to_velocity(delta[1], self.gain_pos, self.deadzone_pos_mm)
        vz = delta_to_velocity(delta[2], self.gain_pos, self.deadzone_pos_mm)

        # 姿态角 delta
        f, n, _ = palm_basis(pts)
        f_base = apply_rotation(self.R, np.array([f]))[0]
        n_base = apply_rotation(self.R, np.array([n]))[0]
        pitch_deg = math.degrees(pitch_angle(f_base))
        roll_deg = math.degrees(roll_angle(n_base, f_base, self._ref_n, self._ref_f))
        ref_pitch = math.degrees(pitch_angle(self._ref_f))
        j5_cmd = delta_to_velocity(pitch_deg - ref_pitch,
                                   self.gain_pitch, self.deadzone_ang_deg)
        j6_cmd = delta_to_velocity(roll_deg, self.gain_roll, self.deadzone_ang_deg)

        # J5/J6 位置跟踪 + 边界钳制 (固件无限位, PC 侧负责)
        self.j5_pos_deg = float(np.clip(
            self.j5_pos_deg + j5_cmd * self.j5_rate_deg_s * self.dt,
            *self.j5_range))
        self.j6_pos_deg = float(np.clip(
            self.j6_pos_deg + j6_cmd * self.j6_rate_deg_s * self.dt,
            *self.j6_range))
        if (self.j5_pos_deg <= self.j5_range[0] and j5_cmd < 0) or \
           (self.j5_pos_deg >= self.j5_range[1] and j5_cmd > 0):
            j5_cmd = 0.0
        if (self.j6_pos_deg <= self.j6_range[0] and j6_cmd < 0) or \
           (self.j6_pos_deg >= self.j6_range[1] and j6_cmd > 0):
            j6_cmd = 0.0

        return (vx, vy, vz, j5_cmd, j6_cmd)

    def no_hand(self):
        """无手帧: 输出全 0 (速度命令清零, 臂保持)。"""
        self._pos_filt.reset()
        self.last_delta_base = None
        return (0.0, 0.0, 0.0, 0.0, 0.0)
