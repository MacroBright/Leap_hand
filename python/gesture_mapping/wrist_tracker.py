"""手腕 3D + 掌参考系 → 机械臂差分速度 (核心遥操模块)。

纯函数部分: 深度反投影、掌参考系、俯仰/滚转角、delta→速度。
Task 8 追加 WristTracker 类 (动态参考 + 滤波 + J5/J6 钳制)。
"""
import math
from typing import Optional, Tuple

import numpy as np

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

    任一所需关键点深度无效时返回 None。hand.landmark_xy 是 (21,2) 像素。
    """
    if depth is None or K is None:
        return None
    xy = hand.landmark_xy
    pts = np.zeros((21, 3))
    for i in (_WRIST, _MCP_INDEX, _MCP_MIDDLE, _MCP_PINKY):
        u, v = xy[i]
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
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
