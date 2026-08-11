"""wrist_tracker 纯函数单测。"""
import numpy as np
import pytest

from gesture_mapping.wrist_tracker import (
    WristTracker,
    backproject, build_palm_pts, delta_to_velocity, median_depth_at,
    palm_basis, pitch_angle, roll_angle,
)
from gesture_mapping.handeye_calib import rot_from_euler


def test_backproject_center():
    # u=cx,v=cy → X=Y=0, Z=depth
    xyz = backproject(320.0, 240.0, 1000.0, (500.0, 500.0, 320.0, 240.0))
    np.testing.assert_allclose(xyz, [0, 0, 1000], atol=1e-6)


def test_backproject_offcenter():
    # X = (u-cx)*Z/fx = (420-320)*1000/500 = 200
    xyz = backproject(420.0, 240.0, 1000.0, (500.0, 500.0, 320.0, 240.0))
    np.testing.assert_allclose(xyz, [200, 0, 1000], atol=1e-6)


def test_median_depth_at_ignores_outliers():
    d = np.ones((20, 20), dtype=np.uint16) * 800
    d[9:12, 9:12] = 0          # 中心飞点
    d[10, 10] = 9999           # 深噪点
    v = median_depth_at(d, 10.0, 10.0, patch=7)
    assert v == 800.0


def test_median_depth_nan_when_empty():
    d = np.zeros((20, 20), dtype=np.uint16)
    assert np.isnan(median_depth_at(d, 10, 10, patch=7))


def test_palm_basis_orthonormal():
    pts = np.zeros((21, 3))
    pts[0] = [0, 0, 0]             # wrist
    pts[5] = [10, 0, 5]            # index_mcp
    pts[9] = [12, 0, 0]            # middle_mcp
    pts[17] = [8, 0, -5]           # pinky_mcp
    f, n, lat = palm_basis(pts)
    for v in (f, n, lat):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert abs(np.dot(n, f)) < 1e-9


def test_pitch_angle_up_is_positive():
    assert abs(pitch_angle(np.array([0.0, 0.0, 1.0])) - np.pi / 2) < 1e-9
    assert abs(pitch_angle(np.array([1.0, 0.0, 0.0]))) < 1e-9


def test_roll_angle():
    # f=+z, n_ref=+x, n_now=+y → 绕 z 从 x 转 +90°
    roll = roll_angle(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                      np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert abs(roll - np.pi / 2) < 1e-9


def test_delta_to_velocity_deadzone_saturation():
    assert delta_to_velocity(5.0, gain=0.01, deadzone=10.0) == 0.0
    v = delta_to_velocity(50.0, gain=0.01, deadzone=10.0)
    assert abs(v - 0.4) < 1e-9      # (50-10)*0.01
    assert delta_to_velocity(1000.0, gain=0.01, deadzone=10.0) == 1.0  # 饱和


# ── WristTracker 类 (位置跟随) ─────────────────────────────


def _identity_pts21(hand_pts):
    pts = np.zeros((21, 3))
    pts[0] = hand_pts[0]       # wrist
    pts[5] = hand_pts[1]       # index_mcp
    pts[9] = hand_pts[2]       # middle_mcp
    pts[17] = hand_pts[3]      # pinky_mcp
    return pts


def test_no_hand_zeroes():
    wt = WristTracker(R=rot_from_euler(0, 0, 0))
    assert wt.update(None, np.zeros(3), 0, 0) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_position_loop_drives_velocity():
    R = rot_from_euler(0, 0, 0)
    wt = WristTracker(R=R)                       # k_pos=0.01, deadzone=8mm
    ref = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 0, 1000], [8, 0, 995]])
    wt.capture(ref, np.zeros(3), 0.0, 0.0)
    # 手沿 +x 移 50mm → 目标 (50,0,0)mm; ee 还在原点 → error 50mm
    now = _identity_pts21([[50, 0, 1000], [60, 0, 1005], [62, 0, 1000], [58, 0, 995]])
    vx, vy, vz, j5, j6 = wt.update(now, np.zeros(3), 0.0, 0.0)
    assert vx > 0.03                     # (50-8)*0.01 = 0.42
    assert vy == 0.0 and vz == 0.0


def test_position_loop_closes_error():
    R = rot_from_euler(0, 0, 0)
    wt = WristTracker(R=R)
    ref = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 0, 1000], [8, 0, 995]])
    wt.capture(ref, np.zeros(3), 0.0, 0.0)
    # 手 +50mm, 但 ee 已到 (45,0,0)mm → 剩 5mm < 死区 8mm → 停
    now = _identity_pts21([[50, 0, 1000], [60, 0, 1005], [62, 0, 1000], [58, 0, 995]])
    vx, vy, vz, j5, j6 = wt.update(now, np.array([45.0, 0.0, 0.0]), 0.0, 0.0)
    assert vx == 0.0
    assert vy == 0.0 and vz == 0.0


def test_j5_target_clamped():
    R = rot_from_euler(0, 0, 0)
    wt = WristTracker(R=R)
    ref = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 0, 1000], [8, 0, 995]])
    wt.capture(ref, np.zeros(3), 0.0, 0.0)
    # 手一直"向上" (f 的 z 分量大) → j5 目标应钳制 ≤90°, j5/j6 命令有界
    up = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 0, 1050], [8, 0, 995]])
    vx, vy, vz, j5, j6 = wt.update(up, np.zeros(3), 0.0, 0.0)
    assert wt.last_target_j5 <= 90.0
    assert -1.0 <= j5 <= 1.0
    assert -1.0 <= j6 <= 1.0


def test_capture_reanchors():
    R = rot_from_euler(0, 0, 0)
    wt = WristTracker(R=R)
    ref = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 0, 1000], [8, 0, 995]])
    wt.capture(ref, np.zeros(3), 0.0, 0.0)
    # 重锚定: 手参考+臂锚点一起更新 → 该手位/该臂位下误差为 0 → 全速 0
    hand2 = _identity_pts21([[50, 0, 1000], [60, 0, 1005], [62, 0, 1000], [58, 0, 995]])
    ee2 = np.array([30.0, 0.0, 0.0])
    wt.capture(hand2, ee2, 0.0, 0.0)
    vx, vy, vz, j5, j6 = wt.update(hand2, ee2, 0.0, 0.0)
    assert vx == 0.0 and vy == 0.0 and vz == 0.0


# ── build_palm_pts: HandResult 归一化 landmarks → 像素反投影 ──


class _FakeLM:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _FakeHand:
    def __init__(self, lms):
        self.landmarks = lms


def test_build_palm_pts_backprojects_pixels():
    from gesture_mapping.wrist_tracker import build_palm_pts
    h, w = 480, 640
    depth = np.full((h, w), 1000, dtype=np.uint16)
    lms = [_FakeLM(0, 0)] * 21
    lms[0] = _FakeLM(320 / w, 240 / h)      # wrist → 中心 → 反投影 (0,0,1000)
    lms[5] = _FakeLM(340 / w, 220 / h)      # index_mcp
    lms[9] = _FakeLM(360 / w, 240 / h)      # middle_mcp
    lms[17] = _FakeLM(300 / w, 260 / h)     # pinky_mcp
    pts = build_palm_pts(_FakeHand(lms), depth, (500.0, 500.0, 320.0, 240.0))
    assert pts is not None
    np.testing.assert_allclose(pts[0], [0, 0, 1000], atol=1e-6)
    assert np.all(np.isfinite(pts[5])) and np.all(np.isfinite(pts[9]))


def test_build_palm_pts_none_when_hand_missing_depth():
    from gesture_mapping.wrist_tracker import build_palm_pts
    assert build_palm_pts(None, None, None) is None
