"""wrist_tracker 纯函数单测。"""
import numpy as np
import pytest

from gesture_mapping.wrist_tracker import (
    backproject, build_palm_pts, delta_to_velocity, median_depth_at,
    palm_basis, pitch_angle, roll_angle,
)


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
