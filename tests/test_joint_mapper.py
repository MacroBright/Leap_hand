"""Tests for JointMapper.map_points_to_leap (3D point-cloud path)."""
import numpy as np
import pytest
from types import SimpleNamespace

from gesture_mapping import JointMapper
from gesture_mapping.hand_tracker import HandResult


# ─── Synthetic hand builders ──────────────────────────────────────
# A flat open hand: wrist at origin, fingers as straight rays → all flexion ≈ 0.

_JCHAIN = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
_JJOINT = {"mcp": 0, "pip": 1, "dip": 2}
_SEG = {  # per-finger segment lengths [wrist→MCP, MCP→PIP, PIP→DIP, DIP→TIP]
    "thumb":  [0.45, 0.35, 0.30, 0.30],
    "index":  [0.50, 0.50, 0.40, 0.40],
    "middle": [0.50, 0.50, 0.40, 0.40],
    "ring":   [0.50, 0.50, 0.40, 0.40],
    "pinky":  [0.50, 0.50, 0.40, 0.40],
}
_DIRS = {
    "thumb":  np.array([0.95, 0.30, 0.0]),
    "index":  np.array([0.35, 0.94, 0.0]),
    "middle": np.array([0.12, 0.99, 0.0]),
    "ring":   np.array([-0.10, 0.99, 0.0]),
    "pinky":  np.array([-0.30, 0.95, 0.0]),
}


def _open_hand_pts() -> np.ndarray:
    pts = np.zeros((21, 3), dtype=np.float64)
    for name, chain in _JCHAIN.items():
        d = _DIRS[name] / np.linalg.norm(_DIRS[name])
        seg = _SEG[name]
        for j, idx in enumerate(chain):
            pts[idx] = d * sum(seg[:j + 1])
    return pts


def _rot(p, pivot, axis, theta):
    v = p - pivot
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    return pivot + v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)


def _bend_joint(pts, finger, joint, theta) -> np.ndarray:
    """Rotate the part of `finger` distal to `joint` by theta out of the palm plane."""
    ch = _JCHAIN[finger]
    j = _JJOINT[joint]
    pivot = pts[ch[j]]
    proximal = pts[0] if j == 0 else pts[ch[j - 1]]
    axis = np.cross(pts[ch[j]] - proximal, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return pts.copy()
    axis = axis / n
    out = pts.copy()
    for k in range(j + 1, len(ch)):
        out[ch[k]] = _rot(pts[ch[k]], pivot, axis, theta)
    return out


def _synthetic_hand_result(pts):
    lms = [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
    return HandResult(landmarks=lms, handedness="Right")


# ─── Tests ────────────────────────────────────────────────────────

def test_open_hand_has_low_flexion():
    mapper = JointMapper()
    a = mapper.map_points_to_leap(_open_hand_pts())
    flexion_ids = [1, 2, 3, 5, 6, 7, 9, 10, 11]  # index/middle/pinky mcp+pip+dip
    assert np.all(np.abs(a[flexion_ids]) < 0.05)


def test_bending_mcp_increases_index_mcp_only():
    mapper = JointMapper()
    open_a = mapper.map_points_to_leap(_open_hand_pts())
    bent = _bend_joint(_open_hand_pts(), "index", "mcp", 0.8)
    a = mapper.map_points_to_leap(bent)
    assert a[1] > 0.5        # index mcp
    assert abs(a[2]) < 0.05  # index pip unchanged
    assert abs(a[3]) < 0.05  # index dip unchanged


def test_bending_pip_increases_index_pip_only():
    mapper = JointMapper()
    open_a = mapper.map_points_to_leap(_open_hand_pts())
    bent = _bend_joint(_open_hand_pts(), "index", "pip", 0.8)
    a = mapper.map_points_to_leap(bent)
    assert a[2] > 0.5
    assert abs(a[1]) < 0.05
    assert abs(a[3]) < 0.05


def test_bending_dip_increases_index_dip_only():
    mapper = JointMapper()
    open_a = mapper.map_points_to_leap(_open_hand_pts())
    bent = _bend_joint(_open_hand_pts(), "index", "dip", 0.8)
    a = mapper.map_points_to_leap(bent)
    assert a[3] > 0.5
    assert abs(a[1]) < 0.05
    assert abs(a[2]) < 0.05


def test_map_keypoints_delegates_to_map_points():
    """Refactor guard: both entry points must agree on the same point cloud."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, size=(21, 3))
    hr = _synthetic_hand_result(pts)
    mapper = JointMapper()
    assert np.allclose(
        mapper.map_keypoints_to_leap(hr, (1, 1)),
        mapper.map_points_to_leap(pts),
    )


def test_map_points_to_leap_dict_roundtrip():
    mapper = JointMapper()
    pts = _open_hand_pts()
    arr = mapper.map_points_to_leap(pts)
    d = mapper.map_points_to_leap_dict(pts)
    assert set(d.keys()) == {"index", "middle", "pinky", "thumb"}
    # standard fingers: group order [abd, mcp, pip, dip]
    for i, name in enumerate(["index", "middle", "pinky"]):
        assert d[name]["abduction"] == pytest.approx(arr[i * 4 + 0])
        assert d[name]["mcp"] == pytest.approx(arr[i * 4 + 1])
        assert d[name]["pip"] == pytest.approx(arr[i * 4 + 2])
        assert d[name]["dip"] == pytest.approx(arr[i * 4 + 3])
    # thumb physical order differs: [mcp, abd, pip, dip]
    assert d["thumb"]["mcp"] == pytest.approx(arr[12])
    assert d["thumb"]["abduction"] == pytest.approx(arr[13])
    assert d["thumb"]["pip"] == pytest.approx(arr[14])
    assert d["thumb"]["dip"] == pytest.approx(arr[15])


def test_map_points_rejects_bad_shape():
    mapper = JointMapper()
    with pytest.raises(ValueError):
        mapper.map_points_to_leap(np.zeros((20, 3)))


def test_map_points_accepts_explicit_frame():
    mapper = JointMapper()
    pts = _open_hand_pts()
    auto = mapper.map_points_to_leap(pts)
    # 传入与 _palm_frame 相同的参考系 → 结果与自动计算一致
    frame = mapper._palm_frame(pts)
    assert np.allclose(auto, mapper.map_points_to_leap(pts, frame=frame))
    # 传入乱参考系 → 结果不同 (证明 frame 被使用)
    bad = (frame[0], np.array([1.0, 0.0, 0.0]),
           np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert not np.allclose(auto, mapper.map_points_to_leap(pts, frame=bad))


# ─── 拇指对掌增强 (方案B, 2026-08-06) ─────────────────────────────

def test_thumb_opposition_flat_open_is_zero():
    """张开手 → 对掌度≈0 → 拇指 mcp/pip 不被补偿 (回归守卫)."""
    mapper = JointMapper()
    a = mapper.map_points_to_leap(_open_hand_pts())
    assert abs(a[12]) < 0.05   # thumb mcp
    assert abs(a[14]) < 0.05   # thumb pip


def test_thumb_tip_to_palm_increases_flexion():
    """拇指 TIP 移向掌心 (对掌) → thumb mcp/pip 屈曲显著增大."""
    mapper = JointMapper()
    open_a = mapper.map_points_to_leap(_open_hand_pts())
    pts = _open_hand_pts()
    palm_center = 0.5 * (pts[5] + pts[17])   # index_mcp 与 pinky_mcp 中点
    pts[4] = palm_center + np.array([0.0, 0.0, 0.05])   # THUMB_TIP → 掌心
    a = mapper.map_points_to_leap(pts)
    assert a[12] > open_a[12] + 0.3   # thumb mcp 增大
    assert a[14] > open_a[14] + 0.3   # thumb pip 增大
    assert a[12] > 0.3 and a[14] > 0.3
