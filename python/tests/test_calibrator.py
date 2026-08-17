"""Tests for Calibrator/FingerIdentifier 3D point-cloud entry points."""
import numpy as np

from gesture_mapping import JointMapper, Calibrator, FingerIdentifier
from test_joint_mapper import _open_hand_pts, _bend_joint, _synthetic_hand_result


def test_calibrator_points_open_is_zero():
    mapper = JointMapper()
    cal = Calibrator(mapper)
    open_pts = _open_hand_pts()
    cal.calibrate_points(open_pts)
    assert cal.is_calibrated
    assert np.allclose(cal.map_points(open_pts), 0.0, atol=1e-9)


def test_calibrator_points_isolates_bend():
    mapper = JointMapper()
    cal = Calibrator(mapper)
    open_pts = _open_hand_pts()
    cal.calibrate_points(open_pts)
    bent = _bend_joint(open_pts, "index", "mcp", 0.9)
    a = cal.map_points(bent)
    assert a[1] > 0.5        # index mcp bends away from baseline
    assert abs(a[5]) < 0.05  # middle unaffected


def test_identify_points_open_is_none():
    mapper = JointMapper()
    fi = FingerIdentifier(mapper, bend_threshold=0.20)
    bent, scores = fi.identify_points(_open_hand_pts())
    assert bent is None


def test_identify_points_index_bent():
    mapper = JointMapper()
    fi = FingerIdentifier(mapper, bend_threshold=0.20)
    pts = _bend_joint(_open_hand_pts(), "index", "mcp", 0.9)
    pts = _bend_joint(pts, "index", "pip", 0.6)
    pts = _bend_joint(pts, "index", "dip", 0.4)
    bent, scores = fi.identify_points(pts)
    assert bent == "index"
    assert scores["index"] >= 0.20


def test_points_baseline_save_load(tmp_path):
    """Persisted points baseline survives a new Calibrator (no per-session SPACE)."""
    mapper = JointMapper()
    open_pts = _open_hand_pts()
    cal = Calibrator(mapper)
    cal.calibrate_points(open_pts)
    p = tmp_path / "calib_3d.json"
    cal.save_points_baseline(str(p))

    cal2 = Calibrator(mapper)
    assert not cal2.is_calibrated
    assert cal2.load_points_baseline(str(p))
    assert cal2.is_calibrated
    # 载入基线后, 同一张开点云应映射回 0
    assert np.allclose(cal2.map_points(open_pts), 0.0, atol=1e-9)


def test_points_and_hand_baselines_are_independent():
    """calibrate_points must not zero the HandResult-based map() and vice versa."""
    mapper = JointMapper()
    cal = Calibrator(mapper)
    open_pts = _open_hand_pts()
    bent_pts = _bend_joint(open_pts, "index", "mcp", 0.9)
    hr_open = _synthetic_hand_result(open_pts)
    hr_bent = _synthetic_hand_result(bent_pts)
    raw_open = mapper.map_points_to_leap(open_pts)

    # Points path calibrated with a BENT cloud (non-zero baseline) → only the
    # points path is zeroed. The HandResult path must still return RAW angles.
    cal.calibrate_points(bent_pts)
    assert cal.is_calibrated
    assert np.allclose(cal.map_points(bent_pts), 0.0, atol=1e-9)
    assert np.allclose(cal.map(hr_open, (1, 1)), raw_open)

    # Vice versa: a HandResult baseline must NOT zero the points path.
    cal2 = Calibrator(mapper)
    cal2.calibrate(hr_bent, (1, 1))
    assert cal2.is_calibrated
    assert np.allclose(cal2.map(hr_bent, (1, 1)), 0.0, atol=1e-9)
    assert np.allclose(cal2.map_points(open_pts), raw_open)
