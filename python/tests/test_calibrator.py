"""Tests for Calibrator/FingerIdentifier 3D point-cloud entry points."""
import numpy as np

from gesture_mapping import JointMapper, Calibrator, FingerIdentifier
from test_joint_mapper import _open_hand_pts, _bend_joint


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
