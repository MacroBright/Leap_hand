"""Tests for hamer_3d pure helpers (bbox + order mapping)."""
import numpy as np

from gesture_mapping.hamer_3d import hand_bbox_from_landmarks, to_mediapipe_order


def _spread_pts():
    # 21 points spanning x:[100,320] y:[100,340] in a 400x400 frame
    xs = np.linspace(100, 320, 21)
    ys = np.linspace(100, 340, 21)
    return np.stack([xs, ys], axis=1)


def test_to_mediapipe_order_is_identity():
    pts = np.arange(63, dtype=np.float64).reshape(21, 3)
    out = to_mediapipe_order(pts)
    assert out.dtype == np.float64
    assert np.array_equal(out, pts)


def test_bbox_square_and_clamped():
    pts = _spread_pts()
    bbox = hand_bbox_from_landmarks(pts, (400, 400), margin=1.5, square=True)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 >= 0 and y0 >= 0 and x1 <= 400 and y1 <= 400
    assert (x1 - x0) == (y1 - y0)                       # square
    assert x0 <= pts[:, 0].min() and x1 >= pts[:, 0].max()
    assert y0 <= pts[:, 1].min() and y1 >= pts[:, 1].max()


def test_bbox_margin_expands():
    pts = _spread_pts()
    bbox1 = hand_bbox_from_landmarks(pts, (400, 400), margin=1.0, square=True)
    bbox2 = hand_bbox_from_landmarks(pts, (400, 400), margin=2.0, square=True)
    assert (bbox2[2] - bbox2[0]) > (bbox1[2] - bbox1[0])


def test_bbox_too_small_returns_none():
    pts = np.full((21, 2), 50.0)
    assert hand_bbox_from_landmarks(pts, (100, 100), margin=1.5) is None
