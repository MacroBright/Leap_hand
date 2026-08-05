"""Offline integration: MediaPipe bbox → hamer regress → projection alignment.

Skips when hamer/GPU/mediapipe model are unavailable.
Validates the projection formula empirically: projected MANO kp must land
near MediaPipe kp (median < 60px). If it fails, check box_center/box_size
convention in project_to_frame.
"""
import numpy as np
import pytest

from gesture_mapping import HandTracker
from gesture_mapping.hamer_3d import HaMeR3D, hand_bbox_from_landmarks

HAMER_IMG = "/home/bright/office/hamer/example_data/test1.jpg"


def test_regress_and_projection_alignment():
    try:
        h3d = HaMeR3D()
    except Exception:
        pytest.skip("hamer init raised")
    if not h3d.available:
        pytest.skip("no GPU / hamer unavailable")

    import cv2
    img = cv2.imread(HAMER_IMG)
    if img is None:
        pytest.skip(f"missing test image: {HAMER_IMG}")
    h, w = img.shape[:2]

    tracker = HandTracker(max_num_hands=1)
    results = tracker.detect(img)
    if not results:
        pytest.skip("MediaPipe found no hand in test image")
    hand = results[0]
    mp_pts = tracker.landmark_xy(hand, (h, w))
    bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
    assert bbox is not None

    hres = h3d.regress(img, bbox)
    assert hres is not None, "hamer regression returned None"
    assert hres.kp3d.shape == (21, 3)
    assert np.isfinite(hres.kp3d).all()
    assert hres.verts.shape == (778, 3)

    proj = h3d.project_to_frame(hres, hres.kp3d)
    dist = np.linalg.norm(proj - mp_pts, axis=1)
    assert np.median(dist) < 60.0, f"projection misaligned, median={np.median(dist):.1f}px"

    tracker.close()
