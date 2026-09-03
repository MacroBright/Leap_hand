"""Compare MediaPipe pseudo-3D vs hamer 3D flexion under tilt/rotate/clench.

For each test image: run both sources, print both angle vectors. Skips on
missing GPU/hamer. Saves a comparison overlay to python/images/<stem>_compare.jpg.
"""
import glob
import os

import numpy as np
import pytest

from gesture_mapping import HandTracker, JointMapper
from gesture_mapping.hamer_3d import HaMeR3D, hand_bbox_from_landmarks
from gesture_mapping.demo_hamer3d import _draw_hamer_overlay

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def test_hamer_and_mediapipe_angles_for_pose_images():
    h3d = HaMeR3D()
    if not h3d.available:
        pytest.skip("no GPU / hamer unavailable")
    import cv2

    tracker = HandTracker(max_num_hands=1)
    mapper = JointMapper()

    imgs = sorted(p for p in glob.glob(os.path.join(IMAGES_DIR, "*.jpg"))
                  if "_compare" not in os.path.basename(p))
    if not imgs:
        pytest.skip("no images in python/images/")

    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        results = tracker.detect(img)
        if not results:
            print(f"  {os.path.basename(path)}: no hand detected")
            continue
        hand = results[0]
        mp_pts = tracker.landmark_xy(hand, (h, w))
        bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
        hres = h3d.regress(img, bbox) if bbox is not None else None

        mp_angles = mapper.map_keypoints_to_leap(hand, (h, w))
        h3_angles = mapper.map_points_to_leap(hres.kp3d) if hres is not None else np.zeros(16)

        print(f"\n  === {os.path.basename(path)} ===")
        print(f"  {'DOF':>4s} {'MediaPipe':>10s} {'hamer':>10s}")
        for i in range(16):
            print(f"  {i:>4d} {mp_angles[i]:>+10.3f} {h3_angles[i]:>+10.3f}")

        out = tracker.draw_landmarks(img, [hand])
        if hres is not None:
            out = _draw_hamer_overlay(out, h3d, hres, mp_pts)
        out_path = os.path.join(IMAGES_DIR, os.path.splitext(os.path.basename(path))[0] + "_compare.jpg")
        cv2.imwrite(out_path, out)
        print(f"  overlay: {out_path}")

    tracker.close()
