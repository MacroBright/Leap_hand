#!/usr/bin/env python3
"""Real-time demo: Camera → MediaPipe → hamer 3D MANO → JointMapper → LEAP.

Solves "3D fails when the hand tilts/rotates/clenches": hamer gives real
MANO 3D keypoints instead of MediaPipe's pseudo-3D z.

Controls:
    SPACE — calibrate zero-point (hold hand fully open)
    D     — toggle MANO 3D diagnostic overlay
    M     — toggle 3D source (hamer / MediaPipe pseudo-3D)
    S     — save gains
    Q/ESC — quit

Design: docs/design/2026-08-05-hamer-3d-integration-w1.md
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesture_mapping import HandTracker, JointMapper, Calibrator, FingerIdentifier
from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.camera import open_realsense
from gesture_mapping.hamer_3d import HaMeR3D, hand_bbox_from_landmarks
from gesture_mapping.demo_realtime import (
    _OpenCVCamera, find_best_camera, draw_hud, print_motor_mapping,
    print_angles_table,
)


# MANO skeleton connectivity (MediaPipe-index) for the 3D overlay
_KP_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _select_hand(results, hand: str):
    if hand == "first" or len(results) == 1:
        return results[0]
    for r in results:
        if r.handedness.lower() == hand:
            return r
    return results[0]


def _draw_hamer_overlay(frame, h3d, hres, mp_pts):
    """Draw projected MANO kp skeleton (yellow) + MediaPipe kp (green) for diagnosis."""
    kp2d = h3d.project_to_frame(hres, hres.kp3d)
    for a, b in _KP_CONN:
        cv2.line(frame,
                 (int(round(kp2d[a][0])), int(round(kp2d[a][1]))),
                 (int(round(kp2d[b][0])), int(round(kp2d[b][1]))),
                 (0, 255, 255), 2)
    for (x, y) in kp2d:
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 200, 255), -1)
    # MediaPipe kp (green) for direct comparison
    for i in range(21):
        cv2.circle(frame, (int(round(mp_pts[i][0])), int(round(mp_pts[i][1]))),
                   2, (0, 255, 0), -1)
    return frame


def run_image(path, tracker, h3d, mapper):
    """Single-image offline run: detect → regress → angles → save overlay."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] cannot read {path}")
        return
    h, w = frame.shape[:2]
    results = tracker.detect(frame)
    if not results:
        print("  (no hand detected)")
        return
    hand = results[0]
    mp_pts = tracker.landmark_xy(hand, (h, w))
    bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
    if bbox is None:
        print("  (bbox too small)")
        return
    hres = h3d.regress(frame, bbox)
    if hres is None:
        print("  (hamer regression failed)")
        return
    angles = mapper.map_points_to_leap(hres.kp3d)
    print(f"  kp3d finite={bool(np.isfinite(hres.kp3d).all())} "
          f"range=[{hres.kp3d.min():.3f}, {hres.kp3d.max():.3f}] (m)")
    print_angles_table(angles, None, {})
    out = tracker.draw_landmarks(frame, [hand])
    out = _draw_hamer_overlay(out, h3d, hres, mp_pts)
    out_path = Path(path).with_name(Path(path).stem + "_hamer3d.jpg")
    cv2.imwrite(str(out_path), out)
    print(f"  overlay saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera index (default: auto-detect)")
    parser.add_argument("--drive", action="store_true", help="Drive LEAP Hand hardware")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--skip", type=int, default=0,
                        help="run hamer every (skip+1) frames (0 = every frame)")
    parser.add_argument("--hand", type=str, default="first",
                        choices=["first", "right", "left"],
                        help="which MediaPipe hand to track")
    parser.add_argument("--img", type=str, default=None,
                        help="run on a single image and exit")
    args = parser.parse_args()

    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.5)
    h3d = HaMeR3D()
    if h3d.available:
        print("[INFO] HaMeR 3D ready (fp16, MANO regression)")
    else:
        print("[WARN] hamer unavailable (no CUDA / not installed) → MediaPipe pseudo-3D fallback")

    mapper = JointMapper()
    calibrator = Calibrator(mapper)
    finger_id = FingerIdentifier(mapper, bend_threshold=0.20)
    angle_filter = OneEuroFilter(n_joints=16, min_cutoff=1.0, beta=0.007)

    gain_path = Path(__file__).resolve().parent / "joint_gain.json"
    if gain_path.exists():
        mapper.load_gain_from(str(gain_path))

    JOINT_DIR = np.array([-1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1,  1, -1, -1, -1])

    leap = None
    if args.drive:
        from main import LeapNode, OPEN_POSE
        try:
            leap = LeapNode()
            print("[INFO] LEAP Hand connected.")
        except OSError as e:
            print(f"[WARN] Cannot connect: {e}")

    if args.img:
        run_image(args.img, tracker, h3d, mapper)
        tracker.close()
        if leap is not None:
            leap.disconnect()
        return

    cam = open_realsense()
    if cam is not None:
        print("[INFO] Using RealSense SDK color stream (pyrealsense2)")
    else:
        cam_idx = args.camera
        if cam_idx < 0:
            print("[INFO] Auto-detecting camera (OpenCV)...")
            cam_idx = find_best_camera()
            if cam_idx is None:
                print("[ERROR] No working camera found. Check USB connection.")
                tracker.close()
                return
            print(f"[INFO] Using camera index {cam_idx}")
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {cam_idx}")
            tracker.close()
            return
        cam = _OpenCVCamera(cap)

    print("\n" + "=" * 50)
    print("  LEAP Hand — hamer 3D Gesture Mapper")
    print("  SPACE=calib | D=diag | M=source | S=save | Q=quit")
    print("=" * 50)
    print_motor_mapping()

    print("[INFO] Warming up camera (RealSense auto-calibration ~3s)...")
    warm_t0 = time.time()
    while time.time() - warm_t0 < 3.0:
        cam.read()
    print("[INFO] Camera warm. Starting control loop.")

    frame_count = 0
    show_diag = False
    hamer_on = True
    last_hres = None

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)   # mirror
            h, w = frame.shape[:2]
            results = tracker.detect(frame)

            if results:
                hand = _select_hand(results, args.hand)
                mp_pts = tracker.landmark_xy(hand, (h, w))
                frame = tracker.draw_landmarks(frame, [hand])

                hres = None
                if h3d.available and hamer_on:
                    if frame_count % (args.skip + 1) == 0:
                        bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
                        if bbox is not None:
                            new_hres = h3d.regress(frame, bbox)
                            if new_hres is not None:
                                last_hres = new_hres
                    hres = last_hres

                if hres is not None:
                    pts = hres.kp3d
                    angles = calibrator.map_points(pts)
                    bent, scores = finger_id.identify_points(pts)
                    if show_diag:
                        frame = _draw_hamer_overlay(frame, h3d, hres, mp_pts)
                    source = "HAMER 3D"
                else:
                    angles = calibrator.map(hand, (h, w))
                    bent, scores = finger_id.identify(hand, (h, w))
                    source = "MP FALLBACK"

                angles = angle_filter(angles)

                if leap is not None:
                    from main import OPEN_POSE
                    leap.set_leap(OPEN_POSE + JOINT_DIR * angles)

                draw_hud(frame, angles, calibrator, bent, scores, show_diag)
                cv2.putText(frame, f"3D: {source}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255) if source == "HAMER 3D" else (0, 120, 255), 2)

                if frame_count % 20 == 0:
                    print_angles_table(angles, bent, scores)
            else:
                last_hres = None
                if frame_count % 30 == 0:
                    print("  (no hand detected)")
                if leap is not None:
                    leap.set_open()

            if not args.no_display:
                if frame_count == 0:
                    cv2.namedWindow("LEAP Hand — hamer 3D Mapper", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("LEAP Hand — hamer 3D Mapper", 960, 720)
                cv2.imshow("LEAP Hand — hamer 3D Mapper", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    if results:
                        if hres is not None:
                            baseline = calibrator.calibrate_points(hres.kp3d)
                        else:
                            baseline = calibrator.calibrate(hand, (h, w))
                        angle_filter.reset()
                        print(f"\n  *** CALIBRATED! baseline max: {baseline.max():.3f} rad ***\n")
                elif key == ord("d"):
                    show_diag = not show_diag
                    print(f"\n  Diagnostic overlay: {'ON' if show_diag else 'OFF'}\n")
                elif key == ord("m"):
                    hamer_on = not hamer_on
                    print(f"\n  3D source: {'hamer' if hamer_on else 'MediaPipe pseudo-3D'}\n")
                elif key == ord("s"):
                    mapper.save_gain(str(gain_path))

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        if leap is not None:
            leap.set_open()
            leap.disconnect()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Demo stopped.")


if __name__ == "__main__":
    main()
