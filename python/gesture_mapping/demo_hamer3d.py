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
    _OpenCVCamera, _MOTOR_DIAG, find_best_camera, draw_hud,
    print_motor_mapping, print_angles_table,
)


# MANO skeleton connectivity (MediaPipe-index) for the 3D overlay
_KP_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# The frame is mirrored (cv2.flip(frame, 1)) before detection, so MediaPipe's
# handedness label is the OPPOSITE of the user's physical hand: a physical right
# hand appears as MediaPipe "Left". --hand refers to the USER's physical hand;
# translate it to the mirrored label here.
_MIRRORED_LABEL = {"right": "left", "left": "right"}


def _select_hand(results, hand: str):
    """Select the tracked hand. `hand` is the USER's physical hand ("right" /
    "left"); due to the mirror flip it is matched against the opposite MediaPipe
    label. "first" tracks the first detected hand (no handedness gate)."""
    if hand == "first":
        return results[0]
    target = _MIRRORED_LABEL.get(hand, hand)
    for r in results:
        if r.handedness.lower() == target:
            return r
    return None


def _draw_hamer_overlay(frame, h3d, hres, mp_pts, pts3d=None):
    """Draw projected MANO kp skeleton (yellow) + MediaPipe kp (green) for diagnosis."""
    kp2d = h3d.project_to_frame(hres, hres.kp3d if pts3d is None else pts3d)
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
    parser.add_argument("--skip", type=int, default=1,
                        help="run hamer every (skip+1) frames (1 = every 2nd; "
                             "keeps MediaPipe at camera rate for smooth keypoints)")
    parser.add_argument("--hand", type=str, default="right",
                        choices=["first", "right", "left"],
                        help="which PHYSICAL hand to track (default 'right' = your "
                             "right hand, for the LEAP right hand). The frame is "
                             "mirrored (cv2.flip), so a physical right hand is "
                             "MediaPipe 'Left' — this flag handles the inversion. "
                             "'first' = first detected hand, no gate")
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

    # hamer 的 3D 弯曲角本身已准确, 不再需要 MediaPipe 伪 3D 的 1.5× 放大。
    # 用独立增益文件, 默认恒等 (1.0), 供实时调参后保存 (与 demo_realtime 互不影响)。
    gain_path = Path(__file__).resolve().parent / "joint_gain_3d.json"
    if gain_path.exists():
        mapper.load_gain_from(str(gain_path))
    else:
        mapper.joint_gain = np.ones(16)

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
    print("  SPACE=calib | D=diag | M=source | TAB=gain | [ ]=± | R=reset | S=save | Q=quit")
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
    smoothed_kp = None          # last OneEuro-smoothed kp3d (angles + calibration source)
    kp_smoother = OneEuroFilter(n_joints=63, min_cutoff=0.8, beta=0.005)
    bbox_ema = None             # (cx, cy, size) EMA → stable hamer crop across frames
    prev_time = time.monotonic()
    fps = 0.0
    cur_joint = 0

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            dt = now - prev_time
            prev_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

            frame = cv2.flip(frame, 1)   # mirror
            h, w = frame.shape[:2]
            results = tracker.detect(frame)

            if results:
                hand = _select_hand(results, args.hand)
                hres = None
                if hand is None:
                    # no hand matching the configured handedness → no-hand branch
                    last_hres = None
                    smoothed_kp = None
                    bbox_ema = None
                    if frame_count % 30 == 0:
                        print(f"  (no {args.hand} hand detected)")
                    if leap is not None:
                        leap.set_open()
                else:
                    mp_pts = tracker.landmark_xy(hand, (h, w))
                    frame = tracker.draw_landmarks(frame, [hand])

                    # Gate hamer on the configured hand: the right-MANO model must
                    # never see a left-hand crop (silently mirrored/wrong angles).
                    # "first" is a legacy escape hatch that skips the handedness gate.
                    run_hamer = (args.hand == "first"
                                 or hand.handedness.lower() == _MIRRORED_LABEL.get(args.hand, args.hand))

                    hres = None
                    if run_hamer and h3d.available and hamer_on:
                        if frame_count % (args.skip + 1) == 0:
                            bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
                            if bbox is not None:
                                # EMA the bbox center+size so consecutive hamer crops
                                # stay consistent (MediaPipe landmark jitter → stable 3D)
                                cx = (bbox[0] + bbox[2]) / 2.0
                                cy = (bbox[1] + bbox[3]) / 2.0
                                sz = bbox[2] - bbox[0]
                                if bbox_ema is None:
                                    bbox_ema = np.array([cx, cy, sz])
                                else:
                                    bbox_ema = 0.5 * bbox_ema + 0.5 * np.array([cx, cy, sz])
                                half = bbox_ema[2] / 2.0
                                eb = (int(round(bbox_ema[0] - half)),
                                      int(round(bbox_ema[1] - half)),
                                      int(round(bbox_ema[0] + half)),
                                      int(round(bbox_ema[1] + half)))
                                new_hres = h3d.regress(frame, eb)
                                if new_hres is not None:
                                    last_hres = new_hres
                        hres = last_hres

                    if hres is not None:
                        pts = smoothed_kp = kp_smoother(hres.kp3d)
                        angles = calibrator.map_points(pts)
                        bent, scores = finger_id.identify_points(pts)
                        if show_diag:
                            frame = _draw_hamer_overlay(frame, h3d, hres, mp_pts, pts3d=pts)
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
                    cv2.putText(frame, f"3D: {source}   {fps:4.0f} fps",
                                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255) if source == "HAMER 3D" else (0, 120, 255), 2)

                    if frame_count % 20 == 0:
                        print_angles_table(angles, bent, scores)
            else:
                last_hres = None
                smoothed_kp = None
                bbox_ema = None
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
                    if results and hand is not None:
                        if hres is not None:
                            baseline = calibrator.calibrate_points(
                                smoothed_kp if smoothed_kp is not None else hres.kp3d)
                        else:
                            baseline = calibrator.calibrate(hand, (h, w))
                        angle_filter.reset()
                        kp_smoother.reset()
                        print(f"\n  *** CALIBRATED! baseline max: {baseline.max():.3f} rad ***\n")
                elif key == ord("d"):
                    show_diag = not show_diag
                    print(f"\n  Diagnostic overlay: {'ON' if show_diag else 'OFF'}\n")
                elif key == ord("m"):
                    hamer_on = not hamer_on
                    print(f"\n  3D source: {'hamer' if hamer_on else 'MediaPipe pseudo-3D'}\n")
                elif key == 9:  # Tab — cycle joint 0-15 for gain tuning
                    cur_joint = (cur_joint + 1) % 16
                    _, f, j, _ = _MOTOR_DIAG[cur_joint]
                    print(f"  ▶ Joint {cur_joint} ({f} {j}) gain={mapper.joint_gain[cur_joint]:.2f}")
                elif key in (ord("["), ord("]")):
                    delta = -0.05 if key == ord("[") else 0.05
                    mapper.joint_gain[cur_joint] += delta
                    _, f, j, _ = _MOTOR_DIAG[cur_joint]
                    print(f"  Joint {cur_joint} ({f} {j}) gain={mapper.joint_gain[cur_joint]:+.2f}")
                elif key == ord("r"):
                    mapper.joint_gain[cur_joint] = 1.0
                    _, f, j, _ = _MOTOR_DIAG[cur_joint]
                    print(f"  Joint {cur_joint} ({f} {j}) gain reset → 1.00")
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
