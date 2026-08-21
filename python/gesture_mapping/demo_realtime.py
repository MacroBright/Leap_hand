#!/usr/bin/env python3
"""Real-time demo: Camera → Hands → JointMapper → display + calibration + finger ID.

Controls:
    SPACE  — calibrate zero-point (hold hand fully open)
    q / ESC — quit

HUD shows:
    - LEAP 16-DOF angles with human→LEAP group labels
    - Calibration status
    - Which finger is currently bent

Workstream: W1 手势映射 — .claude/workstreams/01-gesture-mapping.md
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesture_mapping import HandTracker, JointMapper, Calibrator, FingerIdentifier
from gesture_mapping.calibrator import LEAP_GROUP_LABELS, HUMAN_FINGER_LABELS
from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.camera import open_realsense


# Motor ID → landmark mapping (for HUD overlay)
_MOTOR_DIAG = [
    # ID, finger, joint, landmarks used
    ( 0, "Idx", "Abd", "MCP→PIP vs palm"),
    ( 1, "Idx", "MCP", "Wrist→MCP→PIP"),
    ( 2, "Idx", "PIP", "MCP→PIP→DIP"),
    ( 3, "Idx", "DIP", "PIP→DIP→TIP"),
    ( 4, "Mid", "Abd", "MCP→PIP vs palm"),
    ( 5, "Mid", "MCP", "Wrist→MCP→PIP"),
    ( 6, "Mid", "PIP", "MCP→PIP→DIP"),
    ( 7, "Mid", "DIP", "PIP→DIP→TIP"),
    ( 8, "Pky", "Abd", "MCP→PIP vs palm"),
    ( 9, "Pky", "MCP", "Wrist→MCP→PIP"),
    (10, "Pky", "PIP", "MCP→PIP→DIP"),
    (11, "Pky", "DIP", "PIP→DIP→TIP"),
    (12, "Thb", "Abd", "CMC→MCP vs palm"),
    (13, "Thb", "MCP", "Wrist→CMC→MCP"),
    (14, "Thb", "PIP", "CMC→MCP→IP"),
    (15, "Thb", "DIP", "MCP→IP→TIP"),
]


class _OpenCVCamera:
    """Wrap an OpenCV VideoCapture to the CameraSource read()/release() interface."""

    def __init__(self, cap):
        self._cap = cap

    def read(self):
        ok, frame = self._cap.read()
        return ok, frame

    def release(self):
        self._cap.release()


def _probe_frame(idx: int):
    """Open a camera index and return a stable frame, or None."""
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None and frame.size > 0:
        return frame
    return None


def _color_score(frame) -> float:
    """Score how 'colorful' a frame is. Real RGB ≫ grayscale/depth/IR.

    RealSense port enumeration is UNSTABLE — the same index can expose a
    different stream type on each run. So instead of trusting a fixed index,
    we scan every port each launch and pick the most colorful stream.
    """
    if frame is None:
        return -1.0
    ch_spread = max(np.abs(frame[:, :, 0] - frame[:, :, 1]).mean(),
                    np.abs(frame[:, :, 0] - frame[:, :, 2]).mean(),
                    np.abs(frame[:, :, 1] - frame[:, :, 2]).mean())
    return float(ch_spread)


def find_best_camera(max_index: int = 10) -> Optional[int]:
    """Return the best camera index by scanning all ports each launch.

    RealSense D455 enumerates its color/depth/IR nodes dynamically — the
    index that was color last run may be a depth or grayscale feed now. So:
      1. Probe every index 0..max_index
      2. Keep the one with the highest color score (real RGB ≫ depth/IR)
      3. Persist the winner to camera_pref.json so a single explicit port
         can be used next time (--camera) without re-scanning.

    Returns:
        Camera index, or None if no port yields a color frame.
    """
    pref_path = Path(__file__).resolve().parent / "camera_pref.json"

    # 1) Probe the saved preference first (cheap, usually right)
    best_idx, best_score = None, -1.0
    if pref_path.exists():
        import json
        try:
            with open(pref_path) as f:
                saved = json.load(f).get("index")
            if saved is not None:
                best_idx, best_score = int(saved), _color_score(_probe_frame(saved))
        except Exception:
            pass

    # 2) Full scan — pick the most colorful stream (beats stale pref)
    for idx in range(max_index):
        if idx == best_idx:
            continue
        score = _color_score(_probe_frame(idx))
        if score > best_score:
            best_idx, best_score = idx, score

    # Only accept if it's a real color stream (well above depth/IR noise floor)
    if best_idx is None or best_score < 10.0:
        return None

    # 3) Persist so a future --camera or next run starts from the right port
    try:
        with open(pref_path, "w") as f:
            import json
            json.dump({"index": best_idx}, f)
    except Exception:
        pass
    return best_idx


def print_motor_mapping():
    """Print motor ID → landmark mapping to terminal (called on startup)."""
    print("\n  MOTOR ID → MEDIAPIPE LANDMARK MAPPING")
    print("  " + "=" * 58)
    print(f"  {'ID':>3s}  {'Finger':>6s}  {'Joint':>4s}  Landmarks used")
    print("  " + "-" * 58)
    for id_, finger, joint, desc in _MOTOR_DIAG:
        print(f"  {id_:>3d}  {finger:>6s}  {joint:>4s}  {desc}")


def draw_hud(image, angles, calibrator, bent_finger, bend_scores, show_diag=False,
             cur_joint=None, joint_gain=None):
    """Semi-transparent HUD: angle table + cal status + finger ID + gain status."""
    panel_w = 520 if show_diag else 430
    panel_h = 240 if not show_diag else 360
    panel = image.copy()
    cv2.rectangle(panel, (5, 5), (panel_w, panel_h), (30, 30, 30), -1)
    image[:] = cv2.addWeighted(image, 0.55, panel, 0.45, 0)

    y = 22
    title = "ANGLES (calibrated)" if calibrator.is_calibrated else "ANGLES (raw)"
    cv2.putText(image, title, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.putText(image, f"{'':20s}  {'Abd':>6s}  {'MCP':>6s}  {'PIP':>6s}  {'DIP':>6s}",
                (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    for i, label in enumerate(LEAP_GROUP_LABELS):
        s = i * 4
        row_y = 54 + i * 16
        color = (0, 240, 0) if angles[s + 1] < 1.0 else (0, 200, 255)
        # Highlight the currently-selected joint in this group
        if cur_joint is not None and s <= cur_joint < s + 4:
            color = (255, 200, 0)
        cv2.putText(image, f"{label:20s} {angles[s]:+6.2f}  {angles[s+1]:+6.2f}  {angles[s+2]:+6.2f}  {angles[s+3]:+6.2f}",
                    (12, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    y = 130
    cal_text = "CALIBRATED" if calibrator.is_calibrated else "NOT CALIBRATED"
    cal_color = (0, 255, 0) if calibrator.is_calibrated else (0, 140, 255)
    cv2.putText(image, cal_text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, cal_color, 1)
    cv2.putText(image, "SPACE=calib | D=diag | TAB=joint | [ ]=gain | S=save | Q=quit",
                (12, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (120, 120, 120), 1)

    y = 175
    if bent_finger:
        label = HUMAN_FINGER_LABELS.get(bent_finger, bent_finger)
        score = bend_scores.get(bent_finger, 0.0)
        cv2.putText(image, f"BENT: {label}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)
        cv2.putText(image, f"score={score:.3f}", (260, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255), 1)
    else:
        cv2.putText(image, "BENT: (none)", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)

    # Gain tuning status line
    if cur_joint is not None and joint_gain is not None:
        try:
            _, finger, jnt, _ = _MOTOR_DIAG[cur_joint]
        except IndexError:
            finger, jnt = "?", "?"
        cv2.putText(image, f"▶ JOINT {cur_joint} ({finger} {jnt}) gain={joint_gain[cur_joint]:.2f}",
                    (12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

    # Diagnostic: motor ID → landmark mapping
    if show_diag:
        dy = 200
        cv2.putText(image, "MOTOR ID → LANDMARK DIAGNOSTIC", (12, dy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 0), 1)
        cv2.putText(image, f"{'ID':>3s} {'Finger':>6s} {'Jnt':>4s}  Landmarks{'':>6s}  {'Angle':>8s}",
                    (12, dy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)
        for id_, finger, joint, desc in _MOTOR_DIAG:
            row_y = dy + 32 + id_ * 9
            cv2.putText(image,
                        f"{id_:>3d} {finger:>6s} {joint:>4s}  {desc:20s}  {angles[id_]:+8.4f}",
                        (12, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.28,
                        (0, 255, 0) if abs(angles[id_]) < 0.5 else (0, 255, 255), 1)
        cv2.putText(image, "BENT: (none — all fingers straight)",
                    (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)


def print_angles_table(angles, bent_finger, bend_scores):
    """Terminal output with finger bend scores."""
    human_map = {"index": 0, "middle": 1, "pinky": 2, "thumb": 3}
    print("\n" + "=" * 70)
    for i, label in enumerate(LEAP_GROUP_LABELS):
        s = i * 4
        hf = ["index", "middle", "pinky", "thumb"][i]
        sc = bend_scores.get(hf, 0.0)
        marker = " <<<" if hf == bent_finger else ""
        # 拇指电机顺序: 12=mcp, 13=side (与其他指相反)
        if hf == "thumb":
            print(f"  {label:20s} mcp={angles[s+0]:+.3f} abd={angles[s+1]:+.3f}"
                  f" pip={angles[s+2]:+.3f} dip={angles[s+3]:+.3f}  [{sc:.3f}]{marker}")
        else:
            print(f"  {label:20s} abd={angles[s+0]:+.3f} mcp={angles[s+1]:+.3f}"
                  f" pip={angles[s+2]:+.3f} dip={angles[s+3]:+.3f}  [{sc:.3f}]{marker}")
    if bent_finger:
        print(f"  >>> BENT: {HUMAN_FINGER_LABELS.get(bent_finger, bent_finger)} <<<")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera index (default: auto-detect)")
    drive_group = parser.add_mutually_exclusive_group()
    drive_group.add_argument(
        "--drive", action="store_true",
        help="Drive LEAP Hand hardware with legacy settings")
    drive_group.add_argument(
        "--safe-drive", action="store_true",
        help="Drive LEAP Hand hardware with low-force safety settings")
    parser.add_argument("--no-display", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.5)
    mapper = JointMapper()
    calibrator = Calibrator(mapper)
    finger_id = FingerIdentifier(mapper, bend_threshold=0.20)
    angle_filter = OneEuroFilter(n_joints=16, min_cutoff=1.0, beta=0.007)

    # Load saved per-joint gains (persisted from a previous tuning session)
    gain_path = Path(__file__).resolve().parent / "joint_gain.json"
    if gain_path.exists():
        mapper.load_gain_from(str(gain_path))

    print("\n" + "=" * 50)
    print("  LEAP Hand — Gesture Mapping + Finger ID")
    print("  SPACE=calib | D=diag | TAB=joint | [ ]=gain | S=save | Q=quit")
    print("=" * 50)
    print_motor_mapping()

    # Joint direction correction (2026-08-04 真机实测)
    #   mcp/pip/dip: 负值弯向手心 (thumb mcp 例外: +1 弯向手心)
    #   index/middle/ring side: 负值向拇指方向
    # ⚠ 拇指电机顺序: 12=mcp, 13=side → 拇指段 [1, -1, -1, -1]
    JOINT_DIR = np.array([-1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  1, -1, -1, -1])

    leap = None
    safe_leap = None
    if args.safe_drive:
        from leap_hand_utils.dynamixel_client import DynamixelClient
        from main import OPEN_POSE
        from gesture_mapping.safe_leap_controller import SafeLeapController

        port = ("/dev/serial/by-id/"
                "usb-FTDI_USB__-__Serial_Converter_FTB8HNYU-if00-port0")
        client = DynamixelClient(list(range(16)), port, 4000000)
        safe_leap = SafeLeapController(client, OPEN_POSE)
    elif args.drive:
        from main import LeapNode, OPEN_POSE
        try:
            leap = LeapNode()
            print("[INFO] LEAP Hand connected.")
        except OSError as e:
            print(f"[WARN] Cannot connect: {e}")

    # Camera: prefer official RealSense SDK (explicit color stream), else
    # fall back to OpenCV port scan.
    cam = open_realsense()  # None if no pyrealsense2 / no D455 present
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
        # wrap OpenCV cap to the same read()/release() interface
        cam = _OpenCVCamera(cap)

    # Warm up: RealSense depth cameras auto-calibrate for ~3s after open,
    # during which frame reads are very slow (~10fps). Wait it out BEFORE
    # the control loop so the hand doesn't move jerkily at startup.
    print("[INFO] Warming up camera (RealSense auto-calibration ~3s)...")
    warm_t0 = time.time()
    while time.time() - warm_t0 < 3.0:
        cam.read()
    print("[INFO] Camera warm. Starting control loop.")

    # Safe drive starts only after the camera is confirmed working. Camera
    # setup failures above therefore cannot leave motor torque enabled.
    if safe_leap is not None:
        try:
            safe_leap.start()
            print("[INFO] LEAP Hand connected in low-force safety mode.")
        except Exception as e:
            print(f"[WARN] Cannot start safe drive: {e}")
            safe_leap = None

    frame_count = 0
    show_diag = False
    cur_joint = 0  # currently-selected joint for gain tuning (0-15)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = tracker.detect(frame)

            if results:
                hand = results[0]
                angles = calibrator.map(hand, (h, w))
                angles = angle_filter(angles)  # temporal smoothing
                bent, scores = finger_id.identify(hand, (h, w))

                if safe_leap is not None:
                    from main import OPEN_POSE
                    import leap_hand_utils.leap_hand_utils as lhu
                    target = lhu.angle_safety_clip(
                        OPEN_POSE + JOINT_DIR * angles)
                    safe_leap.track(target)
                elif leap is not None:
                    from main import OPEN_POSE
                    leap.set_leap(OPEN_POSE + JOINT_DIR * angles)

                frame = tracker.draw_landmarks(frame, results)
                draw_hud(frame, angles, calibrator, bent, scores, show_diag,
                         cur_joint=cur_joint, joint_gain=mapper.joint_gain)

                if frame_count % 20 == 0:
                    print_angles_table(angles, bent, scores)

            else:
                if frame_count % 30 == 0:
                    print("  (no hand detected)")
                if safe_leap is not None:
                    safe_leap.on_tracking_lost()
                elif leap is not None:
                    leap.set_open()

            if not args.no_display:
                if frame_count == 0:
                    # Make the window larger & resizable (default 640x480 is small)
                    cv2.namedWindow("LEAP Hand — Gesture Mapper", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("LEAP Hand — Gesture Mapper", 960, 720)
                cv2.imshow("LEAP Hand — Gesture Mapper", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    if results:
                        baseline = calibrator.calibrate(results[0], (h, w))
                        angle_filter.reset()
                        print(f"\n  *** CALIBRATED! baseline max: {baseline.max():.3f} rad ***\n")
                elif key == ord("d"):
                    show_diag = not show_diag
                    print(f"\n  Diagnostic overlay: {'ON' if show_diag else 'OFF'}\n")
                elif key == 9:  # Tab — cycle through joint 0-15
                    cur_joint = (cur_joint + 1) % 16
                    try:
                        _, f, j, _ = _MOTOR_DIAG[cur_joint]
                    except IndexError:
                        f, j = "?", "?"
                    print(f"\n  ▶ Joint {cur_joint} ({f} {j}) gain={mapper.joint_gain[cur_joint]:.2f}\n")
                elif key in (ord("["), ord("]")):  # adjust gain
                    delta = -0.05 if key == ord("[") else 0.05
                    mapper.joint_gain[cur_joint] += delta
                    try:
                        _, f, j, _ = _MOTOR_DIAG[cur_joint]
                    except IndexError:
                        f, j = "?", "?"
                    print(f"  Joint {cur_joint} ({f} {j}) gain={mapper.joint_gain[cur_joint]:+.2f}")
                elif key == ord("s"):  # save gains
                    mapper.save_gain(str(gain_path))
                elif key == ord("r"):  # reset current joint gain to 1.0
                    mapper.joint_gain[cur_joint] = 1.0
                    try:
                        _, f, j, _ = _MOTOR_DIAG[cur_joint]
                    except IndexError:
                        f, j = "?", "?"
                    print(f"  Joint {cur_joint} ({f} {j}) gain reset → 1.00")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        if safe_leap is not None:
            safe_leap.shutdown(return_open=True)
        elif leap is not None:
            leap.set_open()
            leap.disconnect()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Demo stopped.")


if __name__ == "__main__":
    main()
