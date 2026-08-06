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

# 3D 源: 0=hamer (真3D MANO, 逐帧重建, 慢但真3D)
#        1=MediaPipe world landmarks (规范3D 手模型, 米制, 相机帧率, 稳)
#        2=MediaPipe 伪3D (原 z)
_SOURCE_NAMES = {0: "HAMER 3D", 1: "WORLD 3D", 2: "MP PSEUDO-3D"}

# 帧级质量门控: 平均关键点可见度低于此阈值 → 坏帧, 保持上一帧好角度 (不外推重建)
_MIN_VIS = 0.55


def _frame_quality(hand) -> float:
    """0-1 平均关键点可见度; 无 visibility 字段时视为好帧 (1.0)."""
    try:
        vis = [lm.visibility for lm in hand.landmarks]
        return float(np.mean(vis)) if vis else 1.0
    except (AttributeError, TypeError):
        return 1.0


def _smoothed_frame(pts, smoother):
    """计算掌心参考系并对 normal/mid_dir/lateral 做时域平滑.

    参考系逐帧抖动/翻转是"张开手移动时 fan 角跳变"的主因;
    平滑后传入 map_points_to_leap 的 frame 参数即可稳定侧摆角。
    """
    wrist, normal, mid_dir, lateral = JointMapper._palm_frame(pts)
    fvec = smoother(np.concatenate([normal, mid_dir, lateral]))
    normal, mid_dir, lateral = fvec[:3], fvec[3:6], fvec[6:9]
    for v in (normal, mid_dir):
        n = np.linalg.norm(v)
        if n > 1e-9:
            v /= n
    lateral = lateral - np.dot(lateral, mid_dir) * mid_dir
    n = np.linalg.norm(lateral)
    if n > 1e-9:
        lateral /= n
    else:
        lateral = np.cross(normal, mid_dir)
        n = np.linalg.norm(lateral)
        if n > 1e-9:
            lateral /= n
    return (wrist, normal, mid_dir, lateral)


# 手丢失时的位姿过渡: 短时保持(避免闪烁抖动) → 平滑回到 OPEN(安全中性位)
_RELAX_HOLD = 0.30   # s
_RELAX_TIME = 0.60   # s


def _relax_pose(now, loss_t0, relax_from, last_commanded_pose, open_pose, motor_limits):
    """手丢失时计算过渡位姿. 返回 (pose, loss_t0, relax_from)."""
    if loss_t0 is None:
        loss_t0 = now
        relax_from = (last_commanded_pose.copy() if last_commanded_pose is not None
                      else open_pose.copy())
    elapsed = now - loss_t0
    if elapsed < _RELAX_HOLD:
        pose = relax_from          # 短时丢失: 保持上一帧, 不平滑则不动
    else:
        t = min(1.0, (elapsed - _RELAX_HOLD) / _RELAX_TIME)
        pose = relax_from + (open_pose - relax_from) * t   # 线性平滑回 OPEN
    if motor_limits is not None:
        pose = np.clip(pose, motor_limits[0], motor_limits[1])
    return pose, loss_t0, relax_from


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
    parser.add_argument("--sim", action="store_true",
                        help="Drive the MuJoCo simulated hand instead of hardware (with --drive)")
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
    print("[INFO] 默认 3D 源: MediaPipe world-3D (按 M 循环: world-3D → hamer → 伪3D)")

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

    # 持久化张开基线: 上次 SPACE 校准结果自动加载, 免去每次会话重新调 0
    calib_path = Path(__file__).resolve().parent / "calibration_3d.json"
    if calibrator.load_points_baseline(str(calib_path)):
        print(f"[INFO] Loaded 3D calibration baseline: {calib_path}")

    # 实测电机限位表 (Task D): 若存在, --drive 写入前裁剪到机械范围.
    # 格式: {"min": [16], "max": [16]} (rad, 伺服真实位置). 无表 → 不裁剪 (兼容旧行为).
    motor_limits = None
    limits_path = Path(__file__).resolve().parent / "motor_limits.json"
    if limits_path.exists():
        import json
        with open(limits_path) as f:
            _lim = json.load(f)
        motor_limits = (np.array(_lim["min"], dtype=np.float64),
                        np.array(_lim["max"], dtype=np.float64))
        print(f"[INFO] Motor limits loaded: {limits_path}")

    JOINT_DIR = np.array([-1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1,  1, -1, -1, -1])

    leap = None
    if args.drive:
        from main import OPEN_POSE
        if args.sim:
            from sim.sim_leap import SimLeap
            try:
                leap = SimLeap()
                print("[INFO] MuJoCo simulation hand ready (--sim).")
            except Exception as e:
                print(f"[WARN] Sim init failed: {e}")
        else:
            # 真机安全: 延迟上电 — 张开手对准相机, 按 SPACE 完成全开校准后才连接上电
            # (防止未校准就上电时关节角度不对, 损伤电机)
            print("[INFO] 真机模式: 电机未上电。张开手对准相机, 按 SPACE 校准后上电驱动。")

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
    source_mode = 1              # 0=hamer, 1=MediaPipe world-3D (默认), 2=伪3D (M 循环)
    last_hres = None
    last_commanded_pose = None   # 实际发送的绝对位姿 (丢失时平滑回 OPEN 的起点)
    loss_t0 = None               # 手丢失时刻 (monotonic)
    relax_from = None            # 丢失时从哪个位姿开始回 OPEN
    last_good = None            # (angles, bent, scores, source) 坏帧保持
    smoothed_kp = None          # last OneEuro-smoothed kp3d (angles + calibration source)
    kp_smoother = OneEuroFilter(n_joints=63, min_cutoff=0.8, beta=0.005)
    world_smoother = OneEuroFilter(n_joints=63, min_cutoff=1.2, beta=0.004)
    frame_smoother = OneEuroFilter(n_joints=9, min_cutoff=1.0, beta=0.005)
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
                # hamer 需右-MANO 门控; world/pseudo 无此风险 → 无匹配时退回首只检测到手
                if source_mode == 0:
                    hand = _select_hand(results, args.hand)
                else:
                    hand = _select_hand(results, args.hand) or results[0]
                hres = None
                if hand is None:
                    # no hand matching the configured handedness → no-hand branch
                    last_hres = None
                    last_good = None
                    smoothed_kp = None
                    bbox_ema = None
                    world_smoother.reset()
                    frame_smoother.reset()
                    if frame_count % 30 == 0:
                        print(f"  (no {args.hand} hand detected)")
                    if leap is not None:
                        from main import OPEN_POSE
                        pose, loss_t0, relax_from = _relax_pose(
                            time.monotonic(), loss_t0, relax_from,
                            last_commanded_pose, OPEN_POSE, motor_limits)
                        leap.set_leap(pose)
                else:
                    loss_t0 = None          # 有手可用 → 重置丢失回退状态
                    relax_from = None
                    mp_pts = tracker.landmark_xy(hand, (h, w))
                    frame = tracker.draw_landmarks(frame, [hand])
                    quality = _frame_quality(hand)

                    # Gate hamer on the configured hand: the right-MANO model must
                    # never see a left-hand crop (silently mirrored/wrong angles).
                    # "first" is a legacy escape hatch that skips the handedness gate.
                    run_hamer = (args.hand == "first"
                                 or hand.handedness.lower() == _MIRRORED_LABEL.get(args.hand, args.hand))

                    # 坏帧 (低可见度) 不喂 hamer, 避免把劣质裁剪写进 last_hres
                    hres = None
                    if run_hamer and h3d.available and source_mode == 0 and quality >= _MIN_VIS:
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

                    if quality < _MIN_VIS and last_good is not None:
                        # 坏帧: 保持上一帧好角度, 不外推重建 (绿色关键点仍显示)
                        angles, bent, scores, source = last_good
                    else:
                        if source_mode == 0 and hres is not None:
                            pts = smoothed_kp = kp_smoother(hres.kp3d.reshape(-1)).reshape(21, 3)
                            palm_frame = _smoothed_frame(pts, frame_smoother)
                            angles = calibrator.map_points(pts, frame=palm_frame)
                            bent, scores = finger_id.identify_points(pts)
                            if show_diag:
                                frame = _draw_hamer_overlay(frame, h3d, hres, mp_pts, pts3d=pts)
                            source = _SOURCE_NAMES[0]
                        elif source_mode == 1 and hand.world_landmarks is not None:
                            # MediaPipe 规范 3D 手模型 (米制, 相机帧率, 无需 hamer 推理)
                            wpts = np.array(
                                [[lm.x, lm.y, lm.z] for lm in hand.world_landmarks],
                                dtype=np.float64)
                            pts = smoothed_kp = world_smoother(wpts.reshape(-1)).reshape(21, 3)
                            palm_frame = _smoothed_frame(pts, frame_smoother)
                            angles = calibrator.map_points(pts, frame=palm_frame)
                            bent, scores = finger_id.identify_points(pts)
                            source = _SOURCE_NAMES[1]
                        else:
                            # pseudo-3D 源: 角度来自 MediaPipe 伪 z, 校准用 HandResult 路径
                            smoothed_kp = None
                            angles = calibrator.map(hand, (h, w))
                            bent, scores = finger_id.identify(hand, (h, w))
                            source = _SOURCE_NAMES[2]

                        angles = angle_filter(angles)
                        last_good = (angles, bent, scores, source)

                    if leap is not None:
                        from main import OPEN_POSE
                        pose = OPEN_POSE + JOINT_DIR * angles
                        if motor_limits is not None:
                            pose = np.clip(pose, motor_limits[0], motor_limits[1])
                        leap.set_leap(pose)
                        last_commanded_pose = pose

                    draw_hud(frame, angles, calibrator, bent, scores, show_diag)
                    cv2.putText(frame, f"3D: {source}   {fps:4.0f} fps",
                                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 255, 255) if source == "HAMER 3D" else (0, 120, 255), 2)

                    if frame_count % 20 == 0:
                        print_angles_table(angles, bent, scores)
            else:
                last_hres = None
                last_good = None
                smoothed_kp = None
                bbox_ema = None
                world_smoother.reset()
                frame_smoother.reset()
                if frame_count % 30 == 0:
                    print("  (no hand detected)")
                if leap is not None:
                    from main import OPEN_POSE
                    pose, loss_t0, relax_from = _relax_pose(
                        time.monotonic(), loss_t0, relax_from,
                        last_commanded_pose, OPEN_POSE, motor_limits)
                    leap.set_leap(pose)

            if not args.no_display:
                if args.drive and not args.sim and leap is None:
                    # 真机未上电提示: 张开手按 SPACE 校准后上电
                    cv2.putText(frame, "SPACE 未上电: 张开手按 SPACE 校准并上电",
                                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 255), 2)
                if frame_count == 0:
                    cv2.namedWindow("LEAP Hand — hamer 3D Mapper", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("LEAP Hand — hamer 3D Mapper", 960, 720)
                cv2.imshow("LEAP Hand — hamer 3D Mapper", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    if results and hand is not None:
                        # 真机延迟上电: 首次按 SPACE (张开手校准) 时才连接上电
                        if args.drive and not args.sim and leap is None:
                            from main import LeapNode
                            try:
                                leap = LeapNode()
                                print("[INFO] LEAP Hand 已上电 (全开位)。")
                            except OSError as e:
                                print(f"[WARN] 上电失败: {e}")
                        # 校准必须用"本帧实际驱动角度的点源": hamer/world 用 points 基线,
                        # 伪 3D 用 HandResult 基线 (两套基线槽位分离, 不可混用)
                        if smoothed_kp is not None:
                            palm_frame = _smoothed_frame(smoothed_kp, frame_smoother)
                            baseline = calibrator.calibrate_points(smoothed_kp, frame=palm_frame)
                            calibrator.save_points_baseline(str(calib_path))
                        else:
                            baseline = calibrator.calibrate(hand, (h, w))
                        angle_filter.reset()
                        kp_smoother.reset()
                        world_smoother.reset()
                        frame_smoother.reset()
                        print(f"\n  *** CALIBRATED! baseline max: {baseline.max():.3f} rad ***\n")
                elif key == ord("d"):
                    show_diag = not show_diag
                    print(f"\n  Diagnostic overlay: {'ON' if show_diag else 'OFF'}\n")
                elif key == ord("m"):
                    source_mode = (source_mode + 1) % 3
                    print(f"\n  3D source: {_SOURCE_NAMES[source_mode]}\n")
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
