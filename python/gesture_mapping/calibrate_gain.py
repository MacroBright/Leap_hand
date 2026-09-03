#!/usr/bin/env python3
"""数据驱动增益标定 — 人手 HUD 角度(a) ↔ 灵巧手标准姿势期望(b) → 拟合每关节 gain.

原理:
  对每个标准姿势 P (poses.json, 须先用 calibrate.py 重新录制可靠数据):
    - 按 N 选姿势, 若 --drive 则灵巧手驱动到该姿势作为对照
    - 人手摆出对应姿势, HUD 显示映射角度 a (gain 临时=1.0, 减 SPACE 校准基线)
    - 期望 b = JOINT_DIR * (P - OPEN_POSE)   (灵巧手相对全开的屈曲量)
  采集 ≥5 个姿势后按 W: 每关节过原点最小二乘拟合 b ≈ gain * a, 写 joint_gain_3d.json
  (demo_hamer3d.py 启动自动加载该增益文件)

用法 (先完成 calibrate.py 校准+录制, 再跑本脚本):
  conda activate leap_hand
  cd ~/office/Leap_Hand/python
  python gesture_mapping/calibrate_gain.py --drive [--camera N]

按键:
  SPACE — 张开手校准基线 (必须先做)
  N     — 选下一个标定姿势 (灵巧手驱动到它, 提示你摆人手对应姿势)
  G     — 重新驱动灵巧手到当前姿势 (对照)
  S     — 采集当前人手 HUD 角度 (a), 记录 (a, b) 对
  D     — 显示已采集的姿势对
  W     — 拟合并写 joint_gain_3d.json
  Q/ESC — 退出

数据文件:
  读 poses.json (标准姿势电机位 → b), main.py 的 OPEN_POSE
  写 gesture_mapping/joint_gain_3d.json  {"joint_gain": [16 浮点]}
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesture_mapping import HandTracker, JointMapper, Calibrator
from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.camera import open_realsense
from gesture_mapping.demo_realtime import _OpenCVCamera, find_best_camera

_MOTOR_LABELS = [
    "Idx abd", "Idx MCP", "Idx PIP", "Idx DIP",
    "Mid abd", "Mid MCP", "Mid PIP", "Mid DIP",
    "Pky abd", "Pky MCP", "Pky PIP", "Pky DIP",
    "Thb MCP", "Thb abd", "Thb PIP", "Thb DIP",
]
_MOTOR_GROUPS = [("Idx", 0), ("Mid", 4), ("Pky", 8), ("Thb", 12)]

# 与 demo_hamer3d 一致: mcp/pip/dip 负 = 向手心弯; 拇指 mcp(ID12) 例外 +1
JOINT_DIR = np.array([-1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1,  1, -1, -1, -1])

_GAIN_MIN, _GAIN_MAX = 0.1, 3.0
_MIN_MOTION = 0.15     # 拟合只用在 |a| 超过此值的姿势 (关节有实际运动)


def fit_all_gains(collected, min_motion=_MIN_MOTION):
    """每关节过原点最小二乘: b ≈ gain * a.

    collected: list[(pose_name, a: (16,), b: (16,))]
    返回 (gains(16), errs(16)); 关节运动不足时 gain 保持 1.0。
    """
    gains = np.ones(16, dtype=np.float64)
    errs = np.zeros(16, dtype=np.float64)
    for j in range(16):
        a_vec, b_vec = [], []
        for _, a, b in collected:
            if abs(float(a[j])) > min_motion:
                a_vec.append(float(a[j]))
                b_vec.append(float(b[j]))
        if len(a_vec) >= 2:
            A = np.array(a_vec)
            B = np.array(b_vec)
            g = float(np.dot(A, B) / max(np.dot(A, A), 1e-9))
            gains[j] = float(np.clip(g, _GAIN_MIN, _GAIN_MAX))
            errs[j] = float(np.mean(np.abs(B - gains[j] * A)))
    return gains, errs


def _draw_hud(frame, angles, calibrator):
    """简单 HUD: 16 关节当前角度 + 校准状态."""
    panel = frame.copy()
    cv2.rectangle(panel, (5, 5), (300, 210), (30, 30, 30), -1)
    frame[:] = cv2.addWeighted(frame, 0.6, panel, 0.4, 0)
    cv2.putText(frame, "CALIBRATED" if calibrator.is_calibrated else "NOT CALIBRATED (SPACE)",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if calibrator.is_calibrated else (0, 140, 255), 1)
    for gi, (fname, s) in enumerate(_MOTOR_GROUPS):
        y = 48 + gi * 24
        if fname == "Thb":
            row = (f"Thb mcp={angles[s+0]:+.2f} abd={angles[s+1]:+.2f} "
                   f"pip={angles[s+2]:+.2f} dip={angles[s+3]:+.2f}")
        else:
            row = (f"{fname:>3s} abd={angles[s+0]:+.2f} mcp={angles[s+1]:+.2f} "
                   f"pip={angles[s+2]:+.2f} dip={angles[s+3]:+.2f}")
        cv2.putText(frame, row, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (0, 220, 0) if angles[s+1] < 1.0 else (0, 200, 255), 1)
    return frame


def _print_gain_table(gains, errs, old_gains):
    print("\n" + "=" * 66)
    print("  拟合结果 (b ≈ gain·a)")
    print(f"  {'ID':>3s} {'label':>9s} {'old':>6s} {'new':>6s} {'err':>6s}")
    print("  " + "-" * 60)
    for j in range(16):
        flag = "" if errs[j] < 0.3 else "  ⚠️ 误差大"
        print(f"  {j:>3d} {_MOTOR_LABELS[j]:>9s} {old_gains[j]:>6.2f} "
              f"{gains[j]:>6.2f} {errs[j]:>6.3f}{flag}")
    print("=" * 66)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1)
    parser.add_argument("--drive", action="store_true",
                        help="连接灵巧手: N/G 驱动到标准姿势对照")
    args = parser.parse_args()

    tracker = HandTracker(max_num_hands=1, min_detection_confidence=0.5)
    mapper = JointMapper()
    mapper.joint_gain = np.ones(16)      # 关键: gain=1.0 采集原始几何角度
    calibrator = Calibrator(mapper)
    angle_filter = OneEuroFilter(n_joints=16, min_cutoff=0.5, beta=0.005)

    # 姿势数据 (须先 calibrate.py 重新录制)
    from main import OPEN_POSE, POSES
    pose_names = [n for n in POSES.keys() if n != "全开/平伸"]
    if not pose_names:
        print("[ERROR] poses.json 无标定姿势 (除全开外), 请先 calibrate.py 录制")
        tracker.close()
        return
    print(f"[INFO] 标定姿势集: {pose_names}")

    # 相机
    cam = open_realsense()
    if cam is not None:
        print("[INFO] Using RealSense SDK color stream")
    else:
        cam_idx = args.camera if args.camera >= 0 else find_best_camera()
        if cam_idx is None:
            print("[ERROR] No camera")
            tracker.close()
            return
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {cam_idx}")
            tracker.close()
            return
        cam = _OpenCVCamera(cap)

    # 灵巧手 (延迟上电: 按 SPACE 校准后)
    leap = None
    if args.drive:
        print("[INFO] 真机模式: 电机未上电。张开手按 SPACE 校准后连接上电。")

    collected = []          # list[(pose_name, a, b)]
    cur_idx = -1
    cur_pose_name = None
    frame_count = 0

    print("\n" + "=" * 50)
    print("  LEAP Hand — Gain Calibration (伪3D, gain=1.0)")
    print("  SPACE=校准 | N=姿势 | G=驱动灵巧手 | S=采集 | D=查看 | W=拟合+写 | Q=退出")
    print("=" * 50)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            results = tracker.detect(frame)
            out = None
            if results:
                hand = results[0]
                frame = tracker.draw_landmarks(frame, [hand])
                out = angle_filter(calibrator.map(hand, (h, w)))
                frame = _draw_hud(frame, out, calibrator)

            if cur_pose_name is not None:
                cv2.putText(frame, f"POSE: {cur_pose_name}  (S=采集)",
                            (12, h - 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (0, 255, 255), 2)

            cv2.imshow("LEAP Gain Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord(" ") and results:
                if args.drive and leap is None:
                    from main import LeapNode
                    try:
                        leap = LeapNode()
                        print("[INFO] LEAP Hand 已上电 (全开位)。")
                    except OSError as e:
                        print(f"[WARN] 上电失败: {e}")
                calibrator.calibrate(results[0], (h, w))
                angle_filter.reset()
                print(f"\n  *** CALIBRATED! baseline max: {calibrator.baseline.max():.3f} rad ***")
            elif key == ord("n"):
                cur_idx = (cur_idx + 1) % len(pose_names)
                cur_pose_name = pose_names[cur_idx]
                print(f"\n  ▶ 姿势 {cur_idx+1}/{len(pose_names)}: 「{cur_pose_name}」"
                      f"  请摆出人手对应姿势 → 按 S 采集")
                if leap is not None:
                    try:
                        leap.set_pose(cur_pose_name)
                        print(f"     (灵巧手已驱动到该姿势, 供对照)")
                    except Exception as e:
                        print(f"  [WARN] 驱动灵巧手失败: {e}")
            elif key == ord("g"):
                if leap is not None and cur_pose_name is not None:
                    try:
                        leap.set_pose(cur_pose_name)
                        print(f"  (重新驱动灵巧手到「{cur_pose_name}」)")
                    except Exception as e:
                        print(f"  [WARN] 驱动失败: {e}")
                elif leap is None:
                    print("  [WARN] 未连接灵巧手 (加 --drive)")
            elif key == ord("s"):
                if results is None or out is None:
                    print("  [WARN] 未检测到手")
                elif cur_pose_name is None:
                    print("  [WARN] 请先按 N 选姿势")
                else:
                    b = JOINT_DIR * (np.array(POSES[cur_pose_name], dtype=np.float64) - OPEN_POSE)
                    collected.append((cur_pose_name, out.copy(), b))
                    print(f"\n  ✅ 采集 [{cur_pose_name}]  a max={np.abs(out).max():.2f}  b max={np.abs(b).max():.2f}")
                    print(f"     累计 {len(collected)} 组:")
                    for name, a, _ in collected:
                        print(f"       {name:8s} a max={np.abs(a).max():.2f}")
            elif key == ord("d"):
                print(f"\n  已采集 {len(collected)} 组:")
                for name, a, b in collected:
                    print(f"    {name:10s} a={np.round(a,2).tolist()}")
                    print(f"             b={np.round(b,2).tolist()}")
            elif key == ord("w"):
                if len(collected) < 3:
                    print(f"  [WARN] 至少需 3 组 (当前 {len(collected)}), 请先多采几组")
                else:
                    gains, errs = fit_all_gains(collected)
                    _print_gain_table(gains, errs, mapper.joint_gain)
                    path = Path(__file__).resolve().parent / "joint_gain_3d.json"
                    mapper.joint_gain = gains
                    mapper.save_gain(str(path))
                    print(f"\n  ✅ 增益已写入 {path}\n  重跑 demo_hamer3d.py 即生效。")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        if leap is not None:
            try:
                leap.set_open()
                leap.disconnect()
            except Exception:
                pass
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("\n[INFO] Calibration stopped.")


if __name__ == "__main__":
    main()
