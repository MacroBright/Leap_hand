"""机械臂视觉遥操 demo: 人手位置/姿态 → remote_event 差分速度。

用法:
  # 仿真臂 (先另开终端: conda activate smolvla && python scripts/mujoco_sim.py --ik --no-camera)
  conda activate leap_hand && cd python
  python gesture_mapping/demo_arm_teleop.py --port socket://localhost:5555

  # 真机臂
  python gesture_mapping/demo_arm_teleop.py --port /dev/ttyUSB0

按键:
  H (按住)   离合器: 按住跟随, 松开=锚定新参考 (走哪停哪)
  C          重载 handeye_calib.json
  Y          e_stop
  Q/ESC      退出
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # python/

from gesture_mapping.arm_client import ArmClient
from gesture_mapping.camera import open_realsense
from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.handeye_calib import load_calib
from gesture_mapping.hand_tracker import HandTracker
from gesture_mapping.wrist_tracker import WristTracker, build_palm_pts

_KEYS = {
    ord("h"): "clutch", ord("H"): "clutch",
    ord("c"): "calib", ord("C"): "calib",
    ord("y"): "estop", ord("Y"): "estop",
    ord("q"): "quit", 27: "quit",
}


def main():
    ap = argparse.ArgumentParser(description="机械臂视觉遥操 (差分速度)")
    ap.add_argument("--port", default="socket://localhost:5555",
                    help="串口或 socket:// (默认仿真 5555)")
    ap.add_argument("--calib", default=str(
        Path(__file__).resolve().parent / "handeye_calib.json"))
    ap.add_argument("--no-drive", action="store_true",
                    help="只显示速度, 不发送命令")
    args = ap.parse_args()

    cam = open_realsense()
    if cam is None:
        sys.exit("未检测到 RealSense (D455) 相机")
    tracker = HandTracker(max_num_hands=1)

    R = None
    if Path(args.calib).exists():
        R = load_calib(args.calib)
        print(f"[标定] 已加载 handeye: {args.calib}")
    else:
        R = np.eye(3)
        print("[标定] 未找到 handeye_calib.json, 使用单位旋转 (仅测试)")

    wt = WristTracker(R=R)
    cmd_smoother = OneEuroFilter(5, min_cutoff=1.5, beta=0.02)
    arm = None if args.no_drive else ArmClient(args.port)
    if arm is not None:
        arm.remote_enable()
        try:
            angles, _, _ = arm.get_state()
            if len(angles) >= 6:
                wt.sync_j5j6(angles[4], angles[5])
        except Exception:
            pass
        print(f"[臂] 已连接 {args.port}, J5={wt.j5_pos_deg:.0f}° J6={wt.j6_pos_deg:.0f}°")

    clutch = False
    print("\n按键: H=离合器(按住跟随,松开锚定)  C=重载标定  Y=急停  Q=退出\n")

    try:
        while True:
            ok, bgr, depth, K = cam.read_with_depth()
            if not ok or bgr is None:
                continue
            hands = tracker.detect(bgr)
            hand = hands[0] if hands else None
            pts = build_palm_pts(hand, depth, K) if hand is not None else None

            key = cv2.waitKey(1) & 0xFF
            if key in _KEYS:
                action = _KEYS[key]
                if action == "clutch":
                    clutch = not clutch
                    wt.capture_reference(pts)
                elif action == "calib" and Path(args.calib).exists():
                    R = load_calib(args.calib)
                    wt.R = R
                    wt.capture_reference(None)   # 清旧参考, 避免 R 系混用
                    print("[标定] 已重载 handeye")
                elif action == "estop" and arm is not None:
                    arm.e_stop()
                    print("[急停] e_stop")
                elif action == "quit":
                    break

            if pts is None:
                cmd = wt.no_hand()
            elif clutch:
                cmd = wt.update(pts)
            else:
                cmd = wt.no_hand()

            cmd = cmd_smoother(np.array(cmd))
            if arm is not None:
                arm.remote_event(*cmd)

            # HUD
            h, w = bgr.shape[:2]
            cv2.putText(bgr, f"CLUTCH:{'ON' if clutch else 'OFF'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if clutch else (0, 0, 255), 2)
            cv2.putText(bgr, f"v=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                             f"J5={cmd[3]:+.2f} J6={cmd[4]:+.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(bgr, f"J5pos={wt.j5_pos_deg:5.1f}° J6pos={wt.j6_pos_deg:5.1f}°",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Arm Teleop", bgr)
    finally:
        if arm is not None:
            arm.remote_disable()
            arm.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[退出] 已安全断开")


if __name__ == "__main__":
    main()
