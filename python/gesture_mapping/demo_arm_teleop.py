"""机械臂视觉遥操 demo: 人手位置/姿态 → 位置跟随 → remote_event 速度.

用法:
  # 仿真臂 (先另开终端: conda activate smolvla && python scripts/mujoco_sim.py --ik --no-camera)
  conda activate leap_hand && cd python
  python gesture_mapping/demo_arm_teleop.py --port socket://localhost:5555

  # 真机臂
  python gesture_mapping/demo_arm_teleop.py --port /dev/ttyUSB0

控制范式: 位置跟随 (手位移 → 目标末端位置 → P 位置环 → 速度命令).
按住 H 时手位移决定目标; 松开 H 重新锚定 (走哪停哪).
仿真臂经 get_ee 读末端反馈 (米→mm); 真机固件无 get_ee 时位置环退回差分模式.

按键:
  H (按住)   离合器: 按住跟随, 松开=重锚定 (走哪停哪)
  C          重载 handeye_calib.json
  K          轴对齐校准向导 (手沿3方向挥动+选1-6方向码, 自动求解手眼R并保存)
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
from gesture_mapping.handeye_calib import load_calib, save_calib, solve_handeye
from gesture_mapping.hand_tracker import HandTracker
from gesture_mapping.wrist_tracker import WristTracker, build_palm_pts

_KEYS = {
    ord("h"): "clutch", ord("H"): "clutch",
    ord("c"): "calib", ord("C"): "calib",
    ord("k"): "calib", ord("K"): "calib",
    ord("y"): "estop", ord("Y"): "estop",
    ord("q"): "quit", 27: "quit",
}

_CALIB_STEP_HINTS = [
    "步骤1: 手沿画面\"向右\"移动一段距离(按住H), 完成后按 SPACE, 再按 1-6 选臂应去方向",
    "步骤2: 手沿画面\"向上\"移动一段距离(按住H), 完成后按 SPACE, 再按 1-6",
    "步骤3: 手\"向前(靠近相机)\"移动一段距离(按住H), 完成后按 SPACE, 再按 1-6",
]
_DIR_CODE_HINT = "方向码: 1=+X 2=-X 3=+Y 4=-Y 5=+Z(上) 6=-Z(下)"


def main():
    ap = argparse.ArgumentParser(description="机械臂视觉遥操 (位置跟随)")
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
    cmd_smoother = OneEuroFilter(5, min_cutoff=5.0, beta=0.05)
    arm = None if args.no_drive else ArmClient(args.port)
    if arm is not None:
        arm.remote_enable()
        try:
            angles, _, _ = arm.get_state()
            if len(angles) >= 6:
                wt.capture(None, None, angles[4], angles[5])   # 锚定 J5/J6
        except Exception:
            pass
        print(f"[臂] 已连接 {args.port}, J5anchor={wt.last_target_j5:.0f}° "
              f"J6anchor={wt.last_target_j6:.0f}°")

    clutch = False

    # 校准状态机 (K 进入轴对齐向导)
    calib_step = 0            # 0=off, 1/2/3=收集第几步
    calib_buf = []            # 滚动 wrist_cam 缓冲
    calib_cam = []            # 已确认的相机系单位方向
    calib_codes = []          # 对应基座方向码
    calib_pending = None      # 待选方向的 cam 单位向量
    CALIB_BUF_MAX = 20

    print("\n按键: H=离合器(按住跟随,松开重锚定)  C=重载标定  K=轴对齐校准向导  Y=急停  Q=退出\n")
    print("[控制范式] 位置跟随遥操: 按住H后, 手相对锚点的位移 → 末端目标位置, "
          "位置环将臂驱动到目标 (误差随臂接近而趋零, 回手锚点即停); "
          "松开H重锚定当前手位+臂位, 走哪停哪. "
          "手在画面中的运动方向经手眼标定R映射到机械臂基座系.\n")

    try:
        while True:
            ok, bgr, depth, K = cam.read_with_depth()
            if not ok or bgr is None:
                continue
            hands = tracker.detect(bgr)
            hand = hands[0] if hands else None
            pts = build_palm_pts(hand, depth, K) if hand is not None else None

            # 读末端+关节反馈 (仿真 get_ee 米→mm; 无 arm 时用零)
            ee_mm = None
            j5c = j6c = 0.0
            if arm is not None:
                ee = arm.get_ee()
                if ee is not None:
                    ee_mm = np.array(ee) * 1000.0
                angles, _, _ = arm.get_state()
                if len(angles) >= 6:
                    j5c, j6c = angles[4], angles[5]

            key = cv2.waitKey(1) & 0xFF

            # 校准帧采集: 记录相机系 wrist 位移缓冲 (未过 R)
            if 1 <= calib_step <= 3 and pts is not None:
                calib_buf.append(pts[0])
                if len(calib_buf) > CALIB_BUF_MAX:
                    calib_buf.pop(0)

            # 校准向导按键 (独立于 _KEYS)
            if 1 <= calib_step <= 3 and key == ord(" ") and pts is not None and len(calib_buf) >= 5:
                d = calib_buf[-1] - calib_buf[0]
                if np.linalg.norm(d) < 30.0:
                    print("位移太小, 重试 (沿该方向移动更远距离)")
                else:
                    calib_pending = d / np.linalg.norm(d)
                    print(f"采集到相机系方向 {np.round(calib_pending, 3)}. " + _DIR_CODE_HINT)
            elif calib_pending is not None and ord("1") <= key <= ord("6"):
                code = key - ord("1") + 1
                calib_cam.append(calib_pending)
                calib_codes.append(code)
                calib_pending = None
                calib_buf = []      # 步骤间清空, 下步 SPACE 只采当前方向位移
                if len(calib_codes) == 3:
                    R = solve_handeye(calib_cam, calib_codes)
                    save_calib(args.calib, R)
                    wt.R = R
                    wt.capture(None, None, 0.0, 0.0)   # 清旧参考, 避免 R 系混用
                    print(f"[校准] R 已保存到 {args.calib}:\n{R}")
                    print("CALIB: 验证 Z 方向 - 手向相机移动, 确认臂朝期望方向; 反了按 Z 翻转, 正常按 SPACE 完成")
                    calib_step = 4
                    calib_buf = []
                else:
                    calib_step += 1
                    print(_CALIB_STEP_HINTS[calib_step - 1])
            elif calib_step == 4 and key == ord(" "):
                calib_step = 0
                calib_buf = []
                print("校准完成")
            elif calib_step == 4 and key in (ord("z"), ord("Z")):
                R = np.asarray(R, float) @ np.diag([1.0, 1.0, -1.0])
                save_calib(args.calib, R)
                wt.R = R
                wt.capture(None, None, 0.0, 0.0)
                print("已翻转 Z 方向并保存")
            elif key in _KEYS:
                action = _KEYS[key]
                if action == "clutch":
                    clutch = not clutch
                    wt.capture(pts, ee_mm, j5c, j6c)   # 按下/松开都重锚定(手参考+臂锚点)
                elif action == "calib" and key in (ord("k"), ord("K")):
                    if calib_step == 0:
                        calib_step = 1
                        calib_buf = []
                        calib_cam = []
                        calib_codes = []
                        calib_pending = None
                        print(_DIR_CODE_HINT)
                        print(_CALIB_STEP_HINTS[0])
                    else:
                        calib_step = 0
                        calib_pending = None
                        calib_buf = []
                        print("[校准] 已退出")
                elif action == "calib" and Path(args.calib).exists():
                    R = load_calib(args.calib)
                    wt.R = R
                    wt.capture(None, None, 0.0, 0.0)   # 清旧参考, 避免 R 系混用
                    print("[标定] 已重载 handeye")
                elif action == "estop" and arm is not None:
                    arm.e_stop()
                    print("[急停] e_stop")
                elif action == "quit":
                    break

            if pts is None:
                cmd = wt.update(None, ee_mm, j5c, j6c)
            elif clutch:
                cmd = wt.update(pts, ee_mm, j5c, j6c)
            else:
                cmd = wt.no_hand()
                wt.capture(pts, ee_mm, j5c, j6c)   # 未按住时也持续重锚定(手参考+臂锚点)

            cmd = cmd_smoother(np.array(cmd))
            if arm is not None:
                arm.remote_event(*cmd)

            # HUD
            h, w = bgr.shape[:2]
            if calib_step > 0:
                if calib_step == 4:
                    cv2.putText(bgr, "CALIB: 验证 Z 方向 - 手向相机移动, 反了按 Z 翻转, 正常按 SPACE 完成",
                                (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 200, 255), 2)
                else:
                    cv2.putText(bgr,
                                f"CALIB: step {calib_step} | pending: "
                                f"{'Y' if calib_pending is not None else 'N'} "
                                f"| pairs: {len(calib_codes)}/3",
                                (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 200, 255), 2)
            cv2.putText(bgr, f"CLUTCH:{'ON' if clutch else 'OFF'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if clutch else (0, 0, 255), 2)
            cv2.putText(bgr, f"v=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}) "
                             f"J5={cmd[3]:+.2f} J6={cmd[4]:+.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(bgr, f"J5tgt={wt.last_target_j5:5.1f}° J6tgt={wt.last_target_j6:5.1f}°",
                        (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            d = wt.last_delta_base
            cv2.putText(bgr, f"d=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f})mm ANCHOR:set",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            if ee_mm is not None:
                t = wt.last_target_ee
                e = t - ee_mm
                cv2.putText(bgr, f"tgt=({t[0]:+.0f},{t[1]:+.0f},{t[2]:+.0f})mm "
                                 f"err=({e[0]:+.0f},{e[1]:+.0f},{e[2]:+.0f})mm",
                            (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 255), 1)
            if pts is not None:
                cv2.putText(bgr,
                            f"depth={pts[0][2]:.0f}mm roll={wt.last_roll_deg:+.1f}deg "
                            f"pitch={wt.last_pitch_deg:+.1f}deg",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)
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
