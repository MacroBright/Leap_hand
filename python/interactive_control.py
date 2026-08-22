#!/usr/bin/env python3
"""
LEAP Hand 右手交互式控制脚本 (基于实测姿势)
============================================
逐指、逐关节控制，手势预设基于你录制的实测姿势。

关节编号:
  食指: ID 0(MCP侧摆) 1(MCP前后) 2(PIP) 3(DIP)
  中指: ID 4(MCP侧摆) 5(MCP前后) 6(PIP) 7(DIP)
  无名指: ID 8(MCP侧摆) 9(MCP前后) 10(PIP) 11(DIP)
  拇指: ID 12(MCP前后) 13(MCP侧摆) 14(PIP) 15(DIP)   ← 拇指顺序与其他指不同
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from main import LeapNode, POSES, OPEN_POSE
import time

FINGER_NAMES = ["食指(Index)", "中指(Middle)", "无名指(Ring)", "拇指(Thumb)"]
JOINT_NAMES = ["MCP侧摆", "MCP前后", "PIP", "DIP"]        # 食指/中指/无名指
THUMB_JOINT_NAMES = ["MCP前后", "MCP侧摆", "PIP", "DIP"]  # 拇指: ID 12=mcp, 13=side

# 关节方向: +1 = 用户正输入 → 电机加, -1 = 用户正输入 → 电机减
# 实测:mcp/pip/dip 负值弯向手心, 但thumb的mcp正值弯向手心，index、middle、ring的side 负值向拇指方向
# ⚠ 拇指电机顺序与其他指不同: ID 12=mcp, 13=side, 14=pip, 15=dip
JOINT_DIR = np.array([-1, -1, -1, -1,
                      -1, -1, -1, -1,
                      -1, -1, -1, -1,
                      1, -1, -1, -1])  # thumb: [mcp, side, pip, dip]

FINGER_START = {"食指": 0, "index": 0, "中指": 4, "middle": 4,
                "无名指": 8, "ring": 8, "拇指": 12, "thumb": 12}

# 快捷命令 → 实测姿势名
GESTURE_MAP = {
    "open":    "全开/平伸",
    "fist":    "全握拳",
    "half":    "半握",
    "point":   "食指指",
    "ok":      "OK手势",
    "peace":   "比耶",
    "thumbup": "竖拇指",
}


def print_help():
    print("""
╔══════════════════════════════════════════════════════════════╗
║            LEAP Hand 右手交互控制 (实测姿势)                ║
╠══════════════════════════════════════════════════════════════╣
║  命令格式:  <手指> <关节> <角度值>                          ║
║                                                              ║
║  【手指】index / middle / ring / thumb                       ║
║         食指   中指     无名   拇指                          ║
║                                                              ║
║  【关节】side / mcp / pip / dip / all                        ║
║         侧摆  前后  中关节 远关节 整指                        ║
║                                                              ║
║  【角度】相对位移 (正值=向手心弯曲, 负值=向手背伸展, rad)  ║
║         例: 0.5 = 从全开位向手心弯 0.5 rad                    ║
║  ⚠ thumb 关节顺序: 12=mcp, 13=side (与其他指相反)           ║
║                                                              ║
║  ─── 手势 (基于你的实测数据) ─────────────────────────────  ║
║  open       全开/平伸                                        ║
║  fist       全握拳                                           ║
║  half       半握                                             ║
║  point      食指指                                           ║
║  peace      比耶                                             ║
║  ok         OK手势                                           ║
║  thumbup    竖拇指                                           ║
║                                                              ║
║  ─── 其他命令 ────────────────────────────────────────────  ║
║  state      显示当前位置 (LEAP角度 + 相对位移)                ║
║  raw <ID> <LEAP角度>  直接写LEAP角度                         ║
║  poses      列出所有已录姿势                                  ║
║  help / quit                                                 ║
╚══════════════════════════════════════════════════════════════╝
""")


class HandController:
    def __init__(self, safe_mode=False):
        self.leap = LeapNode(safe_mode=safe_mode)
        self.target = OPEN_POSE.copy()

    def set_joint_relative(self, finger: str, joint: str, angle: float):
        """以实测全开位为基准, 设置关节相对角度"""
        start_id = FINGER_START.get(finger.lower())
        if start_id is None:
            print(f"[错误] 未知手指: {finger}")
            return False

        # ⚠ 拇指物理顺序与其他指不同: ID 12=mcp, 13=side, 14=pip, 15=dip
        if finger.lower() in ("thumb", "拇指"):
            joint_map = {"mcp": 0, "side": 1, "pip": 2, "dip": 3}
        else:
            joint_map = {"side": 0, "mcp": 1, "pip": 2, "dip": 3}
        if joint.lower() == "all":
            ids = list(range(start_id, start_id + 4))
        elif joint.lower() in joint_map:
            ids = [start_id + joint_map[joint.lower()]]
        else:
            print(f"[错误] 未知关节: {joint}，请用 side/mcp/pip/dip/all")
            return False

        for mid in ids:
            self.target[mid] = OPEN_POSE[mid] + JOINT_DIR[mid] * angle
        self.leap.set_leap(self.target)
        return True

    def print_state(self):
        pos, vel, cur = self.leap.pos_vel_eff_srv()
        print(f"\n{'─'*72}")
        print(f"  {'手指':<10} {'ID':>3}  {'关节':<8} {'LEAP实际':>10} {'LEAP全开':>10} {'相对位移':>10}")
        print(f"{'─'*72}")
        for fi, fname in enumerate(FINGER_NAMES):
            names = THUMB_JOINT_NAMES if fname.startswith("拇指") else JOINT_NAMES
            for ji, jname in enumerate(names):
                mid = fi * 4 + ji
                actual = pos[mid]
                base = OPEN_POSE[mid]
                offset = actual - base
                print(f"  {fname:<10} {mid:>3}  {jname:<8} {actual:>10.4f} {base:>10.4f} {offset:>+10.4f}")
        print(f"{'─'*72}")
        print(f"  电流: {cur.min():.0f}~{cur.max():.0f} mA")
        print()

    def disconnect(self):
        self.leap.set_open()
        time.sleep(0.2)
        self.leap.disconnect()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--safe-drive",
        action="store_true",
        help="Drive with the centralized low-force safety profile",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    print("\n启动 LEAP Hand 控制器...")
    ctrl = HandController(safe_mode=args.safe_drive)
    print("就绪！输入 'help' 查看命令，'quit' 退出。\n")

    try:
        while True:
            try:
                cmd = input("LEAP> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not cmd:
                continue

            parts = cmd.lower().split()
            cmd_name = parts[0]

            if cmd_name in ("quit", "exit", "q"):
                break

            elif cmd_name == "help":
                print_help()

            elif cmd_name in ("state", "read"):
                ctrl.print_state()

            # ── 手势 (直接使用 main.py 里的实测姿势) ──
            elif cmd_name in GESTURE_MAP:
                pose_name = GESTURE_MAP[cmd_name]
                ctrl.leap.set_pose(pose_name)
                ctrl.target = POSES[pose_name].copy()
                print(f"[OK] 执行: {pose_name}")
                time.sleep(0.05)
                ctrl.print_state()

            # ── 列出所有姿势 ──
            elif cmd_name == "poses":
                for i, name in enumerate(POSES.keys()):
                    print(f"  {i+1}. {name}")
                print()

            # ── 逐指控制 ──
            elif cmd_name in FINGER_START and len(parts) >= 3:
                finger = cmd_name
                joint = parts[1]
                try:
                    angle = float(parts[2])
                except ValueError:
                    print(f"[错误] 角度必须是数字")
                    continue

                if ctrl.set_joint_relative(finger, joint, angle):
                    start = FINGER_START[finger]
                    time.sleep(0.05)
                    pos = ctrl.leap.read_pos()
                    print(f"[OK] {finger} {joint} 偏移 {angle:+.3f} rad")
                    print(f"     目标: {np.array2string(ctrl.target[start:start+4], precision=3)}")
                    print(f"     实际: {np.array2string(pos[start:start+4], precision=3)}")

            # ── 直接设 LEAP 角度 ──
            elif cmd_name == "raw" and len(parts) >= 3:
                try:
                    mid = int(parts[1])
                    lap = float(parts[2])
                except ValueError:
                    print("[错误] raw <电机ID 0-15> <LEAP角度>")
                    continue
                if 0 <= mid <= 15:
                    ctrl.target[mid] = lap
                    ctrl.leap.set_leap(ctrl.target)
                    print(f"[OK] 电机 {mid} → LEAP {lap:.4f} rad")
                else:
                    print("[错误] 电机ID 必须在 0-15")

            else:
                print(f"[错误] 未知命令: '{cmd}' — 输入 'help' 查看帮助")

    finally:
        print("\n正在安全断开...")
        ctrl.disconnect()
        print("已断开。")


if __name__ == "__main__":
    main()
