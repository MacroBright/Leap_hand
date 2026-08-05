#!/usr/bin/env python3
"""
LEAP Hand 右手校准 & 录制工具
==============================
1. 校准 — 重录全开位 (OPEN_POSE)
2. 录制 — 重录已有姿势或新增自定义姿势

所有数据保存到 poses.json, main.py 启动时自动加载。
"""

import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from main import LeapNode, POSES, _is_valid_pose

FINGERS = ["食指", "中指", "无名指", "拇指"]
JOINTS = ["侧摆", "前后", "PIP", "DIP"]
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "poses.json")


def load_poses():
    """加载现有姿势 (优先文件, 回退到 main.py 硬编码)"""
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {k: v.tolist() for k, v in POSES.items()}


def save_poses(poses):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(poses, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 已保存到: {OUTPUT_FILE}")


def read_pos_validated(leap, retries=6):
    """安全读取位置: 串口超时会导致 DynamixelClient 回退上一帧/全零数据,
    必须校验读数是否为真实值, 否则坏数据会被当成姿势入库."""
    for attempt in range(1, retries + 1):
        pos = np.array(leap.read_pos(), dtype=float)
        if _is_valid_pose(pos):
            return pos
        print(f"[重试 {attempt}/{retries}] 位置读数异常 (可能串口超时), 重新读取...")
        time.sleep(0.3)
    raise RuntimeError(
        "多次读取位置失败, 操作中止。请检查 USB 串口连接后重试。")


def print_table(positions, title=""):
    print(f"\n  {title}")
    print(f"  {'手指':<6} {'ID':>3}  {'关节':<6} {'LEAP角度(rad)':>14}")
    print(f"  {'─'*42}")
    for fi, fname in enumerate(FINGERS):
        for ji, jname in enumerate(JOINTS):
            mid = fi * 4 + ji
            print(f"  {fname:<6} {mid:>3}  {jname:<6} {positions[mid]:>14.4f}")
    print()


def record_single(leap, pose_name):
    """录制单个姿势"""
    print(f"\n  📐 录制: {pose_name}")
    print(f"  请手动摆好「{pose_name}」的姿势，然后按 Enter...")
    input()
    pos = read_pos_validated(leap)
    print_table(pos, f"已录制: {pose_name}")
    return [round(float(p), 4) for p in pos]


def do_calibrate(leap):
    """校准模式: 重录全开位"""
    print("\n" + "=" * 55)
    print("  📐 校准 — 重录全开位")
    print("=" * 55)
    print("""
  请把手掌完全张开 (五指伸直、并拢、摊平)，
  保持住这个姿势，然后按 Enter。
""")
    input("按 Enter 记录全开位...")

    pos = read_pos_validated(leap)
    print_table(pos, "新全开位 (OPEN_POSE)")

    poses = load_poses()
    poses["全开/平伸"] = [round(float(p), 4) for p in pos]
    save_poses(poses)

    print("\n✅ 校准完成！重启 Python 脚本即可生效。")


def do_record(leap):
    """录制模式: 循环选择姿势重录或自定义, 选 0 返回"""
    while True:
        poses = load_poses()
        names = list(poses.keys())

        print("\n" + "=" * 55)
        print("  🎬 录制姿势")
        print("=" * 55)
        print("\n  已有姿势:")
        for i, name in enumerate(names):
            print(f"    {i+1}. {name}")
        print(f"    {len(names)+1}. ⭐ 自定义...")
        print(f"    0. 返回主菜单")
        print()

        choice = input(f"请选择 (0-{len(names)+1}): ").strip()

        if choice == "0":
            break

        try:
            idx = int(choice) - 1
        except ValueError:
            print("[错误] 请输入数字")
            continue

        if 0 <= idx < len(names):
            pose_name = names[idx]
            angles = record_single(leap, pose_name)
            poses[pose_name] = angles
            save_poses(poses)

        elif idx == len(names):
            pose_name = input("  请输入新姿势名称: ").strip()
            if not pose_name:
                print("[错误] 名称不能为空")
                continue
            if pose_name in poses:
                confirm = input(f"  「{pose_name}」已存在，覆盖? (y/n): ").strip().lower()
                if confirm != "y":
                    continue

            angles = record_single(leap, pose_name)
            poses[pose_name] = angles
            save_poses(poses)
            print(f"\n✅ 新姿势「{pose_name}」已保存！")

        else:
            print("[错误] 无效选择")


def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   LEAP Hand 校准 & 录制            ║")
    print("╠══════════════════════════════════════╣")
    print("║  1. 校准 (重录全开位)               ║")
    print("║  2. 录制 (录/重录姿势)              ║")
    print("╚══════════════════════════════════════╝")

    choice = input("\n请选择 (1/2): ").strip()

    print("\n连接 LEAP Hand...")
    # calib_mode=True: 即使 poses.json 全开位数据无效也允许连接, 以便重新校准
    leap = LeapNode(calib_mode=True)
    print("已连接。")

    print("关闭扭矩...")
    leap.dxl_client.set_torque_enabled(leap.motors, False)
    print("扭矩已关，手指可自由活动。\n")

    try:
        if choice == "1":
            do_calibrate(leap)
        elif choice == "2":
            do_record(leap)
        else:
            print("[错误] 无效选择，请选 1 或 2")
    except RuntimeError as e:
        print(f"\n[⚠️ 告警] {e}")
        print("          校准/录制中止, 未保存任何数据。")
    finally:
        leap.disconnect()
        print("\n已断开。")


if __name__ == "__main__":
    main()
