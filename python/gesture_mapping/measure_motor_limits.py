#!/usr/bin/env python3
"""手动推指记录法 — 实测每个电机的真实机械限位 (min/max rad)。

安全设计:
  - 电机转矩关闭 (set_torque_enabled False) → 手指可自由手动活动, 无强迫过行程
  - 你手动把每个关节轻推到自然挡块 (软停), 脚本读取编码器角度
  - 不伤腱/齿轮 (无强迫驱动到硬挡块)

流程 (逐个电机 0-15):
  1) 手动把该关节推到 MIN 端 → 按 Enter 记录
  2) 手动把该关节推到 MAX 端 → 按 Enter 记录
  3) 确认记录值: Enter=保存继续 / r=重测该电机

输出: python/gesture_mapping/motor_limits.json  {"min": [16], "max": [16]}
  供 demo_hamer3d.py --drive 裁剪 (OPEN_POSE + JOINT_DIR*angles 裁剪到实测范围)

用法 (先接好 LEAP Hand USB + 上电):
  conda activate hamer
  python gesture_mapping/measure_motor_limits.py [--port /dev/ttyUSB0]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from main import LeapNode, OPEN_POSE  # 复用串口搜索 + 初始化

# 电机 → 手指/关节 标签 (对齐 demo_realtime._MOTOR_DIAG)
_MOTOR_LABELS = [
    "Idx MCP-side", "Idx MCP", "Idx PIP", "Idx DIP",
    "Mid MCP-side", "Mid MCP", "Mid PIP", "Mid DIP",
    "Pky MCP-side", "Pky MCP", "Pky PIP", "Pky DIP",
    "Thb MCP", "Thb side", "Thb PIP", "Thb DIP",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=str, default=None,
                        help="串口 (默认自动搜索 ttyUSB0/1, COM13)")
    parser.add_argument("--out", type=str, default=None,
                        help="输出 json 路径 (默认 python/gesture_mapping/motor_limits.json)")
    args = parser.parse_args()

    leap = LeapNode(port=args.port)
    motors = list(leap.motors)
    dxl = leap.dxl_client

    limits_min = np.zeros(16, dtype=np.float64)
    limits_max = np.zeros(16, dtype=np.float64)

    try:
        # 转矩关闭 → 手指可手动活动
        dxl.set_torque_enabled(motors, False)
        print("[INFO] 转矩已关闭, 手指可手动推动。Ctrl+C 随时退出。")
        input("    准备就绪后按 Enter 开始逐个电机测量...")

        for mid in range(16):
            label = _MOTOR_LABELS[mid]
            while True:
                print(f"\n=== Motor {mid:2d} ({label}) ===")
                input(f"  推到 {label} 的 MIN 端 (最开/伸直), 按 Enter 记录...")
                time.sleep(0.3)
                pmin = float(dxl.read_pos([mid])[0])
                input(f"  推到 {label} 的 MAX 端 (最弯/收紧), 按 Enter 记录...")
                time.sleep(0.3)
                pmax = float(dxl.read_pos([mid])[0])
                lo, hi = min(pmin, pmax), max(pmin, pmax)
                print(f"  读得: min={lo:.4f}  max={hi:.4f}  (range {hi - lo:.4f} rad)")
                choice = input("  确认? [Enter=保存, r=重测该电机] ").strip().lower()
                if choice != "r":
                    limits_min[mid] = lo
                    limits_max[mid] = hi
                    break

    finally:
        # 恢复: 转矩开启 + 回到全开位 (不把手留在瘫软状态)
        try:
            dxl.set_torque_enabled(motors, True)
            dxl.write_desired_pos(motors, OPEN_POSE)
            time.sleep(0.5)
            print("[INFO] 转矩已恢复, 手回到全开位。")
        except Exception as e:
            print(f"[WARN] 恢复转矩失败: {e}")

    out = args.out or str(Path(__file__).resolve().parent / "motor_limits.json")
    data = {
        "min": [float(x) for x in limits_min],
        "max": [float(x) for x in limits_max],
    }
    with open(out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n已保存: {out}")
    print(f"  {'ID':>3s}  {'label':>12s}  {'min':>8s}  {'max':>8s}  {'range':>8s}   OPEN在范围?")
    for i in range(16):
        ok = "✓" if limits_min[i] <= OPEN_POSE[i] <= limits_max[i] else "✗ OPEN超范围!"
        print(f"  {i:>3d}  {_MOTOR_LABELS[i]:>12s}  {limits_min[i]:>8.4f}  "
              f"{limits_max[i]:>8.4f}  {limits_max[i] - limits_min[i]:>8.4f}  {ok}")

    bad = [i for i in range(16)
           if not (limits_min[i] <= OPEN_POSE[i] <= limits_max[i])]
    if bad:
        print(f"\n[WARN] 电机 {bad} 的 OPEN_POSE 不在实测范围内 — 需重测这些电机, "
              "否则 --drive 裁剪会把它们顶到限位。")


if __name__ == "__main__":
    main()
