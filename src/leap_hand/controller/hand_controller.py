#!/usr/bin/env python3
"""LEAP Hand 16-DOF 右手核心控制器 (LeapNode).

关节编号:
  食指: ID 0(MCP侧摆) 1(MCP前后) 2(PIP) 3(DIP)
  中指: ID 4(MCP侧摆) 5(MCP前后) 6(PIP) 7(DIP)
  无名指: ID 8(MCP侧摆) 9(MCP前后) 10(PIP) 11(DIP)
  拇指: ID 12(MCP弯曲) 13(MCP侧摆) 14(PIP) 15(DIP)
"""
import time
from typing import Optional, Sequence, Union

import numpy as np

from ..driver.dynamixel_client import DynamixelClient
from ..kinematics import limits as lhu
from ..kinematics.limits import angle_safety_clip
from .pose_manager import (
    DEFAULT_POSES,
    is_valid_pose as _is_valid_pose,
    load_poses,
    save_poses,
    unwrap_to_limits,
)

# ─── 姿态数据导出 (兼顾老代码直接访问全局变量) ───────────────────
POSES, OPEN_POSE_VALID = load_poses()
OPEN_POSE = POSES["全开/平伸"]
_OPEN_POSE_HARDCODED = DEFAULT_POSES["全开/平伸"].copy()

if not OPEN_POSE_VALID:
    print("[⚠️ 告警] poses.json 的全开位无效 (可能来自读取失败时记录的全零值)")
    print("          已回退到硬编码全开位; 驱动前将拒绝启动, 请重新校准。")
    OPEN_POSE = _OPEN_POSE_HARDCODED


class LeapNode:
    """LEAP Hand 16-DOF 舵机控制节点."""

    def __init__(
        self,
        port: Optional[str] = None,
        calib_mode: bool = False,
        kP: int = 300,
        kI: int = 0,
        kD: int = 100,
        curr_lim: int = 150,
    ):
        """初始化并连接 LEAP Hand.

        calib_mode=True 供 calibrate.py 使用: 全开位数据无效时仍允许连接
        (用于重新校准), 且跳过安全门, 不会把无效位置写入电机.
        """
        # 安全门: 校准数据无效时拒绝驱动电机 (校准模式除外)
        if not OPEN_POSE_VALID and not calib_mode:
            print("\n[⚠️ 告警] LEAP Hand 全开位校准数据无效, 拒绝驱动电机。")
            print("          请重新校准: leap-calibrate 或运行 calibrate.py")
            raise SystemExit("[LEAP] 全开位校准数据无效, 不驱动电机。")

        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.curr_lim = curr_lim

        self.motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # 自动搜索串口
        if port is None:
            ports_to_try = ["/dev/ttyUSB0", "/dev/ttyUSB1", "COM13"]
        else:
            ports_to_try = [port]

        connected = False
        for p in ports_to_try:
            try:
                self.dxl_client = DynamixelClient(self.motors, p, 4000000)
                self.dxl_client.connect()
                print(f"[INFO] Connected on: {p}")
                connected = True
                break
            except Exception as e:
                print(f"[WARN] {p}: {e}")

        if not connected:
            raise OSError("Could not connect. Check power and USB.")

        # 初始化参数
        self.dxl_client.sync_write(self.motors, np.zeros(len(self.motors)), 9, 1)    # Return Delay = 0
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * 5, 11, 1) # 位置-电流模式
        self.dxl_client.set_torque_enabled(self.motors, True)

        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)

        # 写入实测全开位
        self.curr_pos = OPEN_POSE.copy()
        self.prev_pos = OPEN_POSE.copy()
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        print("[INFO] LEAP Hand initialized!")
        print(f"[INFO] kP={self.kP}, kD={self.kD}, curr_lim={self.curr_lim}mA")

    # ─── 核心控制 ─────────────────────────────────────────────

    def set_leap(self, pose: Sequence[float]):
        """直接用 LEAP 角度控制 16 个电机"""
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose, dtype=float)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_pose(self, name: str):
        """用录好的姿势名控制"""
        if name not in POSES:
            print(f"[错误] 未知姿势: {name}，可选: {list(POSES.keys())}")
            return
        self.set_leap(POSES[name])

    def set_open(self):
        """全开"""
        self.set_leap(OPEN_POSE)

    def set_torque(self, enabled: bool):
        """使能或释放全部 16 个舵机扭矩"""
        self.dxl_client.set_torque_enabled(self.motors, enabled)


    # ─── 单关节相对控制 ───────────────────────────────────────
    # 以实测全开位为基准, relative_angle 为相对偏移 (正值 ≈ 弯曲)

    def set_joint(self, motor_id: int, relative_angle: float):
        """设置单个关节, relative_angle 以全开位为基准 (rad)"""
        if 0 <= motor_id <= 15:
            self.curr_pos[motor_id] = OPEN_POSE[motor_id] + relative_angle
            self.curr_pos = angle_safety_clip(self.curr_pos)
            self.set_leap(self.curr_pos)
        else:
            print(f"[错误] 电机ID 必须在 0-15")

    def set_finger(self, finger_start_id: int, relative_angles: Sequence[float]):
        """设置一根手指的 4 个关节"""
        for i, angle in enumerate(relative_angles):
            mid = finger_start_id + i
            self.curr_pos[mid] = OPEN_POSE[mid] + angle
        self.curr_pos = angle_safety_clip(self.curr_pos)
        self.set_leap(self.curr_pos)

    # ─── 读取 ─────────────────────────────────────────────────

    def read_pos(self):
        return self.dxl_client.read_pos()

    def read_vel(self):
        return self.dxl_client.read_vel()

    def read_cur(self):
        return self.dxl_client.read_cur()

    def pos_vel(self):
        return self.dxl_client.read_pos_vel()

    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()

    def disconnect(self):
        self.dxl_client.disconnect()
        print("[INFO] Disconnected.")


def main(**kwargs):
    leap = LeapNode()
    try:
        while True:
            leap.set_open()
            print("Position: " + str(leap.read_pos()))
            time.sleep(0.03)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        leap.disconnect()


if __name__ == "__main__":
    main()
