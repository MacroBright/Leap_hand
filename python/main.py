import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time
#######################################################
"""LEAP Hand 右手控制 (基于实测姿势)

关节编号:
  食指: ID 0(MCP侧摆) 1(MCP前后) 2(PIP) 3(DIP)
  中指: ID 4(MCP侧摆) 5(MCP前后) 6(PIP) 7(DIP)
  无名指: ID 8(MCP侧摆) 9(MCP前后) 10(PIP) 11(DIP)
  拇指: ID 12 13 14 15
"""
########################################################

# ─── 实测姿势 (LEAP 角度, 单位 rad) ──────────────────────────
# 由 record_poses.py 录制, 直接写入源码, 不加任何补偿层

POSES = {
    "全开/平伸": np.array([3.1155, 4.6204, 3.2076, 1.5785,
                          3.2413, 3.0618, 3.117,  4.5927,
                          3.1186, 3.0756, 3.1799, 4.6004,
                          3.1186, 1.5555, 4.6234, 4.7247]),

    "半握": np.array([3.1094, 4.559,  1.8132, -0.0614,
                     1.7579, 3.0434, 3.0588,  3.114,
                     3.1155, 3.0726, 1.8362,  2.8685,
                     3.1462, 1.511,  4.5421,  2.7811]),

    "全握拳": np.array([3.091,  4.0574, 1.3775, -0.1319,
                       1.4711, 2.4636, 3.094,   2.8624,
                       3.1155, 2.3869, 1.4603,  2.8808,
                       3.5343, 1.4972, 3.2122,  2.7826]),

    "食指指": np.array([3.1201, 4.7968, 3.1845, 1.5463,
                       1.4711, 2.3761, 3.0756, 2.8655,
                       3.1462, 2.3853, 1.4619, 2.8808,
                       3.5159, 1.4174, 3.1983, 2.7826]),

    "比耶": np.array([2.8624, 4.8198, 3.1845, 1.5463,
                     2.9974, 3.2628, 3.3441, 4.8167,
                     3.1447, 2.008,  1.5125, 2.8532,
                     3.5205, 1.4281, 3.2137, 2.7826]),

    "OK手势": np.array([3.1416, 3.4975, 1.9282, -0.0383,
                       2.9836, 3.229,  3.0634,  4.964,
                       3.0971, 2.9943, 3.4284,  4.6111,
                       2.8103, 1.1704, 3.3119,  2.7811]),

    "竖拇指": np.array([3.1401, 3.5113, 1.7135, -0.112,
                       1.399,  2.0862, 3.025,   3.4775,
                       3.1431, 1.9957, 1.6674,  3.2306,
                       3.0542, -0.0138, 4.7277, 5.4487]),
}

OPEN_POSE = POSES["全开/平伸"]  # 基准: 手全开
_OPEN_POSE_HARDCODED = OPEN_POSE.copy()  # 硬编码备份, 校准数据无效时回退

# ─── 从 poses.json 动态加载 (calibrate.py 写入, 优先级高于硬编码) ──
import json as _json
import os as _os
_POSES_FILE = _os.path.join(_os.path.dirname(__file__), "poses.json")


def _is_valid_pose(pose):
    """判断一组 LEAP 角度是否为真实电机读数 (排除读取失败回退的全零/常数/越界).

    校准/录制的读数若因串口超时失败, DynamixelClient 会回退到上一帧或
    初始化的全零数据, 必须在此拦截, 否则坏数据会入库并驱动电机乱动.
    """
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (16,):
        return False
    if not np.all(np.isfinite(pose)):
        return False
    # 读取失败回退特征: 全零 或 16 个值完全相同 (真实电机读数必然分散)
    if np.all(np.abs(pose) < 1e-6) or np.ptp(pose) < 1e-6:
        return False
    # 真实手可动作范围: 依据实测 motor_limits.json 全局边界
    #   max 最大值 = ID12(Thb MCP) 8.12, min 最小值 = ID3(Idx DIP) -0.28
    #   (上限必须覆盖 ID12 限位 8.12; 否则比耶等拇指大动作姿势会被误判)
    if pose.min() < -2.5 or pose.max() > 8.5:
        return False
    return True


OPEN_POSE_VALID = True  # 全开位数据是否有效, 无效时拒绝驱动电机
if _os.path.exists(_POSES_FILE):
    with open(_POSES_FILE, "r", encoding="utf-8") as _f:
        _loaded = _json.load(_f)
    POSES.update({k: np.array(v) for k, v in _loaded.items()})
    OPEN_POSE = POSES["全开/平伸"]
    if not _is_valid_pose(OPEN_POSE):
        print("[⚠️ 告警] poses.json 的全开位无效 (可能来自 calibrate.py 读取失败时记录的全零值)")
        print("          已回退到硬编码全开位; 驱动前将拒绝启动, 请重新校准。")
        OPEN_POSE = _OPEN_POSE_HARDCODED
        OPEN_POSE_VALID = False


class LeapNode:
    def __init__(self, port=None, calib_mode=False):
        """初始化并连接 LEAP Hand.

        calib_mode=True 供 calibrate.py 使用: 全开位数据无效时仍允许连接
        (用于重新校准), 且跳过安全门, 不会把无效位置写入电机.
        """
        # 安全门: 校准数据无效时拒绝驱动电机 (校准模式除外)
        if not OPEN_POSE_VALID and not calib_mode:
            print("\n[⚠️ 告警] LEAP Hand 全开位校准数据无效, 拒绝驱动电机。")
            print("          请重新校准: conda activate leap_hand && python calibrate.py (选 1 校准)")
            print("          或删除 python/poses.json 恢复硬编码姿势。\n")
            raise SystemExit("[LEAP] 全开位校准数据无效, 不驱动电机。")

        self.kP = 600
        self.kI = 0
        self.kD = 200
        self.curr_lim = 350

        self.motors = motors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        # 自动搜索串口
        if port is None:
            ports_to_try = ['/dev/ttyUSB0', '/dev/ttyUSB1', 'COM13']
        else:
            ports_to_try = [port]

        connected = False
        for p in ports_to_try:
            try:
                self.dxl_client = DynamixelClient(motors, p, 4000000)
                self.dxl_client.connect()
                print(f"[INFO] Connected on: {p}")
                connected = True
                break
            except Exception as e:
                print(f"[WARN] {p}: {e}")

        if not connected:
            raise OSError("Could not connect. Check power and USB.")

        # 初始化参数
        self.dxl_client.sync_write(motors, np.zeros(len(motors)), 9, 1)    # Return Delay = 0
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1) # 位置-电流模式
        self.dxl_client.set_torque_enabled(motors, True)

        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * 0.75), 84, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * 0.75), 80, 2)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)

        # 写入实测全开位
        self.curr_pos = OPEN_POSE.copy()
        self.prev_pos = OPEN_POSE.copy()
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
        print("[INFO] LEAP Hand initialized!")
        print(f"[INFO] kP={self.kP}, kD={self.kD}, curr_lim={self.curr_lim}mA")

    # ─── 核心控制 ─────────────────────────────────────────────

    def set_leap(self, pose):
        """直接用 LEAP 角度控制 16 个电机"""
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def set_pose(self, name):
        """用录好的姿势名控制"""
        if name not in POSES:
            print(f"[错误] 未知姿势: {name}，可选: {list(POSES.keys())}")
            return
        self.set_leap(POSES[name])

    def set_open(self):
        """全开"""
        self.set_leap(OPEN_POSE)

    # ─── 单关节相对控制 ───────────────────────────────────────
    # 以实测全开位为基准, relative_angle 为相对偏移 (正值 ≈ 弯曲)

    def set_joint(self, motor_id, relative_angle):
        """设置单个关节, relative_angle 以全开位为基准 (rad)"""
        if 0 <= motor_id <= 15:
            self.curr_pos[motor_id] = OPEN_POSE[motor_id] + relative_angle
            self.curr_pos = lhu.angle_safety_clip(self.curr_pos)
            self.set_leap(self.curr_pos)
        else:
            print(f"[错误] 电机ID 必须在 0-15")

    def set_finger(self, finger_start_id, relative_angles):
        """设置一根手指的 4 个关节"""
        for i, angle in enumerate(relative_angles):
            mid = finger_start_id + i
            self.curr_pos[mid] = OPEN_POSE[mid] + angle
        self.curr_pos = lhu.angle_safety_clip(self.curr_pos)
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


# ─── 主函数 ───────────────────────────────────────────────────

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
