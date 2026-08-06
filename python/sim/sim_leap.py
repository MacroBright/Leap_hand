"""SimLeap — MuJoCo 仿真灵巧手, 接口对齐硬件 LeapNode.set_leap/set_open.

在 demo_hamer3d.py 用 `--drive --sim` 替代真手: 同一套手势映射管线
(相机→3D→16DOF→[仿真手]), 零硬件风险验证驱动行为 (平滑/限位/回退).

模型: python/sim/leap_hand/robot_mj.xml (官方 LEAP_Hand_Sim URDF 转 MJCF)
控制: 运动学 qpos 直接设关节角 (稳定、无伺服震荡), 指令裁剪到 URDF 关节限位
       → 仿真手精确跟随映射角度, 限位行为可见
"""

from pathlib import Path

import os
os.environ.setdefault("GLFW_PLATFORM", "x11")  # NVIDIA+Wayland 下 glfw viewer 退出易段错误 → 强制 X11

import numpy as np
import mujoco
import mujoco.viewer  # 显式导入: mujoco 3.11 不随 import mujoco 自动加载 viewer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import OPEN_POSE  # 真机全开位 (绝对位姿), 用于换算相对偏差


class SimLeap:
    """MuJoCo 仿真手 (运动学 qpos 控制). 接口: set_leap(pose)/set_open()/disconnect()."""

    def __init__(self, model_path: str = None, show: bool = True):
        model_path = model_path or str(
            Path(__file__).resolve().parent / "leap_hand" / "robot_mj.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.open_pose = np.array(OPEN_POSE, dtype=np.float64)

        # motor_id (关节名 str(i)) → qpos 索引 (URDF 转 MJCF 后 qpos 顺序 ≠ 电机序)
        name2q = {self.model.joint(i).name: i for i in range(self.model.njnt)}
        self.qpos_idx = [name2q[str(i)] for i in range(16)]
        # 每电机关节限位 (rad, 仿真约定)
        self.ranges = np.array(
            [self.model.joint(name2q[str(i)]).range for i in range(16)])

        # 每关节符号: 真机绝对位姿 → 仿真关节角 (先全 +1, 视觉验证后调)
        self.sim_sign = np.ones(16)

        self._handle = None
        if show:
            self._handle = mujoco.viewer.launch_passive(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ─── 电机位姿 → 仿真关节角 (裁剪到限位) ─────────────────────
    def _pose_to_sim(self, pose: np.ndarray) -> np.ndarray:
        """绝对电机位姿 → 仿真关节角 (相对张开偏差*符号, 裁剪到 URDF 限位)."""
        raw = (np.asarray(pose, dtype=np.float64) - self.open_pose) * self.sim_sign
        return np.clip(raw, self.ranges[:, 0], self.ranges[:, 1])

    # ─── 接口 (对齐 LeapNode) ────────────────────────────────────
    def set_leap(self, pose: np.ndarray):
        """用 LEAP 角度控制 16 个电机 (仿真: 运动学设置关节角)."""
        self.data.qpos[self.qpos_idx] = self._pose_to_sim(pose)
        mujoco.mj_forward(self.model, self.data)
        if self._handle is not None:
            self._handle.sync()

    def set_open(self):
        """全开 (仿真: 所有相对偏差 = 0)."""
        self.set_leap(self.open_pose)

    def disconnect(self):
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def close(self):
        self.disconnect()


def smoke_offscreen(model_path: str = None):
    """无窗口自检: 驱动各电机到相对 ±1 rad, 确认 qpos 跟随 + 限位裁剪. 返回 SimLeap."""
    sim = SimLeap(model_path=model_path, show=False)
    m, d = sim.model, sim.data
    print(f"[smoke] model ok: nbody={m.nbody} njoint={m.njnt} qpos_idx={sim.qpos_idx}")
    print(f"  joint ranges:\n  {np.round(sim.ranges, 3)}")
    # 相对偏差 +0.6 rad → 各关节应跟随 (并受限位裁剪)
    sim.set_leap(sim.open_pose + 0.6 * np.ones(16))
    got = np.array([d.qpos[i] for i in sim.qpos_idx])
    print(f"  +0.6 rad → sim qpos: {np.round(got, 3)}")
    ok = np.allclose(got, np.clip(0.6 * sim.sim_sign, sim.ranges[:, 0], sim.ranges[:, 1]), atol=1e-4)
    print(f"  follow+clip OK: {ok}")
    # 相对偏差 -1.5 rad → 低于下界应裁剪到下限
    sim.set_leap(sim.open_pose - 1.5 * np.ones(16))
    got = np.array([d.qpos[i] for i in sim.qpos_idx])
    print(f"  -1.5 rad → sim qpos (裁剪): {np.round(got, 3)}")
    lo_clipped = np.allclose(got, sim.ranges[:, 0], atol=1e-4)
    print(f"  lower-limit clip OK: {lo_clipped}")
    return sim


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--offscreen", action="store_true", help="无窗口自检 (不弹 viewer)")
    args = ap.parse_args()
    if args.offscreen:
        smoke_offscreen()
    else:
        sim = SimLeap()
        print("[SimLeap] 仿真手已启动 (viewer 窗口). 按 Esc 关闭 viewer, Ctrl+C 退出.")
        try:
            import time
            while sim._handle is not None:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        sim.disconnect()
