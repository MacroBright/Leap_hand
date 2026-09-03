# 机械臂视觉遥操 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> ⚠ **路径迁移注 (2026-08)**：本 plan 描述的 `R_L/python/gesture_mapping/handeye_calib.py` /
> `arm_client.py` / `demo_arm_teleop.py` 已迁至 `Arm-robot_VLA/scripts/`。`wrist_tracker.py`
> 等共用模块留 Leap_Hand。下方"Task 5: handeye_calib.py"等任务清单保留历史,
> 新代码提交请直接落到 Arm-robot_VLA/scripts/ 下。

**Goal:** 用 D455 深度反投影的人手手腕位置/姿态，通过差分速度（`remote_event`）实时遥操 zero-robotic-arm 末端（位置 + J5 上下 + J6 旋转），仿真先行。

**Architecture:** 视觉侧（Leap_Hand）产出 `(vx,vy,vz,j5,j6)∈[-1,1]` 速度命令 → `remote_event` → STM32 固件（位置 IK + J5/J6 关节速度）或 MuJoCo 仿真。核心新模块 `wrist_tracker.py`（深度反投影 + 掌参考系 + 动态参考 + 速度生成），机械臂侧仅做仿真 `remote_event` 语义对齐（与固件逐字节一致）。

**Tech Stack:** pyrealsense2（D455）、MediaPipe Hands（已有 `hand_tracker.py`）、numpy、pyserial（`socket://` 仿真）、MuJoCo 仿真（Arm-robot_VLA `mujoco_sim.py`）、pytest（Leap_Hand）、独立脚本（Arm-robot_VLA，无 pytest）。

---

## Global Constraints

- **仓库**：`R_A` = `/home/bright/win_office/ubantu_files/project/Arm-robot_VLA`（conda env `smolvla`）；`R_L` = `/home/bright/win_office/ubantu_files/project/Leap_Hand`（conda env `leap_hand`）。每任务标注所属仓库。
- **备份规则**（R_L CLAUDE.md §九）：改动已存在文件前，先复制到备份目录。R_L → `python/backups/2026-08-11_arm-visual-teleop/`；R_A → `backups/2026-08-11_arm-visual-teleop/`。
- **`remote_event` 语义基准**：STM32 固件 `firmware/src/robot_cmd.c` `robot_remote_event_handle()`。解析 `remote_event p0 p1 p2 p3 p4 p5`：`vx=-p0`、`vy=p1`、`vz=(p4-p5)/2`、`rx=-p3`(→J5)、`ry=p2`(→J6)。**客户端生成公式**：`p0=-vx, p1=vy, p2=j6, p3=-j5, p4=vz, p5=-vz`。
- **测试运行**：R_L `cd python && pytest tests/test_xxx.py -v`（conda leap_hand）；R_A 无 pytest，用 `python scripts/test_xxx.py` 直接运行。
- **仿真/真机符号、轴分配以 M2 实测为准**：M2 阶段如发现 J5/J6 方向反或轴对调，只改 R_L 侧 `wrist_tracker.py`/`demo` 的符号常量，不动固件。

---

### Task 0: 改动前备份（R_A + R_L）

**Files:**
- Create: `R_A/backups/2026-08-11_arm-visual-teleop/mujoco_sim.py`（副本）
- Create: `R_L/python/backups/2026-08-11_arm-visual-teleop/camera.py`（副本）

**Interfaces:**
- Consumes: —
- Produces: 备份目录，供后续任务参考原始版。

- [ ] **Step 1: 创建 R_A 备份**

```bash
mkdir -p /home/bright/win_office/ubantu_files/project/Arm-robot_VLA/backups/2026-08-11_arm-visual-teleop
cp /home/bright/win_office/ubantu_files/project/Arm-robot_VLA/scripts/mujoco_sim.py \
   /home/bright/win_office/ubantu_files/project/Arm-robot_VLA/backups/2026-08-11_arm-visual-teleop/mujoco_sim.py
```

- [ ] **Step 2: 创建 R_L 备份**

```bash
mkdir -p /home/bright/win_office/ubantu_files/project/Leap_Hand/python/backups/2026-08-11_arm-visual-teleop
cp /home/bright/win_office/ubantu_files/project/Leap_Hand/python/gesture_mapping/camera.py \
   /home/bright/win_office/ubantu_files/project/Leap_Hand/python/backups/2026-08-11_arm-visual-teleop/camera.py
```

- [ ] **Step 3: 验证备份**

```bash
ls -la /home/bright/win_office/ubantu_files/project/Arm-robot_VLA/backups/2026-08-11_arm-visual-teleop/
ls -la /home/bright/win_office/ubantu_files/project/Leap_Hand/python/backups/2026-08-11_arm-visual-teleop/
```

Expected: 两个备份文件存在。

- [ ] **Step 4: Commit**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand && git add python/backups/ docs/design/2026-08-11-arm-visual-teleop-design.md && git commit -m "docs(design): 机械臂视觉遥操设计 + 改动备份"
```

---

### Task 1: 固件 remote_event 语义纯函数（R_A）

**Files:**
- Create: `R_A/scripts/remote_semantics.py`
- Test: `R_A/scripts/test_remote_semantics.py`

**Interfaces:**
- Consumes: —
- Produces: `parse_remote_event(vals: Sequence[float]) -> (np.ndarray(3,), float, float)` —— 返回 `(v_lin系数, j5系数, j6系数)`，系数与固件映射一致（调用方乘各自增益）。Task 2 的 `mujoco_sim.py` 调用它。

- [ ] **Step 1: 写失败测试**

创建 `R_A/scripts/test_remote_semantics.py`：

```python
"""remote_semantics 纯函数单测（R_A 无 pytest，直接运行）。

用法: conda activate smolvla && python scripts/test_remote_semantics.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_semantics import parse_remote_event


def test_vx_negated():
    v_lin, j5, j6 = parse_remote_event([1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(v_lin, [-1, 0, 0])
    assert j5 == 0.0 and j6 == 0.0


def test_vy_direct():
    v_lin, _, _ = parse_remote_event([0, 0.5, 0, 0, 0, 0])
    np.testing.assert_allclose(v_lin, [0, 0.5, 0])


def test_vz_from_p4p5():
    # vz = (p4 - p5)/2: p4=0.6, p5=0.2 → 0.2
    v_lin, _, _ = parse_remote_event([0, 0, 0, 0, 0.6, 0.2])
    np.testing.assert_allclose(v_lin, [0, 0, 0.2])


def test_j6_from_p2():
    v_lin, j5, j6 = parse_remote_event([0, 0, 0.8, 0, 0, 0])
    assert j6 == 0.8 and j5 == 0.0


def test_j5_negated_from_p3():
    v_lin, j5, j6 = parse_remote_event([0, 0, 0, 0.4, 0, 0])
    assert j5 == -0.4 and j6 == 0.0


def test_client_roundtrip():
    # 客户端公式 p0=-vx p1=vy p2=j6 p3=-j5 p4=vz p5=-vz → 应还原原值
    vx, vy, vz, j5, j6 = 0.7, -0.3, 0.5, 0.4, -0.9
    vals = [-vx, vy, j6, -j5, vz, -vz]
    v_lin, j5_out, j6_out = parse_remote_event(vals)
    np.testing.assert_allclose(v_lin, [vx, vy, vz])
    assert j5_out == j5 and j6_out == j6


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    print("ALL PASS" if failed == 0 else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
```

- [ ] **Step 2: 运行确认失败**

Run: `conda activate smolvla && cd /home/bright/win_office/ubantu_files/project/Arm-robot_VLA && python scripts/test_remote_semantics.py`
Expected: `ModuleNotFoundError: No module named 'remote_semantics'`。

- [ ] **Step 3: 写最小实现**

创建 `R_A/scripts/remote_semantics.py`：

```python
"""固件 remote_event 语义的纯函数封装（供 mujoco_sim 与单测使用）。

与 STM32 固件 firmware/src/robot_cmd.c 的 robot_remote_event_handle()
逐字节一致:

    remote_event p0 p1 p2 p3 p4 p5
      vx = -p0            # 基座系 x 线速度系数
      vy =  p1            # 基座系 y 线速度系数
      vz = (p4 - p5)/2    # 基座系 z 线速度系数
      rx = -p3            # J5 关节速度系数（末端上下）
      ry =  p2            # J6 关节速度系数（末端旋转）

本模块不含增益/单位换算——调用方（mujoco_sim）自行乘线/角速度增益。
"""
from typing import Sequence, Tuple

import numpy as np


def parse_remote_event(vals: Sequence[float]) -> Tuple[np.ndarray, float, float]:
    """把 6 个 remote_event 参数解析为 (v_lin(3,), j5_coef, j6_coef)。

    v_lin 为基座系线速度系数 [-p0, p1, (p4-p5)/2]；j5/j6 为关节速度系数。
    """
    p0, p1, p2, p3, p4, p5 = (float(v) for v in vals[:6])
    v_lin = np.array([-p0, p1, (p4 - p5) / 2.0])
    j5_coef = -p3
    j6_coef = p2
    return v_lin, j5_coef, j6_coef
```

- [ ] **Step 4: 运行确认通过**

Run: `python scripts/test_remote_semantics.py`
Expected: `ALL PASS`，exit 0。

- [ ] **Step 5: Commit（R_A 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Arm-robot_VLA
git add scripts/remote_semantics.py scripts/test_remote_semantics.py
git commit -m "feat(sim): 固件 remote_event 语义纯函数 + 单测"
```

---

### Task 2: mujoco_sim.py 接入固件语义 + 逐轴验证（R_A）

**Files:**
- Modify: `R_A/scripts/mujoco_sim.py`（`step()` 的 remote 转换块约 L205-303；`main()` 的日志 L781-782；顶部 import）
- Create: `R_A/scripts/verify_remote_semantics.py`

**Interfaces:**
- Consumes: `parse_remote_event`（Task 1）
- Produces: 仿真 `remote_event` 行为与固件逐字节一致；`verify_remote_semantics.py` 逐轴验证脚本。

- [ ] **Step 1: 顶部 import**

在 `mujoco_sim.py` 的 import 区（`import numpy as np` 之后）加：

```python
from remote_semantics import parse_remote_event
```

- [ ] **Step 2: 改 `step()` 的 remote 转换块**

将 `step()` 中 `if remote_active:` 块（约 L205-241）整体替换为：

```python
        if remote_active:
            # EMA 滤波
            alpha = 0.4
            if not self._ema_initialized:
                self._ema_vals = list(raw)
                self._ema_initialized = True
            else:
                for i in range(6):
                    self._ema_vals[i] = (alpha * raw[i] +
                                         (1 - alpha) * self._ema_vals[i])
            # 摇杆残余死区 (视觉遥操在 PC 端已做死区)
            for i, val in enumerate(self._ema_vals):
                if abs(val) < 0.03:
                    self._ema_vals[i] = 0.0
            # 固件语义 (robot_cmd.c): vx/vy/vz 基座系线速度, p3→J5, p2→J6
            v_lin, j5_coef, j6_coef = parse_remote_event(self._ema_vals)
            v_lin = v_lin * REMOTE_LIN_GAIN           # 系数 → m/s
            j5_vel = j5_coef * REMOTE_GAIN_RAD        # J5 关节速度 rad/s
            j6_vel = j6_coef * REMOTE_GAIN_RAD        # J6 关节速度 rad/s
            v_ang = np.zeros(3)                       # 固件固定末端方向
        else:
            self._ema_initialized = False
```

- [ ] **Step 3: 改 `step()` 的 use_ik 扭矩块**

将子步进内 `elif remote_active and self.use_ik:` 块（约 L256-296）替换为：

```python
            if freeze:
                self.data.ctrl[:NUM_JOINTS] = 0.0
            elif remote_active and self.use_ik:
                # 位置 IK (与固件 robot_pid_remote 一致, 末端方向固定), J5/J6 直接关节速度
                mujoco.mj_jacSite(self.model, self.data,
                                  jac_pos, jac_rot, self._ee_site_id)
                if np.any(np.abs(v_lin) > 1e-6):
                    Jp = jac_pos[:, :NUM_JOINTS]
                    JJT = Jp @ Jp.T + lam * lam * np.eye(3)
                    dq = Jp.T @ np.linalg.solve(JJT, v_lin)
                else:
                    dq = np.zeros(NUM_JOINTS)
                dq[4] = j5_vel
                dq[5] = j6_vel
                for i in range(NUM_JOINTS):
                    self.data.ctrl[i] = (-PID_KV[i] * (qvel[i] - dq[i])
                                         + gravity[i])
            elif remote_active:
                # 非 IK 路径: 同参数语义 (vx/vy/vz→J1-J3 关节速度近似)
                target_vel = [v_lin[0] * REMOTE_GAIN_RAD,
                              v_lin[1] * REMOTE_GAIN_RAD,
                              v_lin[2] * REMOTE_GAIN_RAD,
                              0.0, j5_vel, j6_vel]
                for i in range(NUM_JOINTS):
                    self.data.ctrl[i] = (-PID_KV[i] * (qvel[i] - target_vel[i])
                                         + gravity[i])
            else:
                for i in range(NUM_JOINTS):
                    pos_err = ctrl_targets[i] - qpos[i]
                    self.data.ctrl[i] = (PID_KP[i] * pos_err
                                         - PID_KV[i] * qvel[i]
                                         + gravity[i])
```

> 说明：删除原 `v_ang`/`v_full`/`has_ang` 分支（固件无末端角速度）。`target_vel` 现在在 `elif remote_active:` 分支内局部定义，不需要上方 `target_vel = [0.0]*NUM_JOINTS` 的旧赋值，可删除。

- [ ] **Step 4: 修正 `main()` 误导日志**

将 `main()` 中 `--ik` 分支的日志（约 L780-782）改为：

```python
    if args.ik:
        log("Jacobian IK 笛卡尔控制已启用")
        log("remote_event 语义: vx/vy/vz 基座系线速度 (与固件 robot_cmd.c 一致)")
```

- [ ] **Step 5: 写逐轴验证脚本**

创建 `R_A/scripts/verify_remote_semantics.py`：

```python
"""端到端验证: 仿真 remote_event 语义与固件一致。

用法: conda activate smolvla && python scripts/verify_remote_semantics.py
启动无头仿真(--ik --no-camera) → remote_enable → 逐轴发命令 → get_state/get_ee 断言方向。
"""
import socket
import subprocess
import sys
import time
from pathlib import Path

PORT = 5588
ROOT = Path(__file__).resolve().parent.parent
SIM_CMD = [sys.executable, str(ROOT / "scripts" / "mujoco_sim.py"),
           "--port", str(PORT), "--ik", "--no-camera"]


def send(sock, cmd):
    sock.sendall((cmd + "\n").encode())


def _recv_line(sock):
    buf = b""
    while not buf.endswith(b"\n"):
        buf += sock.recv(1)
    return buf.decode().strip()


def get_state(sock):
    send(sock, "get_state")
    line = _recv_line(sock)
    vals = [float(x) for x in line.split(":", 1)[1].split(",")]
    return vals[:6], vals[6:12]


def get_ee(sock):
    send(sock, "get_ee")
    line = _recv_line(sock)
    return [float(x) for x in line.split(":", 1)[1].split(",")[:3]]


def drive(sock, vals, seconds=0.25, hz=50):
    """按 vals 连续发 remote_event seconds 秒."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        send(sock, "remote_event " + " ".join(f"{v:.3f}" for v in vals))
        time.sleep(1.0 / hz)


def check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}")
    return ok


def main():
    proc = subprocess.Popen(SIM_CMD, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    time.sleep(6)  # 等待模型加载 + TCP 就绪
    sock = socket.create_connection(("127.0.0.1", PORT), timeout=5)
    sock.settimeout(3.0)
    send(sock, "remote_enable")
    time.sleep(1.0)

    results = []

    # 1) vx: p0=-1 → vx=+1 → EE +x
    ee0 = get_ee(sock)
    drive(sock, [-1, 0, 0, 0, 0, 0])
    ee1 = get_ee(sock)
    results.append(check("vx→+x", ee1[0] - ee0[0] > 0.003,
                         f"Δx={(ee1[0]-ee0[0])*1000:.1f}mm"))

    # 2) vy: p1=+1 → vy=+1 → EE +y
    ee0 = get_ee(sock)
    drive(sock, [0, 1, 0, 0, 0, 0])
    ee1 = get_ee(sock)
    results.append(check("vy→+y", ee1[1] - ee0[1] > 0.003,
                         f"Δy={(ee1[1]-ee0[1])*1000:.1f}mm"))

    # 3) vz: p4=1,p5=0 → vz=+0.5 → EE +z
    ee0 = get_ee(sock)
    drive(sock, [0, 0, 0, 0, 1, 0])
    ee1 = get_ee(sock)
    results.append(check("vz→+z", ee1[2] - ee0[2] > 0.001,
                         f"Δz={(ee1[2]-ee0[2])*1000:.1f}mm"))

    # 4) J5: p3=-1 → rx=+1 → J5 角度增大
    a0, _ = get_state(sock)
    drive(sock, [0, 0, 0, -1, 0, 0])
    a1, _ = get_state(sock)
    results.append(check("J5(rx)正转", a1[4] - a0[4] > 1.0,
                         f"ΔJ5={a1[4]-a0[4]:.1f}°"))

    # 5) J6: p2=+1 → ry=+1 → J6 角度增大
    a0, _ = get_state(sock)
    drive(sock, [0, 0, 1, 0, 0, 0])
    a1, _ = get_state(sock)
    results.append(check("J6(ry)正转", a1[5] - a0[5] > 1.0,
                         f"ΔJ6={a1[5]-a0[5]:.1f}°"))

    sock.close()
    proc.terminate()
    proc.wait(timeout=5)
    print("ALL PASS" if all(results) else f"{sum(results)}/5 PASS")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
```

> 注：若环境无显示/渲染权限导致相机子进程异常，已用 `--no-camera`；若仍失败，改用 `--camera-gl osmesa`。`get_ee` 返回世界系末端位置（Jacobian 即世界系），断言方向即验证线速度方向。

- [ ] **Step 6: 运行验证脚本**

Run: `conda activate smolvla && cd /home/bright/win_office/ubantu_files/project/Arm-robot_VLA && python scripts/verify_remote_semantics.py`
Expected: `5/5 PASS`。若某轴方向与预期相反，说明仿真世界系与基座系约定差异——记录现象，不改断言，交由 M2 人工复核。

- [ ] **Step 7: 回归——手柄仿真仍可动（可选，有手柄时）**

Run: `python scripts/joystick_control.py --port socket://localhost:5555 --camera 0`
Expected: 手柄仍能驱动仿真臂（方向语义变为线性，与真机一致）。

- [ ] **Step 8: Commit（R_A 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Arm-robot_VLA
git add scripts/mujoco_sim.py scripts/verify_remote_semantics.py
git commit -m "feat(sim): remote_event 对齐固件语义 (线性笛卡尔 + J5/J6 直驱)"
```

---

### Task 3: 修正 SERIAL_COMMANDS.md 关节标注（R_A，文档）

**Files:**
- Modify: `R_A/docs/SERIAL_COMMANDS.md`（关节编号表 + 表格下方注释）

**Interfaces:**
- Consumes: —
- Produces: 文档反映真机实测物理行为（用户 2026-08-11 实测）。

- [ ] **Step 1: 更新关节表说明列**

把 [SERIAL_COMMANDS.md](docs/SERIAL_COMMANDS.md) 关节编号表的 J4/J5/J6 行的"名称/说明"改为（名称保留以兼容引用，说明列标注实测行为）：

| **4** | wrist_flex | 末端旋转（真机实测；旧标注"腕部俯仰"） | -90° ~ 90° | **5** |
| **5** | wrist_roll | 末端上下（真机实测；旧标注"腕部旋转"） | 0° ~ 90° | **6** |
| **6** | gripper | 末端旋转（真机实测；旧标注"末端/夹爪"） | 0° ~ 360° | **7** |

- [ ] **Step 2: 表格下方加注释**

在关节编号表下方追加：

```markdown
> **⚠ 物理行为说明（2026-08-11 真机实测）**：关节 **J4 与 J6** 为末端旋转运动，
> **J5** 为末端上下运动。与文档旧标注（wrist_flex/wrist_roll/gripper）不符，
> 名称保留仅为兼容既有引用；具体轴系方向以真机调试为准。
```

- [ ] **Step 3: Commit（R_A 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Arm-robot_VLA
git add docs/SERIAL_COMMANDS.md
git commit -m "docs: 修正 J4/J5/J6 关节物理行为标注 (真机实测)"
```

---

### Task 4: camera.py 增加 depth 流（R_L）

**Files:**
- Modify: `R_L/python/gesture_mapping/camera.py`（`RealSenseSource` + 新增 `read_with_depth`）
- Test: `R_L/python/tests/test_camera.py`

**Interfaces:**
- Consumes: —
- Produces: `RealSenseSource.read()`（兼容，color only）；`read_with_depth() -> (ok, bgr, depth_mm|None, intrinsics|None)`；`intrinsics() -> (fx,fy,cx,cy)|None`。Task 9 demo 调用 `read_with_depth`。

- [ ] **Step 1: 写失败测试（API 契约冒烟）**

创建 `R_L/python/tests/test_camera.py`：

```python
"""camera.py API 契约测试（深度扩展后不破坏 read() 兼容）。"""
import pytest


def test_real_sense_source_has_depth_api():
    from gesture_mapping.camera import RealSenseSource
    assert hasattr(RealSenseSource, "read")
    assert hasattr(RealSenseSource, "read_with_depth")
    assert hasattr(RealSenseSource, "intrinsics")


def test_open_realsense_returns_source_or_none():
    from gesture_mapping.camera import open_realsense
    cam = open_realsense()
    if cam is not None:  # 有 D455 时
        try:
            ok, bgr = cam.read()
            assert ok and bgr is not None
            assert bgr.shape[2] == 3
        finally:
            cam.release()
    else:  # 无硬件时跳过
        pytest.skip("无 RealSense 设备")
```

- [ ] **Step 2: 运行确认失败**

Run: `conda activate leap_hand && cd /home/bright/win_office/ubantu_files/project/Leap_Hand/python && pytest tests/test_camera.py -v`
Expected: `FAILED`（`RealSenseSource` 无 `read_with_depth`）。

- [ ] **Step 3: 实现 depth 扩展**

修改 `R_L/python/gesture_mapping/camera.py` 的 `RealSenseSource`（保留 `read()` 兼容，新增 depth）：

```python
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 enable_depth: bool = True):
        import pyrealsense2 as rs
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if enable_depth:
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self._profile = self._pipeline.start(config)
        self._align = rs.align(rs.stream.color)
        self._intrinsics = None
        try:
            intr = (self._profile.get_stream(rs.stream.color)
                    .as_video_stream_profile().get_intrinsics())
            self._intrinsics = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        except Exception:
            self._intrinsics = None
        self._width, self._height = width, height

    def _next_aligned(self):
        frames = self._pipeline.wait_for_frames()
        return self._align.process(frames)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frames = self._next_aligned()
            color = frames.get_color_frame()
            if color is None:
                return False, None
            return True, np.asanyarray(color.get_data()).copy()
        except Exception:
            return False, None

    def read_with_depth(self) -> Tuple[bool, Optional[np.ndarray],
                                       Optional[np.ndarray], Optional[tuple]]:
        """返回 (ok, bgr, depth_mm, intrinsics)。depth 已与 color 对齐 (u16 mm)。"""
        try:
            frames = self._next_aligned()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if color is None:
                return False, None, None, None
            bgr = np.asanyarray(color.get_data()).copy()
            depth_mm = (np.asanyarray(depth.get_data()).copy()
                        if depth is not None else None)
            return True, bgr, depth_mm, self._intrinsics
        except Exception:
            return False, None, None, None

    def intrinsics(self):
        return self._intrinsics
```

（`read_with_depth` 每次调用独立取帧；demo 应每循环只调用它一次，避免 color/depth 错帧。）

- [ ] **Step 4: 运行确认通过**

Run: `conda activate leap_hand && cd python && pytest tests/test_camera.py -v`
Expected: `2 passed`（无 D455 时第 2 个 skip）。

- [ ] **Step 5: Commit（R_L 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand
git add python/gesture_mapping/camera.py python/tests/test_camera.py
git commit -m "feat(camera): RealSense 增加对齐深度流 + 内参暴露"
```

---

### Task 5: handeye_calib.py（R_L）

**Files:**
- Create: `R_L/python/gesture_mapping/handeye_calib.py`
- Test: `R_L/python/tests/test_handeye_calib.py`

**Interfaces:**
- Consumes: —
- Produces: `rot_from_euler(rx_deg, ry_deg, rz_deg) -> R(3,3)`；`procrustes_rotation(src_pts, dst_pts) -> R`；`apply_rotation(R, pts(N,3)) -> (N,3)`；`save_calib(path, R)`；`load_calib(path) -> R`。Task 6/9 使用。

- [ ] **Step 1: 写失败测试**

创建 `R_L/python/tests/test_handeye_calib.py`：

```python
"""handeye_calib 纯函数单测。"""
import numpy as np
import pytest

from gesture_mapping.handeye_calib import (
    apply_rotation, load_calib, procrustes_rotation, rot_from_euler, save_calib,
)


def test_euler_zero_is_identity():
    R = rot_from_euler(0, 0, 0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_euler_x_90_rotates_y_to_z():
    R = rot_from_euler(90, 0, 0)
    np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]),
                               [0, 0, 1], atol=1e-9)


def test_procrustes_recovers_rotation():
    rng = np.random.default_rng(0)
    R_true = rot_from_euler(37, -12, 88)
    pts = rng.normal(size=(8, 3))
    dst = (R_true @ pts.T).T
    R = procrustes_rotation(pts, dst)
    np.testing.assert_allclose(R, R_true, atol=1e-9)


def test_apply_rotation():
    R = rot_from_euler(90, 0, 0)
    out = apply_rotation(R, np.array([[0.0, 1.0, 0.0]]))
    np.testing.assert_allclose(out[0], [0, 0, 1], atol=1e-9)


def test_save_load_roundtrip(tmp_path):
    R = rot_from_euler(10, 20, 30)
    p = tmp_path / "calib.json"
    save_calib(p, R)
    np.testing.assert_allclose(load_calib(p), R, atol=1e-9)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd python && pytest tests/test_handeye_calib.py -v`
Expected: `ModuleNotFoundError: No module named 'gesture_mapping.handeye_calib'`。

- [ ] **Step 3: 实现**

创建 `R_L/python/gesture_mapping/handeye_calib.py`：

```python
"""手眼标定：相机系→机器人基座系的旋转 R（差分遥操只需旋转）。

方式 A: 直接填相机安装欧拉角 → rot_from_euler。
方式 B: N 点 Procrustes（≥4 非共面）：手到已知物理位置 + 臂端对应位置。
"""
import json
from pathlib import Path
from typing import Union

import numpy as np

_Path = Union[str, Path]


def rot_from_euler(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """绕 X→Y→Z（相机系）的旋转矩阵 R(3,3)。列向量应用: v_base = R @ v_cam。"""
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def procrustes_rotation(src_pts, dst_pts) -> np.ndarray:
    """最小化 Σ||R@p_i − q_i||² 的旋转 R。src/dst: (N,3)。返回 R(3,3)。"""
    src = np.asarray(src_pts, float).T   # (3,N)
    dst = np.asarray(dst_pts, float).T   # (3,N)
    H = src @ dst.T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def apply_rotation(R: np.ndarray, pts) -> np.ndarray:
    """R(3,3) 作用于 (N,3) 点集（每行一个列向量）。"""
    pts = np.asarray(pts, float)
    return (R @ pts.T).T


def save_calib(path: _Path, R: np.ndarray) -> None:
    Path(path).write_text(json.dumps({"R": np.asarray(R).tolist()}))


def load_calib(path: _Path) -> np.ndarray:
    data = json.loads(Path(path).read_text())
    return np.array(data["R"])
```

- [ ] **Step 4: 运行确认通过**

Run: `cd python && pytest tests/test_handeye_calib.py -v`
Expected: `5 passed`。

- [ ] **Step 5: Commit（R_L 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand
git add python/gesture_mapping/handeye_calib.py python/tests/test_handeye_calib.py
git commit -m "feat(teleop): handeye_calib 旋转标定模块 + 单测"
```

---

### Task 6: arm_client.py 薄客户端（R_L）

**Files:**
- Create: `R_L/python/gesture_mapping/arm_client.py`
- Test: `R_L/python/tests/test_arm_client.py`

**Interfaces:**
- Consumes: —
- Produces: `ArmClient(port)`；`get_state() -> (angles_deg(6,), vels, loads)`；`remote_enable()`；`remote_disable()`；`e_stop()`；`remote_event(vx, vy, vz, j5, j6)`；`close()`。Task 9 demo 使用。

- [ ] **Step 1: 写失败测试（用假串口回环）**

创建 `R_L/python/tests/test_arm_client.py`：

```python
"""arm_client 单测：用 pyserial loop:// 回环捕获写入的 remote_event 命令。"""
import threading
import time

import serial  # noqa: F401  (确保 pyserial 可用)


def test_remote_event_format():
    from gesture_mapping.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://")
    c.remote_event(vx=0.7, vy=-0.3, vz=0.5, j5=0.4, j6=-0.9)
    time.sleep(0.05)
    line = s.readline().decode().strip()
    # 期望 p0=-0.700 p1=-0.300 p2=-0.900 p3=-0.400 p4=0.500 p5=-0.500
    parts = line.split()
    assert parts[0] == "remote_event"
    vals = [float(v) for v in parts[1:7]]
    assert len(vals) == 6
    assert abs(vals[0] - (-0.7)) < 1e-3   # p0=-vx
    assert abs(vals[1] - (-0.3)) < 1e-3   # p1=vy
    assert abs(vals[2] - (-0.9)) < 1e-3   # p2=j6
    assert abs(vals[3] - (-0.4)) < 1e-3   # p3=-j5
    assert abs(vals[4] - 0.5) < 1e-3      # p4=vz
    assert abs(vals[5] - (-0.5)) < 1e-3   # p5=-vz
    c.close()
    s.close()


def test_get_state_parse():
    from gesture_mapping.arm_client import ArmClient
    import serial as _s
    s = _s.serial_for_url("loop://", baudrate=115200, timeout=0.1)
    c = ArmClient("loop://")

    def feed():
        time.sleep(0.05)
        s.write(b"STATE:90.00,45.00,67.00,-157.00,0.00,5.00,"
                b"0,0,0,0,0,0,0,0,0,0,0,0\n")

    t = threading.Thread(target=feed, daemon=True)
    t.start()
    angles, _, _ = c.get_state()
    assert len(angles) == 6
    assert abs(angles[4] - 0.0) < 1e-6
    assert abs(angles[0] - 90.0) < 1e-6
    c.close()
    s.close()
```

- [ ] **Step 2: 运行确认失败**

Run: `cd python && pytest tests/test_arm_client.py -v`
Expected: `ModuleNotFoundError: No module named 'gesture_mapping.arm_client'`。

- [ ] **Step 3: 实现**

创建 `R_L/python/gesture_mapping/arm_client.py`：

```python
"""机械臂串口薄客户端（真机串口 / socket:// 仿真）。

仅实现视觉遥操所需命令，命令语义与固件一致：
  remote_event p0 p1 p2 p3 p4 p5
    vx=-p0, vy=p1, vz=(p4-p5)/2, rx=-p3(J5), ry=p2(J6)
本模块与 Arm-robot_VLA 的 serial_protocol.py 解耦（避免跨仓库运行时依赖）。
"""
import time
from typing import Tuple

import serial


class ArmClient:
    def __init__(self, port: str, baudrate: int = 115200):
        self._ser = serial.serial_for_url(
            port, baudrate=baudrate, timeout=0.05, write_timeout=0.1)
        time.sleep(0.3)
        self._ser.reset_input_buffer()

    # ── 命令 ──────────────────────────────────────────────

    def get_state(self) -> Tuple[list, list, list]:
        """读取关节状态, 返回 (angles_deg, vels, loads) 各 6 元列表."""
        self._ser.write(b"get_state\n")
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            line = self._ser.readline().decode("ascii", errors="replace").strip()
            if line.startswith("STATE:"):
                values = [float(v) for v in line[6:].split(",")]
                n = len(values) // 3
                return values[:n], values[n:2 * n], values[2 * n:]
        return [], [], []

    def remote_enable(self) -> None:
        self._ser.write(b"remote_enable\n")

    def remote_disable(self) -> None:
        self._ser.write(b"remote_disable\n")

    def e_stop(self) -> None:
        self._ser.write(b"e_stop\n")

    def remote_event(self, vx: float, vy: float, vz: float,
                     j5: float, j6: float) -> None:
        """发送差分速度命令。各输入 ∈ [-1,1]（vx/vy/vz 基座系线速度, j5/j6 关节速度）。"""
        p0, p1, p2, p3, p4, p5 = -vx, vy, j6, -j5, vz, -vz
        cmd = (f"remote_event {p0:.3f} {p1:.3f} {p2:.3f} "
               f"{p3:.3f} {p4:.3f} {p5:.3f}\n")
        self._ser.write(cmd.encode())

    def close(self) -> None:
        try:
            self._ser.close()
        except Exception:
            pass
```

- [ ] **Step 4: 运行确认通过**

Run: `cd python && pytest tests/test_arm_client.py -v`
Expected: `2 passed`。

- [ ] **Step 5: Commit（R_L 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand
git add python/gesture_mapping/arm_client.py python/tests/test_arm_client.py
git commit -m "feat(teleop): arm_client 薄客户端 (remote_event 差分速度)"
```

---

### Task 7: wrist_tracker.py 纯函数（R_L）

**Files:**
- Create: `R_L/python/gesture_mapping/wrist_tracker.py`
- Test: `R_L/python/tests/test_wrist_tracker.py`

**Interfaces:**
- Consumes: `apply_rotation`（Task 5）
- Produces 纯函数：`backproject(u, v, depth_mm, K) -> (3,)`；`median_depth_at(depth, u, v, patch=7) -> float`；`build_palm_pts(hand, depth, K) -> Optional[(21,3)]`；`palm_basis(pts21) -> (f,n,lat)`；`pitch_angle(f_base) -> rad`；`roll_angle(n_base, f_base, n_ref, f_ref) -> rad`；`delta_to_velocity(delta, gain, deadzone, max_vel=1.0) -> float`。

- [ ] **Step 1: 写失败测试**

创建 `R_L/python/tests/test_wrist_tracker.py`：

```python
"""wrist_tracker 纯函数单测。"""
import numpy as np
import pytest

from gesture_mapping.wrist_tracker import (
    backproject, build_palm_pts, delta_to_velocity, median_depth_at,
    palm_basis, pitch_angle, roll_angle,
)


def test_backproject_center():
    # u=cx,v=cy → X=Y=0, Z=depth
    xyz = backproject(320.0, 240.0, 1000.0, (500.0, 500.0, 320.0, 240.0))
    np.testing.assert_allclose(xyz, [0, 0, 1000], atol=1e-6)


def test_backproject_offcenter():
    # X = (u-cx)*Z/fx = (420-320)*1000/500 = 200
    xyz = backproject(420.0, 240.0, 1000.0, (500.0, 500.0, 320.0, 240.0))
    np.testing.assert_allclose(xyz, [200, 0, 1000], atol=1e-6)


def test_median_depth_at_ignores_outliers():
    d = np.ones((20, 20), dtype=np.uint16) * 800
    d[9:12, 9:12] = 0          # 中心飞点
    d[10, 10] = 9999           # 深噪点
    v = median_depth_at(d, 10.0, 10.0, patch=7)
    assert v == 800.0


def test_median_depth_nan_when_empty():
    d = np.zeros((20, 20), dtype=np.uint16)
    assert np.isnan(median_depth_at(d, 10, 10, patch=7))


def test_palm_basis_orthonormal():
    pts = np.array([
        [0, 0, 0],                 # wrist
        [10, 0, 5],                # index_mcp
        [12, 0, 0],                # middle_mcp
        [8, 0, -5],                # pinky_mcp
    ] + [[0, 0, 0]] * 17)          # 其余 17 点占位
    f, n, lat = palm_basis(pts)
    for v in (f, n, lat):
        assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert abs(np.dot(n, f)) < 1e-9


def test_pitch_angle_up_is_positive():
    assert abs(pitch_angle(np.array([0.0, 0.0, 1.0])) - np.pi / 2) < 1e-9
    assert abs(pitch_angle(np.array([1.0, 0.0, 0.0]))) < 1e-9


def test_roll_angle():
    # f=+z, n_ref=+x, n_now=+y → 绕 z 从 x 转 +90°
    roll = roll_angle(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]),
                      np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert abs(roll - np.pi / 2) < 1e-9


def test_delta_to_velocity_deadzone_saturation():
    assert delta_to_velocity(5.0, gain=0.01, deadzone=10.0) == 0.0
    v = delta_to_velocity(50.0, gain=0.01, deadzone=10.0)
    assert abs(v - 0.4) < 1e-9      # (50-10)*0.01
    assert delta_to_velocity(1000.0, gain=0.01, deadzone=10.0) == 1.0  # 饱和
```

- [ ] **Step 2: 运行确认失败**

Run: `cd python && pytest tests/test_wrist_tracker.py -v`
Expected: `ModuleNotFoundError: No module named 'gesture_mapping.wrist_tracker'`。

- [ ] **Step 3: 实现纯函数**

创建 `R_L/python/gesture_mapping/wrist_tracker.py`（本任务只写纯函数；`WristTracker` 类在 Task 8 追加）：

```python
"""手腕 3D + 掌参考系 → 机械臂差分速度 (核心遥操模块)。

纯函数部分: 深度反投影、掌参考系、俯仰/滚转角、delta→速度。
Task 8 追加 WristTracker 类 (动态参考 + 滤波 + J5/J6 钳制)。
"""
import math
from typing import Optional, Tuple

import numpy as np

# MediaPipe 21 点索引
_WRIST = 0
_MCP_INDEX = 5
_MCP_MIDDLE = 9
_MCP_PINKY = 17

K = Tuple[float, float, float, float]  # (fx, fy, cx, cy)


# ── 深度反投影 ──────────────────────────────────────────────

def backproject(u: float, v: float, depth_mm: float, K: K) -> np.ndarray:
    """像素 (u,v) + 深度(mm) → 相机系 3D (mm)。"""
    fx, fy, cx, cy = K
    Z = float(depth_mm)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def median_depth_at(depth: np.ndarray, u: float, v: float,
                    patch: int = 7) -> float:
    """wrist 邻域中值深度(mm)。越界/全 0 返回 nan。"""
    h, w = depth.shape
    r = patch // 2
    u0, u1 = max(0, int(u) - r), min(w, int(u) + r + 1)
    v0, v1 = max(0, int(v) - r), min(h, int(v) + r + 1)
    patch_d = depth[v0:v1, u0:u1]
    patch_d = patch_d[patch_d > 0]
    if patch_d.size == 0:
        return float("nan")
    return float(np.median(patch_d))


def build_palm_pts(hand, depth: Optional[np.ndarray],
                   K: Optional[K]) -> Optional[np.ndarray]:
    """从 HandResult + 对齐深度反投影出所需关键点 (21,3) 相机系 mm。

    任一所需关键点深度无效时返回 None。hand.landmark_xy 是 (21,2) 像素。
    """
    if depth is None or K is None:
        return None
    xy = hand.landmark_xy
    pts = np.zeros((21, 3))
    for i in (_WRIST, _MCP_INDEX, _MCP_MIDDLE, _MCP_PINKY):
        u, v = xy[i]
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            return None
        z = median_depth_at(depth, u, v)
        if not math.isfinite(z):
            return None
        pts[i] = backproject(u, v, z, K)
    return pts


# ── 掌参考系 ────────────────────────────────────────────────

def palm_basis(pts21: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(f, n, lat) 单位向量。f: wrist→中指MCP; n: 掌法线; lat: cross(f,n)。"""
    f = pts21[_MCP_MIDDLE] - pts21[_WRIST]
    f = f / (np.linalg.norm(f) + 1e-9)
    across = pts21[_MCP_PINKY] - pts21[_MCP_INDEX]
    n = np.cross(across, f)
    n = n / (np.linalg.norm(n) + 1e-9)
    lat = np.cross(f, n)
    return f, n, lat


# ── 旋转角 (相对参考) ───────────────────────────────────────

def pitch_angle(f_base: np.ndarray) -> float:
    """f 相对基座系水平面的俯仰角 (rad), f 向上为正。"""
    return math.asin(max(-1.0, min(1.0, float(f_base[2]))))


def _proj_perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    v = v - (v @ axis) * axis
    return v / (np.linalg.norm(v) + 1e-9)


def roll_angle(n_base: np.ndarray, f_base: np.ndarray,
               n_ref: np.ndarray, f_ref: np.ndarray) -> float:
    """掌滚转角 (rad): n 绕 f 轴相对参考的有向转角 (旋前为正)。"""
    a = _proj_perp(n_ref, f_ref)
    b = _proj_perp(n_base, f_ref)
    return math.atan2(np.dot(np.cross(a, b), f_ref), np.dot(a, b))


# ── delta → 速度 ────────────────────────────────────────────

def delta_to_velocity(delta: float, gain: float, deadzone: float,
                      max_vel: float = 1.0) -> float:
    """delta → [-1,1] 速度: 死区内 0, 线性增益, 饱和 max_vel。"""
    d = float(delta)
    if abs(d) < deadzone:
        return 0.0
    v = (d - math.copysign(deadzone, d)) * gain
    return float(np.clip(v, -max_vel, max_vel))
```

- [ ] **Step 4: 运行确认通过**

Run: `cd python && pytest tests/test_wrist_tracker.py -v`
Expected: 全部纯函数测试通过。

- [ ] **Step 5: Commit（R_L 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand
git add python/gesture_mapping/wrist_tracker.py python/tests/test_wrist_tracker.py
git commit -m "feat(teleop): wrist_tracker 纯函数 (反投影/掌系/角度/速度)"
```

---

### Task 8: wrist_tracker.py 的 WristTracker 类（R_L）

**Files:**
- Modify: `R_L/python/gesture_mapping/wrist_tracker.py`（追加 `WristTracker` 类）
- Test: `R_L/python/tests/test_wrist_tracker.py`（追加类测试）

**Interfaces:**
- Consumes: 纯函数（Task 7）、`apply_rotation`（Task 5）、`OneEuroFilter`（`gesture_mapping.filter`）
- Produces: `WristTracker(R, gain_pos, gain_pitch, gain_roll, deadzone_pos_mm, deadzone_ang_deg, j5_rate_deg_s, j6_rate_deg_s, j5_range, j6_range, min_cutoff, beta)`；`capture_reference(pts21_cam)`；`update(pts21_cam) -> (vx,vy,vz,j5,j6)`；`no_hand()`（清空）；`sync_j5j6(deg_j5, deg_j6)`（从 get_state 初始化）；`j5_pos_deg` / `j6_pos_deg` 属性。

- [ ] **Step 1: 写失败测试（追加到 test_wrist_tracker.py）**

```python
# ── WristTracker 类 ───────────────────────────────────────

from gesture_mapping.wrist_tracker import WristTracker
from gesture_mapping.handeye_calib import rot_from_euler


def _identity_pts21(hand_pts):
    pts = np.zeros((21, 3))
    pts[0] = hand_pts[0]       # wrist
    pts[5] = hand_pts[1]       # index_mcp
    pts[9] = hand_pts[2]       # middle_mcp
    pts[17] = hand_pts[3]      # pinky_mcp
    return pts


def test_no_hand_zeroes():
    wt = WristTracker(R=rot_from_euler(0, 0, 0))
    assert wt.update(None) == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_delta_drives_velocity():
    R = rot_from_euler(0, 0, 0)
    wt = WristTracker(R=R, gain_pos=0.001, deadzone_pos_mm=10.0)
    ref = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 0, 1000], [8, 0, 995]])
    wt.capture_reference(ref)
    # 手沿 +x 移动 50mm (wrist 移到 (50,0,1000))
    now = _identity_pts21([[50, 0, 1000], [60, 0, 1005], [62, 0, 1000], [58, 0, 995]])
    vx, vy, vz, j5, j6 = wt.update(now)
    assert vx > 0.03                     # (50-10)*0.001 = 0.04
    assert vy == 0.0 and vz == 0.0
    assert j5 == 0.0 and j6 == 0.0


def test_j5j6_clamped_to_ranges():
    wt = WristTracker(R=rot_from_euler(0, 0, 0),
                      gain_pitch=1.0, gain_roll=1.0,
                      deadzone_ang_deg=0.0,
                      j5_rate_deg_s=10.0, j6_rate_deg_s=10.0,
                      dt=1.0)
    ref = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 0, 1000], [8, 0, 995]])
    wt.capture_reference(ref)
    # 把 f 一直向上顶 → j5 应饱和在 j5_range 上限 (默认 90)
    for _ in range(50):
        now = _identity_pts21([[0, 0, 1000], [10, 0, 1005], [12, 50, 1000], [8, 0, 995]])
        wt.update(now)
    assert wt.j5_pos_deg >= 89.0
    assert wt.j6_pos_deg == 0.0
```

- [ ] **Step 2: 运行确认失败**

Run: `cd python && pytest tests/test_wrist_tracker.py -v`
Expected: `AttributeError: module 'gesture_mapping.wrist_tracker' has no attribute 'WristTracker'`。

- [ ] **Step 3: 实现 WristTracker 类（追加到 wrist_tracker.py 末尾）**

```python
# ── WristTracker: 动态参考 + 滤波 + 速度生成 + J5/J6 钳制 ──

class WristTracker:
    """把相机系 3D 手关键点流 → (vx,vy,vz,j5,j6)∈[-1,1] 差分速度。

    用法: capture_reference() 在离合器按下/松开时锚定参考; 之后每次
    update() 输出相对参考的速度。无手调用 no_hand() 清零。
    """

    def __init__(self, R: np.ndarray,
                 gain_pos: float = 0.001,          # 1/mm (50mm→0.04)
                 gain_pitch: float = 0.02,         # 1/deg (50°→0.8)
                 gain_roll: float = 0.02,          # 1/deg
                 deadzone_pos_mm: float = 15.0,
                 deadzone_ang_deg: float = 5.0,
                 j5_rate_deg_s: float = 45.0,
                 j6_rate_deg_s: float = 180.0,
                 j5_range=(0.0, 90.0), j6_range=(0.0, 360.0),
                 dt: float = 1.0 / 30.0,
                 min_cutoff: float = 1.0, beta: float = 0.02):
        self.R = np.asarray(R, float)
        self.gain_pos = gain_pos
        self.gain_pitch = gain_pitch
        self.gain_roll = gain_roll
        self.deadzone_pos_mm = deadzone_pos_mm
        self.deadzone_ang_deg = deadzone_ang_deg
        self.j5_rate_deg_s = j5_rate_deg_s
        self.j6_rate_deg_s = j6_rate_deg_s
        self.j5_range = j5_range
        self.j6_range = j6_range
        self.dt = dt
        # 状态
        self._ref_pts = None          # 参考 21 点 (相机系 mm)
        self._ref_f = None
        self._ref_n = None
        self._pos_filt = OneEuroFilter(3, min_cutoff=min_cutoff, beta=beta)
        self.j5_pos_deg = 0.0
        self.j6_pos_deg = 0.0
        self._has_ref = False

    # ── 参考 ──────────────────────────────────────────────

    def capture_reference(self, pts21_cam: Optional[np.ndarray]) -> None:
        if pts21_cam is None:
            return
        self._ref_pts = np.asarray(pts21_cam, float)
        f, n, _ = palm_basis(self._ref_pts)
        self._ref_f = apply_rotation(self.R, np.array([f]))[0]
        self._ref_n = apply_rotation(self.R, np.array([n]))[0]
        self._pos_filt.reset()
        self._has_ref = True

    def sync_j5j6(self, deg_j5: float, deg_j6: float) -> None:
        """从 get_state 同步 J5/J6 实际角度 (remote_enable 软复位后调用)."""
        self.j5_pos_deg = float(np.clip(deg_j5, *self.j5_range))
        self.j6_pos_deg = float(np.clip(deg_j6, *self.j6_range))

    # ── 主更新 ────────────────────────────────────────────

    def update(self, pts21_cam: Optional[np.ndarray]):
        """返回 (vx,vy,vz,j5_cmd,j6_cmd) ∈ [-1,1]。无手/无参考 → 全 0。"""
        if pts21_cam is None or not self._has_ref:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        pts = np.asarray(pts21_cam, float)

        # 位置 delta (基座系) + 平滑
        wrist = apply_rotation(self.R, np.array([pts[_WRIST]]))[0]
        wrist = self._pos_filt(wrist)
        ref_w = apply_rotation(self.R, np.array([self._ref_pts[_WRIST]]))[0]
        delta = wrist - ref_w
        vx = delta_to_velocity(delta[0], self.gain_pos, self.deadzone_pos_mm)
        vy = delta_to_velocity(delta[1], self.gain_pos, self.deadzone_pos_mm)
        vz = delta_to_velocity(delta[2], self.gain_pos, self.deadzone_pos_mm)

        # 姿态角 delta
        f, n, _ = palm_basis(pts)
        f_base = apply_rotation(self.R, np.array([f]))[0]
        n_base = apply_rotation(self.R, np.array([n]))[0]
        pitch_deg = math.degrees(pitch_angle(f_base))
        roll_deg = math.degrees(roll_angle(n_base, f_base, self._ref_n, self._ref_f))
        ref_pitch = math.degrees(pitch_angle(self._ref_f))
        j5_cmd = delta_to_velocity(pitch_deg - ref_pitch,
                                   self.gain_pitch, self.deadzone_ang_deg)
        j6_cmd = delta_to_velocity(roll_deg, self.gain_roll, self.deadzone_ang_deg)

        # J5/J6 位置跟踪 + 边界钳制 (固件无限位, PC 侧负责)
        self.j5_pos_deg = float(np.clip(
            self.j5_pos_deg + j5_cmd * self.j5_rate_deg_s * self.dt,
            *self.j5_range))
        self.j6_pos_deg = float(np.clip(
            self.j6_pos_deg + j6_cmd * self.j6_rate_deg_s * self.dt,
            *self.j6_range))
        if (self.j5_pos_deg <= self.j5_range[0] and j5_cmd < 0) or \
           (self.j5_pos_deg >= self.j5_range[1] and j5_cmd > 0):
            j5_cmd = 0.0
        if (self.j6_pos_deg <= self.j6_range[0] and j6_cmd < 0) or \
           (self.j6_pos_deg >= self.j6_range[1] and j6_cmd > 0):
            j6_cmd = 0.0

        return (vx, vy, vz, j5_cmd, j6_cmd)

    def no_hand(self):
        """无手帧: 输出全 0 (速度命令清零, 臂保持)。"""
        self._pos_filt.reset()
        return (0.0, 0.0, 0.0, 0.0, 0.0)
```

并在 wrist_tracker.py 顶部 import 区补：

```python
from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.handeye_calib import apply_rotation
```

（`filter`/`handeye_calib` 均不依赖 wrist_tracker，可直接顶层 import。）

- [ ] **Step 4: 运行确认通过**

Run: `cd python && pytest tests/test_wrist_tracker.py -v`
Expected: 全部通过（含 3 个类测试）。

- [ ] **Step 5: Commit（R_L 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand
git add python/gesture_mapping/wrist_tracker.py python/tests/test_wrist_tracker.py
git commit -m "feat(teleop): WristTracker 动态参考 + J5/J6 钳制"
```

---

### Task 9: demo_arm_teleop.py（R_L）

**Files:**
- Create: `R_L/python/gesture_mapping/demo_arm_teleop.py`

**Interfaces:**
- Consumes: `open_realsense`（Task 4）、`HandTracker`（已有）、`ArmClient`（Task 6）、`WristTracker`/`build_palm_pts`（Task 7/8）、`load_calib`（Task 5）
- Produces: 可运行 demo。命令行：`--port`（默认 `socket://localhost:5555`）、`--calib`（默认 `python/gesture_mapping/handeye_calib.json`）、`--no-drive`（只显示速度不发送）。

- [ ] **Step 1: 实现 demo**

创建 `R_L/python/gesture_mapping/demo_arm_teleop.py`：

```python
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

    if arm is not None:
        arm.remote_disable()
        arm.close()
    cam.release()
    cv2.destroyAllWindows()
    print("[退出] 已安全断开")


if __name__ == "__main__":
    main()
```

> 说明：H 键做 toggle（按一下 ON、再按 OFF）。ON 时 capture 参考锚定当前位置、随后跟随；OFF 时再次 capture（锚定新参考）。这等价于"松开即记参考"，即设计文档 §6.1 的动态参考语义。

- [ ] **Step 2: 冒烟运行（无硬件，验证可 import）**

Run: `conda activate leap_hand && cd python && python -c "import gesture_mapping.demo_arm_teleop; print('import ok')"`
Expected: `import ok`（不连接臂、不依赖相机）。

- [ ] **Step 3: 生成默认标定文件（单位旋转占位，M2 实测后替换）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand/python
python - <<'EOF'
from pathlib import Path
from gesture_mapping.handeye_calib import rot_from_euler, save_calib
save_calib("gesture_mapping/handeye_calib.json", rot_from_euler(0, 0, 0))
print("已写入 gesture_mapping/handeye_calib.json (单位旋转, 待实测标定)")
EOF
```

- [ ] **Step 4: Commit（R_L 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand
git add python/gesture_mapping/demo_arm_teleop.py python/gesture_mapping/handeye_calib.json
git commit -m "feat(teleop): demo_arm_teleop 视觉遥操主循环"
```

---

### Task 10: M2 端到端验证 + 文档收尾

**Files:**
- Modify: `R_L/.claude/workstreams/01-gesture-mapping.md`（手腕 6DOF 待办更新）
- Modify: `R_L/docs/design/2026-08-11-arm-visual-teleop-design.md`（如有实测调整，记录）

**Interfaces:**
- Consumes: Task 1-9 全部
- Produces: 端到端验证记录 + workstream 更新。

- [ ] **Step 1: 启动仿真臂（终端 A）**

```bash
conda activate smolvla
cd /home/bright/win_office/ubantu_files/project/Arm-robot_VLA
python scripts/mujoco_sim.py --ik --no-camera
```

- [ ] **Step 2: 启动视觉遥操（终端 B）**

```bash
conda activate leap_hand
cd /home/bright/win_office/ubantu_files/project/Leap_Hand/python
python gesture_mapping/demo_arm_teleop.py --port socket://localhost:5555
```

- [ ] **Step 3: 验证清单（逐项勾选）**

- [ ] 手在相机前**上下/左右/前后**移动（按住 H）→ 仿真臂末端对应方向移动，幅度成比例
- [ ] 松手/松开 H → 臂停止不动
- [ ] 手**上翘/下压** → J5（末端上下）角度变化
- [ ] 手**翻转（旋前/旋后）** → J6（末端旋转）角度变化
- [ ] 手移出视野 → 速度清零，臂保持
- [ ] 无手超时 → 固件/仿真 0.3s 后归零
- [ ] 按 Y → e_stop 生效
- [ ] 记录：若 J5/J6 方向反或轴对调，只改 R_L 侧符号常量，并在设计文档"实测调整"节记录

- [ ] **Step 4: 更新 workstream 文档**

在 `R_L/.claude/workstreams/01-gesture-mapping.md` 的 Current State 中把"手腕 6DOF 空间定位"标记为推进中，并在 Next Tasks 中更新：

```markdown
4. ✅ 手腕 3D 位置估计 (D455 深度反投影, wrist_tracker.py)
5. 🟡 手腕位置 → 机械臂 IK 解算 (差分速度 remote_event, demo_arm_teleop.py; M2 端到端待真机验证)
```

- [ ] **Step 5: Commit（R_L 仓库）**

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand
git add .claude/workstreams/01-gesture-mapping.md docs/design/2026-08-11-arm-visual-teleop-design.md
git commit -m "docs: 手腕6DOF→机械臂遥操 推进记录"
```

---

## 自审清单（plan 编写时已过）

- **Spec 覆盖**：设计文档 §4（语义对齐）→ Task 1/2；§5（反投影/掌系/角度）→ Task 4/7；§6（动态参考/标定/钳制）→ Task 5/8；§7 模块表 → Task 0-9；§8 里程碑 → Task 10；§9 风险 #1 仿真语义 → Task 1/2；#3 J5/J6 钳制 → Task 8；#5 镜像坐标 → Task 4 备注 + M2；#6 文档修正 → Task 3；#7 滚转解卷 → 用相对参考的有向角（M2 视需要加 unwrap）。
- **占位符扫描**：无 TBD/TODO/“适当处理”。唯一"实现期定"项（cross-repo 依赖）已落地为 Task 6 薄客户端。
- **类型一致**：`parse_remote_event` 在 Task 1 定义、Task 2 使用；`backproject/median_depth_at/build_palm_pts/palm_basis/pitch_angle/roll_angle/delta_to_velocity` 在 Task 7 定义、Task 8/9 使用；`rot_from_euler/apply_rotation/load_calib` Task 5 定义、Task 8/9 使用；`ArmClient.remote_event(vx,vy,vz,j5,j6)` Task 6 定义、Task 9 使用；`read_with_depth()` 返回 4 元组 Task 4 定义、Task 9 解包。
