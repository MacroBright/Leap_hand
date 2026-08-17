# 机械臂视觉遥操设计

> 2026-08-11 | 差分速度控制 · D455 深度反投影 · 仿真先行
> 关联：Leap_Hand W1（`wrist_tracker` 收尾）、Arm-robot_VLA 遥操作

> ⚠ **路径迁移注 (2026-08)**：本设计文档描述的 `handeye_calib.py` / `arm_client.py` /
> `demo_arm_teleop.py` 已迁至 `Arm-robot_VLA/scripts/`。`wrist_tracker.py` /
> `camera.py` / `filter.py` 仍留 Leap_Hand, 由 `demo_arm_teleop.py` 通过 sys.path
> 跨仓库导入。下方表格里的"位置"列保留迁移前路径仅作历史记录, 实施以新位置为准。

---

## 1. 目标

将人手关键点（MediaPipe 21 点 + D455 深度）实时映射到 zero-robotic-arm 末端：

| 人手信号 | 机械臂效果 |
|---------|-----------|
| 手腕 3D 位置偏移（相对动态参考位） | 末端位置（vx/vy/vz 差分速度 → J1-J4 IK） |
| 手腕俯仰（手指向竖直面内点头） | J5 —— 末端上下 |
| 掌滚转（前臂旋前/旋后） | J6 —— 末端旋转 |

约束：**仿真先行**，机械臂侧零改动，安全网关（STM32）完整保留。

---

## 2. 总体架构

```
┌─ Leap_Hand PC ────────────────────────────────────────────┐
│ D455(color+depth) → MediaPipe 21点 → wrist 2D 像素          │
│   → rs.align 深度反投影 → wrist 3D(相机系, mm)              │
│   → hand-eye 旋转 R → wrist 3D(机器人基座系)                │
│   → delta = wrist − 动态参考位(按键捕获)                    │
│   → 死区 / 缩放 / 饱和 / OneEuro滤波                        │
│   → (vx,vy,vz, j5_vel, j6_vel) ∈ [-1,1]                    │
│   → SerialProtocol.remote_event @50Hz                       │
└───────────────┬───────────────────────────────────────────┘
                │ serial (真机) / socket://localhost:5555 (仿真)
┌───────────────▼───────────────────────────────────────────┐
│ 机械臂: STM32 robot_pid_remote() @50Hz                     │
│   vx/vy/vz → 位置积分 + IK + PID (J1-J4)                   │
│   rx → J5 关节速度 · ry → J6 关节速度                      │
│   — 或 — MuJoCo 仿真 (mujoco_sim.py --ik, Jacobian)        │
└────────────────────────────────────────────────────────────┘
```

**核心思路**：把"物理手柄"换成"视觉手柄"——手偏离动态参考位的量 → 速度命令，走现有 `remote_event` 路径。IK 已在固件内运行，视觉侧只产出速度。

---

## 3. 关节物理语义（真机确认）

**⚠️ 文档标注与实际物理不符，需修正**（见 §9.5）。以实测为准：

| 关节 | 文档标注 | 物理真实 | remote 模式 | 本设计控制 |
|------|---------|---------|------------|-----------|
| J1 | shoulder_pan | 底座旋转 | 位置 IK 环 | — |
| J2 | shoulder_lift | 肩部抬升 | 位置 IK 环 | — |
| J3 | elbow_flex | 肘部 | 位置 IK 环 | — |
| J4 | wrist_flex | **末端旋转** | 位置 IK 环 | 由 IK 决定，Phase 1 不独立控制 |
| J5 | wrist_roll | **末端上下** | `rx` 开环关节速度 | 手腕俯仰角 → rx |
| J6 | gripper | **末端旋转** | `ry` 开环关节速度 | 掌滚转角 → ry |

远程模式下 J1-J4 被固件 PID 位置环占用（[robot.c](firmware/src/robot.c) `robot_pid_remote()` 对前 4 关节做 IK+PID），**只有 J5/J6 可由 `remote_event` 的 rx/ry 开环驱动**。位置 IK 在积分时固定末端方向为复位姿态（`T_0_6_reset`）。

---

## 4. `remote_event` 协议语义（与真机固件对齐）

### 4.1 固件真实语义（[robot_cmd.c](firmware/src/robot_cmd.c) `robot_remote_event_handle`）

```
接收: remote_event p0 p1 p2 p3 p4 p5
  vx = -p0 × MAX_VEL          # 基座系 x 线速度（取负）
  vy =  p1 × MAX_VEL          # 基座系 y 线速度
  vz = (p4 − p5)/2 × MAX_VEL  # 基座系 z 线速度（两参数差）
  rx = -p3 × MAX_RPM          # J5 关节速度（取负）
  ry =  p2 × MAX_RPM          # J6 关节速度
```

`robot_pid_remote` 对 vx/vy/vz 做**线性位置积分**（非柱坐标），每 50ms 一次 IK+PID。

### 4.2 客户端生成公式（视觉遥操）

设期望速度 `(vx, vy, vz, j5, j6) ∈ [-1,1]`：

```
p0 = -vx
p1 =  vy
p2 =  j6          # J6（掌滚转）
p3 = -j5          # J5（腕俯仰）
p4 =  vz
p5 = -vz          # 使 (p4−p5)/2 = vz
```

### 4.3 ⚠️ 仿真语义不一致（M1 必修）

`mujoco_sim.py` 当前把 `remote_event` 参数当**柱坐标 (θ,r)** 并转换到 Cartesian（[mujoco_sim.py](scripts/mujoco_sim.py) `_cmd_remote_event` / step 内 `_vx = v_radius·cosθ − r·v_theta·sinθ`），且把 p2/p3 当**末端角速度**（Jacobian 转动），z 符号也与固件相反。

**行为**：仿真改为与固件**逐字节一致**：
- p0→-vx, p1→vy, (p4−p5)/2→vz（线性 Cartesian，非柱坐标）
- p2→J6 关节速度、p3→J5 关节速度（直接写 `target_vel[4]`/`target_vel[5]`，非末端角速度）
- 对齐后再仿真的行为即真机行为

---

## 5. 手腕 3D 与掌参考系（相机系）

### 5.1 深度反投影（`wrist_tracker`）

- D455 开 color(640×480@30) + depth(z16)，`rs.align(rs.stream.color)` 对齐
- 内参 `(fx, fy, cx, cy)` 由 SDK 直接给出
- wrist 像素 `(u,v)`（注意 demo 的 `cv2.flip` 镜像后需还原到相机原坐标系）
- 深度取 wrist 邻域 **7×7 中值**（去飞点/噪声）：
  ```
  Z = depth_median / 1000          # mm → m（或统一 mm）
  X = (u − cx) · Z / fx
  Y = (v − cy) · Z / fy
  ```

### 5.2 掌参考系（从深度 3D landmark 构造）

取 wrist + 三指 MCP（index=5, middle=9, pinky=17）：

```
f   = normalize(mid_mcp − wrist)            # 前臂/手指方向
n   = normalize(cross(pinky_mcp − index_mcp, f))   # 掌法线（符号定标后朝掌心外）
lat = cross(f, n)
```

### 5.3 旋转角定义（相对动态参考）

- **手腕俯仰角**（→ J5）：`f`（mid_dir）相对参考的竖直俯仰角，即 `f` 与水平面的夹角变化。俯仰 = 手上翘/下压。
- **掌滚转角**（→ J6）：`n`（掌法线）绕 `f` 轴的滚转角（pronation/supination），即 `n` 在垂直于 `f` 的平面上的投影角相对参考的变化。

二者均经 hand-eye 旋转 R 转至基座系后与参考比较（与位置 delta 同一变换，见 §6.2）。

---

## 6. 动态参考与速度生成

### 6.1 动态参考位（用户确认）

- 按键（clutch）**松开**的瞬间，捕获当前位置为参考 `ref_pos`、当前掌参考系为参考 `ref_f/ref_n`
- 按住 clutch 期间：`delta_pos = wrist − ref_pos`，`delta_pitch = pitch − ref_pitch`，`delta_roll = roll − ref_roll`
- 位置 → 速度：`v = k_p · delta_pos`（饱和到 [-1,1]）
- 松开回参考位 → delta=0 → 停；走到哪停到哪（自然）

### 6.2 手眼标定

差分模式**平移项抵消**，只需旋转 R：`delta_base = R · (wrist_cam − ref_cam)`。同样 R 也作用于掌参考系向量。

- 方式 A（推荐首用）：按相机安装角度直接填 3 个欧拉角到 `handeye_calib.json`
- 方式 B：N 点 Procrustes（≥4 非共面点，手到已知位置 + 臂端对应位置）

### 6.3 滤波与安全钳制

- **死区** ~15mm（位置）、~3°（角度）——手微动不触发
- **饱和**：delta 超过阈值封顶
- **OneEuro 滤波**：位置 3D + 角度各一级（参考跟手经验，滤波越轻越好，调 `min_cutoff`/`beta`）
- **J5/J6 clamp**：远程模式下 J5/J6 是固件开环速度驱动（无文档化限位钳制），PC 侧需自行把 J5/J6 目标位置 clamp 到 **J5: 0-90° / J6: 0-360°**（速度模式下做位置跟踪 + 边界钳制）
- **无手/超时**：清空速度命令；固件 0.3s remote 超时兜底
- **e_stop**：按键绑定

---

## 7. 模块划分

| # | 文件 | 仓库 | 改动 | 说明 |
|---|------|------|------|------|
| 1 | [camera.py](python/gesture_mapping/camera.py) | Leap_Hand | 改 | `RealSenseSource` 增加 depth 流 + `rs.align` + 内参暴露，保持 `read()` 兼容 |
| 2 | `wrist_tracker.py` | Leap_Hand | 新 | **核心**：深度反投影 + 掌参考系 + 动态参考 + delta→速度 + 安全钳制 |
| 3 | `handeye_calib.py` | Leap_Hand | 新 | hand-eye R 求解/存取（欧拉角 / Procrustes）→ JSON |
| 4 | `demo_arm_teleop.py` | Leap_Hand | 新 | 主循环 + 按键（clutch/标定/e_stop/退出）+ HUD |
| 5 | `tests/test_wrist_tracker.py` | Leap_Hand | 新 | 单测：合成深度、变换、delta→速度、死区、clamp |
| 6 | [mujoco_sim.py](scripts/mujoco_sim.py) | Arm-robot_VLA | 改 | remote_event 语义与固件逐字节对齐（§4.3） |
| 7 | [SERIAL_COMMANDS.md](docs/SERIAL_COMMANDS.md) | Arm-robot_VLA | 改 | 修正 J4/J5 命名标注（§9.5） |

跨仓库依赖：`demo_arm_teleop.py` 需复用 Arm-robot_VLA 的 `serial_protocol.py`（真机串口 / 仿真 `socket://`）。以最小耦合方式引用（sys.path 或复制薄封装），实现期定。

---

## 8. 里程碑与验收

### M1 仿真对齐 + 单测（纯代码）
- [ ] `mujoco_sim.py` remote_event 语义对齐固件，用脚本逐参数验证（vx/vy/vz/rx/ry 各轴方向）
- [ ] `camera.py` depth 扩展、`wrist_tracker.py`、`handeye_calib.py`、单测通过

### M2 真手 + 仿真臂端到端（安全验证核心）
- [ ] D455 前挥手 → 仿真臂末端对应方向移动、幅度成比例、松手即停
- [ ] 手腕俯仰 → J5 上下、掌滚转 → J6 旋转，验证轴分配/符号（可能需调）
- [ ] J5 俯仰与 vz 位置环耦合评估（死区调优）
- [ ] 无手/超时停、clutch 行为验证

### M3 接真机 STM32
- [ ] 真机串口验证（USB），安全参数实测调优
- [ ] e_stop、限位、急停实测

**验收标准**：仿真中操作者手移到目标位 → 臂末端到达对应位置并停住；J5/J6 跟随手腕姿态；全程无电机冲击；超时/无手自动停。

---

## 9. 风险与开放问题

| # | 风险 | 处置 |
|---|------|------|
| 1 | 仿真/真机 remote_event 语义不一致 | §4.3 仿真对齐（M1 前提） |
| 2 | J5（腕俯仰）与 vz（位置上抬）耦合 | M2 实测，J5 设大死区 / 调增益 |
| 3 | J5/J6 开环速度无固件限位钳制 | PC 侧 clamp 到 0-90/0-360（§6.3） |
| 4 | D455 深度在 wrist 处噪声/飞点 | 7×7 中值 + 时间滤波；失败时保持上一速度或清零 |
| 5 | 镜像坐标还原 | `cv2.flip` 后像素需逆变换回相机系（§5.1） |
| 6 | J4/J5 文档命名与实际物理不符 | 设计含文档修正任务（§7 表 #7） |
| 7 | 掌法线在 ±90° 滚转时翻转歧义 | 用解卷（unwrap）追踪连续角度，避免跳变 |

---

## 10. 参考

- 固件 remote_event：Arm-robot_VLA `firmware/src/robot_cmd.c` / `robot.c`
- 仿真 remote_event：Arm-robot_VLA `scripts/mujoco_sim.py`
- 串口协议：Arm-robot_VLA `docs/SERIAL_COMMANDS.md`、`lerobot_robot_massage/serial_protocol.py`
- 手部检测：Leap_Hand `python/gesture_mapping/hand_tracker.py` / `camera.py`
- 掌参考系：Leap_Hand `python/gesture_mapping/joint_mapper.py` `_palm_frame()`
- 相关调研：`docs/design/2026-08-10-teleop-following-implementation.md`
