# 基于 LeRobot 的机械臂 + LEAP Hand 通信接入、视觉遥操与数据采集操作方案

> 2026-08-14 | 整合：Arm-robot_VLA（机械臂/STM32/ZDT）+ Leap_Hand（灵巧手/视觉遥操）
> 关联：[机械臂视觉遥操设计](../design/2026-08-11-arm-visual-teleop-design.md) · [操作手册](../2026-08-12-arm-visual-teleop-operation-manual.md) · [串口命令参考](../../Arm-robot_VLA/docs/SERIAL_COMMANDS.md)

---

## 0. 目标与范围

把「6-DOF 机械臂（ZDT 步进闭环电机，经 STM32 安全网关）」+「LEAP Hand 14-DOF 灵巧手（Dynamixel XC330×16，PC 直连）」整合为一个 **LeRobot 机器人（22 DOF）**，接入现有**人手视觉遥操**，用 LeRobot 框架完成**真机数据采集**（图像 + 关节状态 + 动作），为 SmolVLA 训练铺路。

```
┌─ 操作者人手 ──────────────────────────────┐
│  D455(RGB-D) → MediaPipe 21点             │
│    ├─ 腕 6DOF → WristTracker → 臂目标      │  ← 已有 (Leap_Hand)
│    └─ 手势 → joint_mapper → 手 16DOF       │  ← 已有 (demo_hamer3d)
└───────────────┬──────────────────────────┘
                │ 视觉遥操 (动作来源)
┌───────────────▼──────────────────────────┐
│  LeRobot 采集层 (本方案新建)               │
│  observation = 臂关节(6) + 手关节(16) + 图像 │
│  action      = 关节绝对位置 (22 DOF)       │
└───────┬──────────────────────┬───────────┘
        │ 串口文本协议            │ Dynamixel 4M
┌───────▼────────┐     ┌────────▼──────────┐
│ STM32 (安全网关) │     │ LEAP Hand 16电机   │
│  └→ ZDT 电机×6  │     └───────────────────┘
└────────────────┘
```

**关键边界**：STM32 仍是唯一直接控制臂电机的设备（ADR-001 保留）；灵巧手 PC 直连（Leap_Hand ADR-001 保留）。两条总线协议完全不同，互不干扰。

---

## 1. 现状盘点（已完成，直接复用）

| 模块 | 位置 | 状态 |
|------|------|------|
| 人手 6DOF → 末端位姿 → `end_event` 6DOF 速度 | `Leap_Hand/python/gesture_mapping/wrist_tracker.py` + `demo_arm_teleop.py` | ✅ 仿真验证通过（P2+P3），`WristTracker.update()` 每帧产出 `(vx,vy,vz,wx,wy,wz)` |
| 手势 → LEAP 16DOF | `Leap_Hand/python/gesture_mapping/demo_hamer3d.py` + `joint_mapper.py` | ✅ 可用 |
| 臂串口客户端（真机/仿真） | `Leap_Hand/python/gesture_mapping/arm_client.py` | ✅ `get_state/end_event/set_joints/remote_event/e_stop` |
| 手眼标定 K（手→基座系旋转） | `handeye_calib.json` + K 向导 | ✅ |
| 臂 LeRobot 适配器骨架（仅臂 6DOF） | `Arm-robot_VLA/lerobot_robot_massage/massage_robot.py` | ✅ 已实现 `Robot` 子类，`get_observation`/`send_action`/`calibrate`/`emergency_stop`；手部为 TODO |
| 仿真臂（STM32 协议逐字节对齐） | `Arm-robot_VLA/scripts/mujoco_sim.py`（TCP 5555 + 共享内存相机） | ✅ 含 `get_ee_pose`/`end_event` 扩展，可作采集验证环境 |
| LeRobot 0.4.4 | `smolvla` conda env | ✅ `lerobot_record/teleoperate/eval/train` CLI 齐全 |

**关键结论**：视觉遥操与 LeRobot 数据写入两条管线都已有可复用的地基，本方案把它们**接起来**，缺的是三块——①STM32 固件从 Emm_V5 改为 ZDT 电机；②一个臂+手合并的 LeRobot `Robot` 子类（22 DOF）；③一个把视觉遥操接到 LeRobot 数据写入的采集器。

---

## 2. 已确认的关键决策

| # | 决策 | 结论 | 影响 |
|---|------|------|------|
| D1 | 机械臂电机拓扑 | **ZDT 电机 + STM32 网关** | STM32 固件底层从 Emm_V5 CAN 换为 ZDT 协议；PC 侧串口命令接口（`get_state/set_joints/remote_event`）**保持不变**，PC 臂代码基本不改 |
| D2 | 灵巧手接入 LeRobot | **复用现有 `leap_hand_utils.dynamixel_client`**，不强行套 `DynamixelMotorsBus` | 保留 LEAP 专属角度约定/方向/校准；无需给 LeRobot tables.py 补 xc330 型号 |
| D3 | action 空间（训练友好） | **关节绝对位置，22 DOF**（6 臂 + 16 手） | 标准 ACT/SmolVLA 格式；动作 = 实际执行轨迹（延迟状态 BC 标签），与现有 `convert_to_lerobot.py` 约定一致 |
| D4 | 采集实现 | **自定义采集循环 + `LeRobotDataset` 直接写入**（复用 LeRobot 数据基础设施） | 避开 `lerobot_record` 的 `get_action→send_action` 与视觉遥操双驱动冲突；同时保留标准 Teleoperator 路径为后续 HIL/评估用 |

---

## 3. 通信接入层设计

### 3.1 臂：STM32 网关 → ZDT 电机（固件改造，Arm-robot_VLA 侧）

**现状**：STM32 固件 `robot.c`/`robot_cmd.c` 用 Emm_V5 CAN 帧控制电机；PC↔STM32 为 115200 行文本协议（14 条命令）。

**目标**：PC 侧命令不变，STM32 底层换 ZDT。ZDT 说明书（`基于VLA的机械臂设计/参考资料/ZDT_X系列_V2步进闭环驱动说明书Rev1.0.md` §7）关键协议：

| PC 命令（不变） | STM32 底层应发的 ZDT 命令 | 说明 |
|----------------|--------------------------|------|
| `get_state`（读角度/速度/电流） | `0x36` 读位置（×10）、`0x35` 读转速（×10）、`0x27` 读相电流（mA） | 逐轴轮询，或 `0x43 0x7A` 批量读 37 字节 |
| `set_joints`（位置控制） | `0xFB` 直通限速位置 / `0xFD` 梯形位置（绝对标志=01） | 位置字段 ×10，符号位 + 3 字节大端 |
| `remote_event`（速度遥操） | `0xF6` 速度模式（速度 ×10，RPM） | STM32 的 IK/位置积分逻辑保留在固件层 |
| `set_torque` | `0xF3 0xAB` 使能/不使能 | `01` 使能 / `00` 不使能 |
| `e_stop` | `0xFE 0x98` 立即停止（可广播 `00`） | 三种模式通用 |
| 回零 | `0x9A` 触发回零、`0x93 0x88` 设零点 | 推荐多圈无限位碰撞回零（机械臂无额外限位） |
| 同步 | 各命令多机同步标志=01 + 广播 `00 FF 66 6B` 同时启动 | 消除轴间启动时差 |

**ZDT 通信参数**：帧 = `地址(1B) + 功能码 + (辅助码) + 数据 + 校验`；默认校验字节 `0x6B`（可切 XOR / CRC-8，附录 12.1 有 CRC 表）；地址 1-255，`0` 广播；串口 TTL ≤4 台、RS485 理论 256 台、CAN 扩展帧默认 500K。
**STM32↔ZDT 接线**：优先复用 STM32 已有 CAN 外设接 ZDT CAN（帧格式改为 ZDT 扩展帧）；或 STM32 UART → RS485 → ZDT。**注意** ZDT 位置/速度字段放大倍数不一致（位置×10、误差×100），换算务必在固件统一封装。

**固件新增（视觉遥操 M3 遗留）**：真机目前无 `end_event`（6DOF 全 IK）与 `get_ee_pose`（末端位姿反馈）。视觉遥操要用 6DOF 位姿跟随需在 STM32 补这两条（`end_event` → 全 6×6 DLS IK；`get_ee_pose` → FK 输出 xyz+wxyz）。仿真已实现，可直接移植算法。

**验收**：真机串口 `get_state` 返回真实 ZDT 读数；`set_joints` 到位；`remote_event` 各方向正确；`e_stop` 立即停。

### 3.2 手：LEAP Hand → 现有 DynamixelClient（Leap_Hand 侧）

**现状**：`Leap_Hand/python/leap_hand_utils/dynamixel_client.py` 已封装 16 电机（ID 0-15，XC330，Protocol 2.0，4M baud）的 `read_pos_vel_cur` / `write_desired_pos` / `set_torque_enabled`。LEAP 专属角度约定（signed/unsigned、`leap_hand_utils.py` 的 `LEAPsim_to_LEAPhand` 等）已在包内。

**接入方式**：在 LeRobot `Robot` 子类里**组合复用** `DynamixelClient`（不继承 `DynamixelMotorsBus`），只暴露：
- 读：`hand_joint_i.pos` = 16 维手部关节角（用项目现有读取路径，注意 `_is_valid_pose` 安全门拦截坏数据）
- 写：`set_positions(16 维目标)`，带 `motor_limits.json` 钳制

**校准**：`poses.json` 的 8 个姿势即为「校准参考」；`is_calibrated` 直接返回 True（避免 connect 自动触发意外动作，与现有 `MassageRobot` 一致）。

### 3.3 `MassageRobotV2` — 臂+手合并的 LeRobot `Robot` 子类

建议在 `Arm-robot_VLA/lerobot_robot_massage/` 下新建 `massage_robot_v2.py`（或扩展现有类，加 `hand_enabled` 标志）：

```python
@RobotConfig.register_subclass("massage_robot_v2")
@dataclass
class MassageRobotV2Config(RobotConfig):
    port: str = "/dev/ttyUSB0"          # STM32 串口
    baudrate: int = 115200
    hand_port: str = "/dev/ttyUSB1"     # LEAP Hand USB
    arm_joint_names: list[str] = [...6 个...]
    hand_joint_names: list[str] = [...16 个...]
    cameras: dict[str, CameraConfig] = {"cam_top": OpenCVCameraConfig(...)}

class MassageRobotV2(Robot):
    config_class = MassageRobotV2Config
    # __init__: 建 SerialProtocol(臂) + DynamixelClient(手) + 相机（构造时，observation_features 连接前可调）
    # get_observation(): 臂 get_state(6) + 手 read_pos(16) + cam.read_latest() → {"shoulder_pan.pos":..., "hand_0.pos":..., "cam_top":...}
    # send_action(): 按 key 分发 — "*.pos" 中臂 6 个 → set_joints / 手 16 个 → set_positions；返回实际发送（含钳制）
    # emergency_stop(): e_stop + set_torque_enabled(False)
    # is_calibrated: True（手/臂校准均以文件为准）
```

**要点**：
- `observation_features` / `action_features` 必须**连接前可调用**（不读传感器）—— 相机在 `__init__` 建好即可。
- 键约定 `{joint}.pos`；LeRobot 自动加 `observation.state` / `action` 前缀。
- 臂 `send_action` 走 `set_joints`（位置控制）——注意 `serial_protocol.set_joints` 目前 `read_until_ok(timeout=2s)`，30fps 下若 STM32 回复慢会阻塞，需实测；慢则改 fire-and-forget 或提高波特率。
- 手 `read_pos` 失败时用上一帧（现有 `DynamixelReader` 已有此逻辑），防采集循环崩溃。

### 3.4 相机

- 场景相机（记录按摩对象）：`cam_top` OpenCV（D455 RGB 或普通 USB），640×480@30 已验证。
- 遥操相机：D455 RGB-D（depth+color），复用现有 `open_realsense()`。
- 若 D455 兼任场景相机，两个 D455/同型号别插同一 USB HUB（LeRobot 已知坑）。推荐场景相机 + 遥操相机分开，画面命名 `top/side/wrist`。

---

## 4. 视觉遥操接入

### 4.1 现有遥操链路（不改动）
`D455 → HandTracker → build_palm_pts → WristTracker.update() → (vx,vy,vz,wx,wy,wz)`；手部 `demo_hamer3d.py` → 16 DOF。两者独立、都已可用。

### 4.2 两种接入路径

**路径 1（主推，数据采集）：自定义采集循环** —— 视觉遥操**直接驱动**臂+手，同时每帧写 `LeRobotDataset`：
```
每帧:
  1. 读相机帧 + 人手关键点
  2. 臂: WristTracker → end_event → arm_client(STM32/仿真)     # 驱动
  3. 手: joint_mapper → 16 目标 → dynamixel_client.set_positions  # 驱动
  4. observation = 臂 get_state(6) + 手 read_pos(16) + 相机帧
  5. action = 上一帧 observation.state（延迟状态 = BC 标签）     # 记录
  6. dataset.add_frame({...})                                    # LeRobotDataset 写入
```
按键：`H` 离合器、`SPACE` 开/停 episode、`R` 复位、`Y` 急停、`Q` 退出。复用 `demo_arm_teleop.py` 主循环结构 + 键盘事件。

**路径 2（标准，后续 HIL/评估）：`VisualHandTeleoperator` + `lerobot_record`**
```python
@TeleoperatorConfig.register_subclass("visual_hand")
@dataclass
class VisualHandTeleopConfig(TeleoperatorConfig): ...   # 相机/标定/增益参数

class VisualHandTeleoperator(Teleoperator):
    def get_action(self):   # 返回 22 DOF 目标（臂 6 位置 + 手 16 位置）
        ...
```
然后用 `lerobot_record --robot.type=massage_robot_v2 --teleop.type=visual_hand --dataset.single_task="..."` 启动。适合后续接入 HIL 人在回路、在线干预。**注意**：此路径下 action 直接来自 teleop，若 arm 用位置目标需要 PC 侧 IK（见 §6.2 备选）。

**建议**：先跑通路径 1（最快出真机数据），路径 2 作为框架级接入在数据管线稳定后补，两者共用 `MassageRobotV2` 的通信层。

---

## 5. 数据采集方案

### 5.1 数据格式（LeRobotDataset v3.0）
用 `lerobot.datasets.LeRobotDataset` 直接写入（`data/chunk-000/episode_*.parquet` + `videos/` + `meta/{info,stats,episodes}.json` + `action/observation` 列）。时间戳对齐、stats、视频编码（libsvtav1 / h264）由框架处理。fps 目标 30（臂 `get_state` 10-20ms，30fps 可行；手 + 相机无瓶颈）。

### 5.2 特征与动作空间
| 特征 | 维度 | 来源 |
|------|------|------|
| `observation.state` | 22 | 臂 6（°）+ 手 16（LEAP rad）|
| `observation.images.*` | 640×480×3 ×N | 场景相机（+可选腕部相机）|
| `action` | 22 | 延迟状态（臂 ° + 手 rad），即实际执行轨迹 |

> 臂用角度（°）、手用 rad：**记录时保持各关节原始单位**，训练前在数据集侧统一归一化（LeRobot 自动算 stats）。若后续发现混合单位影响训练，可全转 rad。

### 5.3 采集流程（真机）
```
1. 摆好相机（场景相机看按摩对象 + 遥操 D455 看手，见操作手册 §一）
2. 启动 STM32 固件（ZDT 版）→ remote_enable
3. 启动 LEAP Hand（set_torque on，全开位）
4. 手眼标定 K（复用 demo_arm_teleop 的 K 向导）
5. 采集器（§4.2 路径1）：SPACE 开始 episode
6. 人手在 D455 前做按摩动作（臂跟随 + 手跟随），全程画面内
7. SPACE 结束 episode → 框架自动落盘/编码
8. 每姿势/穴位录 ≥10 条；场景相机固定、动作一致
```

### 5.4 采集质量要求（LeRobot 社区经验）
- 按摩对象必须在场景相机画面内，且"只凭看图像能判断穴位位置"（数据合格经验法则）。
- 每条 episode 时长 30-60s（按摩任务），`episode_time_s` 调大；开启 `streaming_encoding` 避免回合间阻塞。
- 背景尽量干净；避免拍到操作者/其他移动物。
- 先固定姿势录一批，稳定后再加变化（穴位偏移、力度变化）。

---

## 6. 分阶段实施计划

### Phase A — 仿真验证数据管线（无新硬件风险）
| # | 任务 | 产出/验收 |
|---|------|----------|
| A1 | `MassageRobotV2`（臂 6DOF，经 `socket://localhost:5555` 仿真 + 手 16DOF 占位/真手） | `get_observation`/`send_action` 在 LeRobot 0.4.4 下可调用 |
| A2 | 自定义采集循环（§4.2 路径1）接到仿真臂 | 仿真视觉遥操 → LeRobotDataset 落盘，episode 可回放 |
| A3 | 手部并入：真手或 16DOF 占位向量 | observation.state=22 维，action=22 维 |
| A4 | `lerobot-replay` 验证数据一致性 | 视频时长 = episode 时长，关节轨迹连续 |
| A5 | （可选）`VisualHandTeleoperator` + `lerobot_record` 跑通 | 标准流程可用 |

> 仿真不含手模型，A3 手部用真手（真机）+ 仿真臂组合，或先占位记录手目标。ADR-003 决定手走实物，故真手优先。

### Phase B — 真机通信打通
| # | 任务 | 产出/验收 |
|---|------|----------|
| B1 | **STM32 固件 ZDT 化**（§3.1）：Emm_V5 → ZDT 协议；保留 PC 命令集 | 串口 `get_state` 返回 ZDT 读数；`set_joints` 到位；`remote_event` 方向正确；`e_stop` 停 |
| B2 | 固件补 `end_event` + `get_ee_pose`（移植仿真 6DOF IK/FK） | 真机视觉遥操 6DOF 位姿跟随可用 |
| B3 | LEAP Hand 上电 + 现有测试脚本 | 16 电机全部使能，`poses.json` 姿势可复现 |
| B4 | `MassageRobotV2` 真机连接（STM32 串口 + 手 USB + 相机） | `get_observation` 返回真实 22 维状态 + 图像 |
| B5 | 视觉遥操真机调参（k_pos/k_ang/死区，沿用操作手册 §四） | 臂+手跟手、松手即停、急停有效 |

### Phase C — 真机数据采集
| # | 任务 | 产出/验收 |
|---|------|----------|
| C1 | 每姿势/穴位采集 ≥10 episodes（§5.3 流程） | `datasets/massage_v1` 就绪，LeRobot 格式 |
| C2 | `lerobot-info`/`lerobot-replay` 质检 | 无丢帧、关节/图像对齐 |
| C3 | 小规模训练验证（SmolVLA，~1000 steps） | loss 下降，推理能驱动臂+手 |

### Phase D — 训练与部署（后续）
| # | 任务 | 说明 |
|---|------|------|
| D1 | 全量训练 `lerobot_train --policy.type=smolvla` | 参考 `configs/train_smolvla.yaml` |
| D2 | 推理 `lerobot_eval` + `MassageRobotV2` 真机执行 | action 22 DOF → send_action |
| D3 | 可选 HIL 人在回路收集失败修正（RaC） | 按摩长任务分布偏移风险高，`VisualHandTeleoperator` 派上用场 |

---

## 7. 里程碑与验收

- **M1（Phase A 完）**：仿真视觉遥操 → LeRobotDataset，`lerobot-replay` 可回放，observation/action 各 22 维。
- **M2（Phase B 完）**：真机 STM32(ZDT)+LEAP Hand 通过 `MassageRobotV2` 连通，视觉遥操真机跟手、急停有效。
- **M3（Phase C 完）**：真机按摩数据 ≥30 episodes 入库，SmolVLA 小规模训练 loss 下降。
- **最终**：`lerobot_eval` 能按采集时的动作规范执行按摩并触碰目标。

---

## 8. 风险与开放问题

| # | 风险/问题 | 处置 |
|---|----------|------|
| 1 | **STM32 固件 ZDT 化工作量大**，且真机 J4 曾有相线问题 | B1 独立工作流，先在仿真（mujoco 已对齐 STM32 语义）验证 ZDT 帧；真机分轴逐个验证 |
| 2 | ZDT 协议字段放大倍数不一致（位置×10/误差×100） | 固件统一封装换算，PC 侧只见角度（°） |
| 3 | 真机无 `get_ee_pose`（当前全 0 降级）→ 视觉遥操姿态环失效 | 必须在 B2 补固件 FK，否则姿态跟手不可用（当前 demo 已依赖仿真反馈） |
| 4 | `serial_protocol.set_joints` 的 `read_until_ok(timeout=2s)` 可能阻塞 30fps | 实测回复延迟；慢则 fire-and-forget + 提高波特率 |
| 5 | 手/臂单位混用（° vs rad）影响训练 | 记录原始单位，数据集归一化；必要时统一 |
| 6 | 视觉遥操 + `lerobot_record` 双驱动冲突 | 主推自定义循环（路径1）；Teleoperator 路径只做标准接入不双驱动 |
| 7 | LEAP Hand 读取坏数据（超时回退全零）驱动乱动 | 复用 `_is_valid_pose` 安全门；采集循环失败时保持上一帧并告警 |
| 8 | 相机识别问题（同型号同 HUB） | 场景相机与遥操相机分接口，`lerobot-find-cameras` 校验 |
| 9 | ZDT 与 LEAP 两条总线（STM32 串口 / Dynamixel 4M）波特率协议不同 | 各自独立 USB/RS485 口，`MassageRobotV2` 内分 bus 管理 |

---

## 9. 参考

- LeRobot 0.4.4：`smolvla` env（`lerobot/robots`、`lerobot/teleoperators`、`lerobot/scripts/lerobot_record.py`）
- LeRobot BYOH 知识库：`Liang/Documents/Bright的知识库/LeRobot/自带硬件 · Hugging Face...md`、`真实世界机器人上的模仿学习.md`、`流式视频编码指南.md`
- 机械臂固件/协议：`Arm-robot_VLA/docs/SERIAL_COMMANDS.md`、`lerobot_robot_massage/{serial_protocol,massage_robot}.py`、`firmware/src/robot_cmd.c`
- ZDT 说明书：`Liang/Documents/Bright的知识库/基于VLA的机械臂设计/参考资料/ZDT_X系列_V2步进闭环驱动说明书Rev1.0.md`
- 视觉遥操：`Leap_Hand/python/gesture_mapping/{wrist_tracker,demo_arm_teleop,arm_client}.py`
- 灵巧手：`Leap_Hand/python/leap_hand_utils/dynamixel_client.py`
- 现有数据转换：`Arm-robot_VLA/scripts/convert_to_lerobot.py`（备选回退路径）
