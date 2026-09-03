<div align="center">

# LEAP Hand
### 16-DOF 灵巧手感知、控制与视觉遥操作子系统

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Tests](https://img.shields.io/badge/Tests-59%20Passed-brightgreen.svg)](tests/)
[![Dynamixel](https://img.shields.io/badge/Dynamixel-4Mbps%20Protocol%202.0-red.svg)](https://emanual.robotis.com/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble%20Ready-blueviolet.svg)](ros2_module/)

<p align="center">
  <b>Leap_Hand</b> 是 TuinaDex 系统的 16 自由度右手灵巧手控制子模块，负责 Dynamixel 舵机总线驱动、视觉手势追踪解算、自适应零点标定、时域防抖滤波，以及键盘与手势实时遥操作。
</p>

[主要特性](#主要特性) •
[系统架构](#系统架构) •
[硬件规格与接线](#硬件规格与接线) •
[环境安装](#环境安装) •
[命令行工具](#命令行工具-cli) •
[常用运行示例](#常用运行示例) •
[Python API](#python-api-示例) •
[测试验证](#测试验证) •
[相关文档](#相关文档)

</div>

---

## 项目简介

本项目基于 [LEAP Hand (Shaw et al., RSS 2023)](https://github.com/leap-hand/LEAP_Hand_API) 开源架构进行重构，针对中医推拿、灵巧抓取与具身示教场景进行模块化解耦，封装为标准的独立 Python 包。

系统集成了 4Mbps 串口同步驱动、位置-电流限制模式、基于 RealSense D455 / WebCam 的 MediaPipe 21 点视觉手势映射、一欧元自适应时域滤波，以及开箱即用的命令行工具。

---

## 主要特性

- **4Mbps 串口同步通信**：基于 Dynamixel Protocol 2.0，使用 `GroupSyncRead` 与 `GroupSyncWrite` 批量同步读写 16 个舵机，单次通信延迟低于 3ms。
- **电流限制与限位保护**：运行于位置-电流混合控制模式（默认限制电流 150~350mA），结合 `configs/motor_limits.json` 物理限位与 $2\pi$ 跨圈自动解回绕算法，防止舵机堵转过热与超程干涉。
- **手势追踪与时域滤波**：支持普通 WebCam 与 RealSense D455 深度流；通过空间向量夹角计算关节弯曲度，配合一欧元滤波器（1€ Filter）抑制静止抖动并保持动态响应。
- **姿态锁定与在线调谐**：支持一键平展手掌自适应校准零点，提供按键姿态死锁（Pose Lock）功能，支持在线微调 16 轴增益参数并持久化存储。
- **标准模块化工程**：采用标准 `src/leap_hand` 布局与 `pyproject.toml`，提供一键命令行工具与 Python SDK，同时保留 `python/` 垫片保障向后兼容。

---

## 系统架构

```text
               ┌────────────────────────────────────────────────────────┐
               │         视觉感知层 (Vision & Gesture Perception)        │
               │  RealSense D455 / 单目 WebCam (RGB / 深度流)             │
               │    ├── MediaPipe Hands (21 关节点)                      │
               │    └── HaMeR 3D (MANO 三维网格回归)                      │
               └───────────────────────────┬────────────────────────────┘
                                           │ (3D 关节点 / 掌骨姿态)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         运动学与映射层 (Kinematics & Mapping)           │
               │    ├── calibrator.py   (人手平展自适应零点标定)            │
               │    ├── joint_mapper.py (空间几何向量夹角解算)             │
               │    ├── filter.py       (1€ Filter 自适应时域防抖滤波)     │
               │    └── leap_fk.py      (16-DOF 正向运动学与指尖解算)      │
               └───────────────────────────┬────────────────────────────┘
                                           │ (平滑后的 16 舵机目标弧度)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         控制器与安全保护层 (Controller & Safety)         │
               │    ├── LeapNode        (位置-电流混合控制 Hub)            │
               │    ├── pose_manager.py (姿态库加载与 2π 跨圈自动解回绕)   │
               │    └── motor_limits    (16 舵机物理限位强制裁剪)          │
               └───────────────────────────┬────────────────────────────┘
                                           │ (协议帧)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         底层硬件驱动层 (Hardware Driver)                │
               │  DynamixelClient (Protocol 2.0 @ 4,000,000 bps)        │
               │    └── USB-TTL (/dev/ttyUSB0) ──▶ 16 × XC330-M288 舵机  │
               └────────────────────────────────────────────────────────┘
```

---

## 硬件规格与接线

| 项目 | 规格说明 |
| :--- | :--- |
| **执行器** | 16 × Dynamixel XC330-M288 智能总线舵机 |
| **供电** | 12V DC 稳压电源（推荐 12V 5A） |
| **通信接口** | USB-TTL 适配器（`/dev/ttyUSB0`），波特率 **4,000,000 bps (4 Mbps)** |
| **手指拓扑** | 食指 (ID 0-3)、中指 (ID 4-7)、无名指/小指 (ID 8-11)、拇指 (ID 12-15) |

> **拇指线序说明**：  
> 食指、中指、无名指采用 `ID x = MCP 侧摆, ID x+1 = MCP 弯曲`；  
> **拇指采用反向线序**：`ID 12 = MCP 弯曲, ID 13 = MCP 侧摆`，且 ID 12 弯曲角度为正值（其他手指弯曲为负值）。本项目的 `JointMapper` 内部已完成解耦，调用时无需额外换算。

---

## 环境安装

### 1. 创建独立环境并安装

推荐使用 Python 3.10 环境：

```bash
# 方式 A：Conda 一键创建与安装 (推荐)
cd Leap_Hand
conda env create -f environment.yml
conda activate leap_hand

# 方式 B：手动创建与 pip 安装
conda create -n leap_hand python=3.10 -y
conda activate leap_hand
cd Leap_Hand
pip install -r requirements.txt
pip install -e .
```

### 2. 配置串口访问权限

```bash
sudo chmod 666 /dev/ttyUSB0
# 或将当前用户加入串口组 (注销重登后生效):
sudo usermod -aG dialout $USER
```

---

## 命令行工具 (CLI)

安装后系统将注册以下全局命令行工具：

| 命令 | 功能说明 | 常用运行示例 |
| :--- | :--- | :--- |
| **`leap-teleop`** | **实时手势视觉遥操主程序** | `leap-teleop --drive`<br/>*(空跑测试: `leap-teleop`)* |
| **`leap-teleop-3d`** | 3D 手势遥操与网格追踪模式 | `leap-teleop-3d --drive` |
| **`leap-control`** | 逐指/逐关节控制与姿态测试终端 | `leap-control --port /dev/ttyUSB0` |
| **`leap-calibrate`** | 硬件零点校准与推拿手法姿态录制 | `leap-calibrate --action 1` |
| **`leap-diagnose`** | 16 舵机物理限位检测与状态诊断 | `leap-diagnose` |
| **`leap-latency`** | 端到端遥操跟手通信时延评估 | `leap-latency` |

### 键盘交互快捷键（`leap-teleop` 窗口内）
- **`空格 (SPACE)`**：一键自适应零点标定并使能上电（右手平展对准镜头按下空格，完成基准捕捉并开始跟手）；
- **`L`**：当前姿态锁定 / 解锁（Pose Lock，锁定后灵巧手保持当前抓握姿势不变，再次按 L 恢复跟随）；
- **`D`**：开启 / 关闭 HUD 状态覆盖层（显示 16 舵机目标转角与弯曲判定）；
- **`Tab`**：轮转选择特定舵机；
- **`[` / `]`**：微调选定舵机的跟踪增益（Gain $\pm 0.05$）；
- **`S`**：保存调整后的增益参数至 `configs/joint_gain.json`；
- **`Q` / `ESC`**：平稳复位回全开位并安全退出。

---

## 常用运行示例

### 1. 纯视觉手势空跑（无需连接真实灵巧手）
```bash
leap-teleop
```

### 2. 真手连接并开始实时视觉遥操
```bash
leap-teleop --drive --port /dev/ttyUSB0
```
1. 启动后右手平展面向相机；
2. 按键盘 **`空格键`**，系统自动捕获手掌参考系并启动舵机跟随；
3. 需要保持抓握姿势时按 **`L`** 键锁定姿态。

### 3. 命令行交互调试终端
```bash
leap-control --port /dev/ttyUSB0
```

---

## Python API 示例

```python
import time
from leap_hand import LeapHand

# 1. 连接机械手 (默认限制电流 150mA，防止堵转过载)
hand = LeapHand(port="/dev/ttyUSB0", curr_lim=150)

# 2. 读取当前 16 关节实际位置 (rad)
print("当前关节位置:", hand.read_pos())

# 3. 执行预设姿势 (从 configs/poses.json 读取)
hand.set_pose("全握拳")
time.sleep(1.5)

# 4. 单独控制指定舵机 (例如食指近端关节转动 0.5 rad)
hand.set_joint(motor_id=2, relative_angle=0.5)
time.sleep(1.0)

# 5. 复位回初始全开位并断开连接
hand.set_open()
hand.disconnect()
```

---

## 测试验证

项目包含 62 个自动化测试，覆盖正运动学、几何解算、拇指反向补偿、滤波算法与接口规范：

```bash
pytest tests/
```

```text
======================== 59 passed, 3 skipped in ~1.3s =========================
```
*(跳过的 3 项为需要实际物理相机连接的测试用例)*

---

## 相关文档

- [硬件配置与接线指南 (docs/guides/hardware_setup.md)](docs/guides/hardware_setup.md)
- [手势视觉遥操使用指南 (docs/guides/teleop_guide.md)](docs/guides/teleop_guide.md)
- [姿态标定与数据库说明 (docs/guides/calibration_and_poses.md)](docs/guides/calibration_and_poses.md)
- [Python SDK API 参考 (docs/api/python_sdk.md)](docs/api/python_sdk.md)
- [跟手时延优化报告 (docs/design/2026-08-10-teleop-following-investigation.md)](docs/design/2026-08-10-teleop-following-investigation.md)

---

## 引用与致谢

LEAP Hand 硬件设计源自卡内基梅隆大学（CMU）开源研究成果：

```bibtex
@inproceedings{shaw2023leaphand,
  title     = {LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning},
  author    = {Shaw, Kenneth and Agarwal, Ananye and Pathak, Deepak},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2023}
}
```

---

## 开源协议

本项目遵循 [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) 协议。