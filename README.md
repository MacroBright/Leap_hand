<div align="center">

# 🖐️ LEAP Hand
### 16-DOF 灵巧手感知、控制与视觉遥操作子系统

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Tests](https://img.shields.io/badge/tests-59%20passed%2C%203%20skipped-brightgreen.svg?style=flat-square)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-src--layout-orange.svg?style=flat-square)](src/leap_hand/)
[![Baud Rate](https://img.shields.io/badge/Dynamixel-4Mbps%20Protocol%202.0-red.svg?style=flat-square)](https://emanual.robotis.com/)
[![Ecosystem](https://img.shields.io/badge/ROS2-Humble%20Ready-blueviolet.svg?style=flat-square)](ros2_module/)

<p align="center">
  <b>高精 16 自由度智能总线控制 · RealSense / WebCam 实时手势遥操 · 一欧元自适应防抖滤波 · HaMeR 3D 网格重建</b>
</p>

[✨ 功能亮点](#-功能亮点) •
[🏗️ 系统架构](#️-系统架构) •
[🔌 硬件规格与接线](#-硬件规格与接线) •
[⚡ 快速安装](#-快速安装) •
[🚀 常用-cli-命令](#-常用-cli-命令) •
[🐍-python-sdk-快速上手](#-python-sdk-快速上手) •
[📁-工程目录结构](#-工程目录结构) •
[📚-文档中心](#-文档中心)

</div>

---

## 📖 项目简介

`Leap_Hand` 是专为机器人中医推拿、灵巧抓取及具身智能训练打造的 **16 自由度（16-DOF）右手灵巧手感知、控制与遥操作子系统**。本项目基于 [LEAP Hand (Shaw et al., RSS 2023)](https://github.com/leap-hand/LEAP_Hand_API) 开源架构深度定制，重构为符合现代工业级规范的独立 Python 工程。

子系统集成了 **4Mbps 高速串口直连驱动**、**位置-电流限制安全保护**、**基于 RealSense D455 / WebCam 的 MediaPipe 21 点与 HaMeR 3D 视觉手势映射**、**一欧元自适应时域滤波** 以及 **开箱即用的命令行工具集（CLI）**。

---

## ✨ 功能亮点

- ⚡ **4Mbps 极速总线驱动**：基于 Dynamixel Protocol 2.0，利用 `GroupSyncRead` 与 `GroupSyncWrite` 在单个 USB 事务中同步读写 16 个舵机（通信时延 < 3ms）。
- 🛡️ **双重安全防爆机制**：运行于**位置-电流限制模式**（默认限制 150~350mA），内嵌实测 `motor_limits.json` 硬限位防过行程保护，配合 $2\pi$ 跨圈自动解回绕算法杜绝舵机暴转。
- 👁️ **低延迟视觉遥操作**：支持普通单目 WebCam 与 RealSense D455 深度流；通过向量夹角解算结合一欧元滤波器（$1€$ Filter），实现**静止时零抖动、快速移动时零相位滞后**。
- 🖐️ **3D MANO 网格重建支持**：集成 HaMeR 深度学习模型，支持从单目图像直接回归真实 3D 手部网格，解决手部旋转/握拳时的深度退化难题。
- 📦 **标准现代化工程架构**：采用标准的 `src/leap_hand` 包布局与 `pyproject.toml`，支持一键 `pip install -e .`，并注册 6 大独立终端命令。
- 🔄 **零破坏向后兼容**：原 `python/` 路径提供轻量兼容垫片，保证上层协同项目（如 `Co_Teleop` 与机械臂工程）无缝直接复用。

---

## 🏗️ 系统架构

```text
               ┌────────────────────────────────────────────────────────┐
               │         视觉感知层 (Vision & Gesture Perception)        │
               │  RealSense D455 / 单目 WebCam (RGB / 深度流)             │
               │    ├── MediaPipe Hands (21 关节点)                      │
               │    └── HaMeR 3D (MANO 真实三维网格回归)                  │
               └───────────────────────────┬────────────────────────────┘
                                           │ (3D 关节点 / 掌骨姿态)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         运动学与映射层 (Kinematics & Mapping)           │
               │    ├── calibrator.py   (人手平摊自适应零点标定)            │
               │    ├── joint_mapper.py (空间几何向量夹角解算)             │
               │    ├── filter.py       (1€ Filter 自适应时域防抖滤波)     │
               │    └── leap_fk.py      (16-DOF 正向运动学与指尖末端解算)  │
               └───────────────────────────┬────────────────────────────┘
                                           │ (平滑后的 16 舵机目标弧度)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         控制器与安全防爆层 (Controller & Safety)         │
               │    ├── LeapNode        (位置-电流混合 PID 调度 Hub)      │
               │    ├── pose_manager.py (姿态库加载与 2π 跨圈自动解回绕)   │
               │    └── motor_limits    (16 舵机机械物理限位强制裁剪)      │
               └───────────────────────────┬────────────────────────────┘
                                           │ (安全协议帧)
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │         硬件驱动层 (Hardware Driver)                    │
               │  DynamixelClient (Protocol 2.0 @ 4,000,000 bps)        │
               │    └── USB-TTL (/dev/ttyUSB0) ──▶ 16 × XC330-M288 舵机  │
               └────────────────────────────────────────────────────────┘
```

---

## 🔌 硬件规格与接线

| 项 | 规格说明 |
| :--- | :--- |
| **执行器** | 16 × Dynamixel XC330-M288 智能总线舵机 |
| **供电** | 12V DC 稳压电源（建议 12V 5A） |
| **通信接口** | USB-TTL 适配板（`/dev/ttyUSB0`），波特率 **4,000,000 bps (4 Mbps)** |
| **关节构型** | 食指 (0-3)、中指 (4-7)、无名指/小指 (8-11)、拇指 (12-15) |

> [!IMPORTANT]
> ### ⚠ 拇指线序特殊反向说明（硬件关键）
> 四指（食指/中指/无名指）采用：`ID x = MCP 侧摆, ID x+1 = MCP 弯曲`；  
> **拇指采用反向线序**：`ID 12 = MCP 弯曲, ID 13 = MCP 侧摆`！  
> 且拇指 ID 12 弯曲符号为正值弯向手心（其余指为负值）。本项目的 `JointMapper` 与控制器内部已完成自动解耦，软件调用时无需额外处理。

详细舵机 ID 表与接线请参阅 👉 **[硬件配置与接线指南](docs/guides/hardware_setup.md)**。

---

## ⚡ 快速安装

### 1. 配置 Conda 环境与依赖
推荐在独立环境 `leap_hand` 中运行：
```bash
# 激活环境
conda activate leap_hand

# 可编辑模式安装本包
pip install -e . --no-deps --no-build-isolation
```

### 2. 配置串口访问权限
```bash
sudo chmod 666 /dev/ttyUSB0
# 或添加用户组: sudo usermod -aG dialout $USER (注销重登生效)
```

---

## 🚀 常用 CLI 命令

重构后，系统注册了 6 个全局开箱即用的终端命令：

| 终端命令 | 核心功能 | 典型运行方式 |
| :--- | :--- | :--- |
| **`leap-teleop`** | ★ **单目/深度手势实时视觉遥操** | `leap-teleop --drive` |
| **`leap-teleop-3d`** | 3D MANO / 多源高精度手势遥操 | `leap-teleop-3d --drive` |
| **`leap-control`** | 逐指/逐关节/手势交互控制终端 | `leap-control --port /dev/ttyUSB0` |
| **`leap-calibrate`** | 硬件零点校准与推拿手法姿势录制 | `leap-calibrate --action 1` |
| **`leap-diagnose`** | 16 舵机物理限位测量与标定设置 | `leap-diagnose` |
| **`leap-latency`** | 视觉遥操跟手端到端时延测量评估 | `leap-latency` |

### 键盘交互快捷键（`leap-teleop` 窗口内）
- `空格 (SPACE)`：★ **一键自适应零点标定 & 硬件延迟上电**（右手平展对准镜头按空格，完成标定并开始跟手）；
- `L`：★ **姿态锁定 / 解锁（Pose Lock）**（按 L 立即锁定灵巧手当前姿态并死锁保持，再次按 L 解锁恢复实时手势遥操）；
- `D`：切换诊断 HUD 覆盖层（查看 16 舵机目标转角与弯曲判定）；
- `Tab`：轮转切换选定舵机；
- `[` / `]`：实时微调选定舵机的跟踪增益（Gain $\pm 0.05$）；
- `S`：一键保存增益参数到 `configs/joint_gain.json`；
- `Q` / `ESC`：平滑复位全开位并安全退出。


---

## 🐍 Python SDK 快速上手

```python
from leap_hand import LeapHand, OPEN_POSE
import time

# 1. 连接机械手 (默认电流上限 150mA，自动防堵转)
hand = LeapHand(port="/dev/ttyUSB0", curr_lim=150)

# 2. 读取当前 16 关节位置 (rad)
print("当前关节位置:", hand.read_pos())

# 3. 执行预设姿势 (从 configs/poses.json 加载)
hand.set_pose("全握拳")
time.sleep(1.5)

# 4. 单独微调食指近端指间关节 (相对全开位弯曲 0.5 rad)
hand.set_joint(motor_id=2, relative_angle=0.5)
time.sleep(1.0)

# 5. 复位回标定的全开位并断开
hand.set_open()
hand.disconnect()
```

更多 Python 接口详情请参阅 👉 **[Python SDK API 参考](docs/api/python_sdk.md)**。

---

## 📁 工程目录结构

```text
Leap_Hand/
├── pyproject.toml                         # ★ PEP 517/621 标准打包与脚本入口配置
├── README.md                              # ★ GitHub 项目主文档
├── configs/                               # ★ 集中化标定与参数数据库
│   ├── poses.json                         # 姿态库 (全开/半握/握拳/揉法等)
│   ├── motor_limits.json                  # 16 舵机实测物理安全限位
│   └── joint_gain_3d.json                 # 3D 关节增益曲线配置
│
├── src/                                   # ★ 核心源码目录 (src-layout)
│   └── leap_hand/                         # 独立 Python 包
│       ├── __init__.py                    # 顶层干净导出
│       ├── py.typed                       # 类型提示支持
│       ├── driver/                        # [底层通信驱动] (DynamixelClient)
│       ├── kinematics/                    # [纯运动学与滤波] (FK, limits, 1€ Filter)
│       ├── controller/                    # [高层控制与姿态] (LeapNode, pose_manager)
│       ├── vision/                        # [视觉手势感知] (MediaPipe, HaMeR, Mapper)
│       └── cli/                           # [命令行工具入口集] (teleop, calibrate 等)
│
├── tests/                                 # ★ 自动化单元测试套件 (62 项测试全部通过)
│   ├── conftest.py                        # pytest 环境配置
│   ├── test_standard_package.py           # 包架构与 API 规范回归测试
│   ├── test_calibrator.py                 # 自适应零点标定测试
│   ├── test_joint_mapper.py               # 关节几何映射测试
│   ├── test_leap_fk.py                    # 正运动学几何闭环测试
│   └── ...                                # 滤波、相机、手腕解耦等专项测试
│
├── python/                                # 兼容保留层 (保障外部旧路径调用不中断)
├── ros_module/                            # ROS1 Noetic 驱动包
├── ros2_module/                           # ROS2 Humble 驱动包
├── cpp/                                   # C++ 原生 SDK 与驱动例程
├── CAD/                                   # 机械硬件 3D 图纸资产 (STL/STEP/URDF)
└── docs/                                  # 设计文档、接线指南与开发手册
```

---

## 🧪 自动化测试验证

全套自动化测试套件位于 [tests/](tests/) 目录，覆盖了正向运动学、几何解算、拇指接线反转补偿、滤波平滑以及包结构导出规范。

```bash
# 运行全套测试套件
pytest tests/

# 预期输出: 59 passed, 3 skipped in 1.15s (跳过的 3 项为无 GPU/无相机的纯离线测试)
```

---

## 📚 文档中心

详细的技术手册与开发参考请查阅 **[docs/README.md](docs/README.md)**：
- 🔌 **[硬件配置与接线指南](docs/guides/hardware_setup.md)**
- 👁️ **[视觉手势遥操完全指南](docs/guides/teleop_guide.md)**
- 🎯 **[姿态标定与姿态数据库指南](docs/guides/calibration_and_poses.md)**
- 🐍 **[Python SDK API 参考](docs/api/python_sdk.md)**
- 🔍 **[遥操跟手问题调研与设计报告](docs/design/2026-08-10-teleop-following-investigation.md)**

---

## 🤝 引用与致谢

LEAP Hand 硬件设计与初始控制架构源自卡内基梅隆大学（CMU）的开源成果：

```bibtex
@inproceedings{shaw2023leaphand,
  title     = {LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning},
  author    = {Shaw, Kenneth and Agarwal, Ananye and Pathak, Deepak},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2023}
}
```

---

<div align="center">
  <sub>Developed with ❤️ by ForgeMind Robotics Team</sub>
</div>