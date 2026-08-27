# 🖐️ Leap_Hand — 16-DOF 灵巧手中医推拿感知与控制子系统

> `Leap_Hand` 是 TuinaDex 项目的灵巧手感知与执行子系统，基于 [LEAP Hand](https://github.com/leap-hand/LEAP_Hand_API)（RSS 2023）开源架构深度定制。子系统集成了 **Dynamixel 4Mbps USB 直连驱动**、**16-DOF 灵巧手控制 Hub**、**基于 RealSense D455 的 MediaPipe 21 点 / HaMeR 3D 视觉手势映射** 以及**逐指非线性增益标定与防抖滤波算法**。

---

## 1. 代码框架与目录结构详解

```text
Leap_Hand/
├── python/                                # ★ Python 主控制 Hub、算法库与视觉流水线
│   ├── main.py                            # LeapNode: 16-DOF 位置-电流混合 PID 控制主入口
│   ├── calibrate.py                       # 灵巧手零点校准与姿态预设录制 (生成 poses.json)
│   ├── interactive_control.py             # 逐指/预设手势交互式控制台
│   ├── poses.json                         # 标定姿态数据库 (张开/握拳/捏取等基准位姿)
│   │
│   ├── gesture_mapping/                   # ★ 视觉手势感知、3D 关键点重定向与映射核心
│   │   ├── camera.py                      # RealSense D455 深度图采集与内参矩阵 (K) 封装
│   │   ├── hand_tracker.py                # MediaPipe Hands 21 关节点检测与 World 3D 提取
│   │   ├── hamer_3d.py                    # HaMeR 3D MANO 网格重建与高精度 3D 关键点回归
│   │   ├── joint_mapper.py                # 人手几何特征 → LEAP Hand 16-DOF 目标弧度映射 (含拇指特殊接线补偿)
│   │   ├── wrist_tracker.py               # 掌骨刚体三角形 (Wrist-0/Index-5/Pinky-17) 姿态解耦
│   │   ├── calibrator.py                  # 人手五指全开自适应零点标定
│   │   ├── calibrate_gain.py              # 逐指关节非线性曲率增益拟合工具
│   │   ├── filter.py                      # 1€ (OneEuroFilter) 自适应时域防抖滤波
│   │   ├── leap_fk.py                     # LEAP Hand 16-DOF 正向运动学 (FK) 与指尖末端计算
│   │   ├── retarget_mapper.py             # 强化学习与模仿学习 Observation 重定向层
│   │   ├── retarget_obs.py                # 观察空间状态向量打包
│   │   ├── measure_following.py           # 视觉跟手端到端延迟与响应时间测量
│   │   ├── measure_motor_limits.py        # 16 舵机物理限位扫描与自检
│   │   ├── demo_realtime.py               # 实时单目手势映射可视化与真机驱动入口
│   │   ├── demo_hamer3d.py                # 多 3D 源 (HaMeR/World/Pseudo) 切换验证
│   │   └── models/                        # 神经网络预训练权重 (HaMeR / Checkpoints)
│   │
│   ├── leap_hand_utils/                   # ★ 底层通信与 Dynamixel 协议驱动封装
│   │   ├── dynamixel_client.py            # Protocol 2.0 4Mbps 批量读写 (SyncRead / SyncWrite)
│   │   └── leap_hand_utils.py             # 电机编码器值与弧度转换工具函数
│   │
│   └── tests/                             # 自动化单元测试套件 (54 项全部通过)
│       ├── test_calibrator.py             # 手势全开自适应标定测试
│       ├── test_joint_mapper.py           # 关节映射数学与拇指几何测试
│       ├── test_leap_fk.py                # 灵巧手正向运动学 (FK) 几何闭环测试
│       ├── test_wrist_tracker.py          # 掌骨刚体解耦与手腕速度滤波测试
│       ├── test_camera.py                 # RealSense 相机流与内参测试
│       ├── test_gain_fit.py               # 非线性增益曲线拟合测试
│       ├── test_retarget.py               # 观察状态重定向测试
│       └── test_hamer_3d.py               # HaMeR 3D 接口测试
│
├── ros_module/                            # ROS1 Noetic 驱动功能包
│   ├── CMakeLists.txt / package.xml       # ROS1 包描述与编译规则
│   ├── launch/ (leaphand.launch, ...)     # 启动节点与参数配置
│   ├── scripts/ (leaphand_node.py)        # ROS1 JointState 发布与控制节点
│   └── srv/ / msg/                        # 自定义 ROS1 服务与消息定义
│
├── ros2_module/                           # ROS2 Humble / Foxy 驱动功能包
│   ├── CMakeLists.txt / package.xml       # ROS2 ament_cmake 构建配置
│   ├── launch/ (leaphand_rviz.launch.py)  # RViz 仿真与真机控制启动文件
│   ├── scripts/ (leaphand_ros2.py)        # ROS2 Lifecycle 兼容控制节点
│   └── srv/                               # ROS2 姿态切换与参数设置服务
│
├── cpp/                                   # C++ 原生 SDK 封装与底层控制例程
│   ├── CMakeLists.txt                     # C++ 构建脚本
│   └── src/                               # Dynamixel C++ 驱动实现
│
├── CAD/                                   # 机械硬件模型与 3D 打印资产
│   └── (STL / STEP / URDF)                # 手指连杆、基座结构三维图纸
│
├── useful_tools/                          # 辅助开发工具集
│   └── mano_to_leap_mapping.py            # MANO 官方数据集到 LEAP 关节角转换脚本
│
└── docs/                                  # 设计文档、接线指南与开发手册
```

---

## 2. 子文件夹及子子文件夹功能详解

### 2.1 `python/` (Python 控制中心与视觉流水线)
- **`main.py` & `interactive_control.py`**：
  - 核心 Python 控制器，负责与 16 个 Dynamixel XC330-M288 舵机建立 4Mbps 高速串口通信，执行位置-电流模式控制（默认电流限制 150~350mA，防止揉捏堵转发热）；
  - 支持调用 `poses.json` 中的预设姿态（如五指伸直、握拳、二指捏合、三指拿捏等）。
- **`gesture_mapping/` (视觉手势映射算法库)**：
  - **`camera.py` & `hand_tracker.py`**：从 RealSense D455 获取对齐的 RGB 与深度帧，使用 MediaPipe 检测 21 个人手三维关节点；
  - **`joint_mapper.py`**：利用几何向量夹角实时解算人手各指节弯曲度与侧摆度，并转换为 LEAP Hand 对应的 16 舵机目标弧度（针对拇指特殊的 ID12=弯曲、ID13=侧摆接线进行了专属反向补偿）；
  - **`wrist_tracker.py`**：提取手腕点与两对掌指关节构成的刚体手掌坐标系，实现人手抓捏动作与宏观手掌姿态的彻底解耦；
  - **`filter.py`**：采用一欧元滤波器（$1€$ Filter），在手部静止时消除微颤，快速挥动时实现零相位滞后；
  - **`leap_fk.py`**：实现手指 DH 参数正向运动学，支持计算每个指尖的三维空间坐标与抓取包络。
- **`leap_hand_utils/` (底层串口通信引擎)**：
  - 封装 Dynamixel Protocol 2.0，使用 `GroupSyncRead` 与 `GroupSyncWrite` 在单个 USB 事务中同时收发 16 个电机的角度、速度与电流。

### 2.2 `ros_module/` 与 `ros2_module/` (ROS 生态支持)
- 提供标准的 ROS1 与 ROS2 节点，订阅 `/leaphand/cmd_joint_angles` 主题控制关节，并以高频率发布 `/joint_states` 供 MoveIt 或 RViz 进行实时可视化监测。

---

## 3. 硬件规格与接线指南

- **执行器**：16 × Dynamixel XC330-M288 智能总线舵机；
- **通信协议**：Dynamixel Protocol 2.0，波特率 **4,000,000 bps (4 Mbps)**；
- **供电电压**：12V DC（建议 12V 5A 稳压电源）；
- **主机连接**：USB 转 TTL 串口转换板（`/dev/ttyUSB0`）。

> [!IMPORTANT]
> **拇指特殊线序说明**：
> 四指（食指/中指/无名指/小指）采用 `ID x = MCP 侧摆, ID x+1 = MCP 弯曲`；
> 拇指采用反向线序：`ID 12 = MCP 弯曲, ID 13 = MCP 侧摆`。`joint_mapper.py` 已在软件层完成自动解耦。

---

## 4. 快速上手指引 (Quick Start)

### 4.1 配置串口权限
```bash
sudo chmod 666 /dev/ttyUSB0
```

### 4.2 运行 Python 交互控制台
```bash
cd python
python interactive_control.py
```

### 4.3 运行单目实时视觉手势映射 (MediaPipe 3D)
```bash
cd python
python gesture_mapping/demo_realtime.py --drive
```

### 4.4 运行 16 舵机物理限位与通信自检
```bash
cd python
python gesture_mapping/measure_motor_limits.py
```

### 4.5 运行全套自动化测试套件
```bash
pytest python/tests/
# 预期: 54 passed, 2 skipped (无 GPU 环境下跳过纯离线 HaMeR 测试)
```

---

## 5. 协同遥操说明
如需进行 **6-DOF 机械臂 + 16-DOF 灵巧手 22 自由度协同视觉遥操**，请使用顶层平级模块：
👉 [Co_Teleop 模块文档](../Co_Teleop/README.md) 或在根目录直接运行 `python run_teleop.py --iface can0 -y`。