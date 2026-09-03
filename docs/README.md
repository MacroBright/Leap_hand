# 📚 LEAP Hand 技术文档中心

欢迎查阅 `Leap_Hand` 16-DOF 灵巧手感知、控制与视觉遥操子系统的完整技术文档。

---

## 🧭 文档快速导航 (Documentation Sitemap)

### 1. 🚀 使用指南 (Guides)
- 🔌 **[硬件配置与接线指南](guides/hardware_setup.md)**：16 个 Dynamixel 舵机 ID 分配、12V 电源规范、拇指反向接线说明与 Linux 串口权限配置。
- 👁️ **[视觉手势遥操完全指南](guides/teleop_guide.md)**：实时单目/深度手势识别、HUD 键盘在线调参、一欧元防抖滤波与真机跟手调优。
- 🎯 **[姿态标定与姿态数据库指南](guides/calibration_and_poses.md)**：全开位标定、自定义手势录制、编码器 $2\pi$ 跨圈自动解回绕与安全门机制。
- 🦾 **[机械臂协同遥操作参考 (归档)](guides/arm_visual_teleop_manual.md)**：机械臂与灵巧手协同遥操的操作范式与手眼变换参考。

### 2. 💻 API 与开发参考 (API Reference)
- 🐍 **[Python SDK 开发指南](api/python_sdk.md)**：`LeapHand` 控制器、`JointMapper` 关键点映射、`LEAPHandFK` 正向运动学与滤波器的 Python 代码调用详解。

### 3. 📐 架构设计与理论推导 (Design & ADRs)
- 🔍 **[视觉遥操跟手问题调研](design/2026-08-10-teleop-following-investigation.md)**：时延分析与跟手误差建模。
- 💡 **[跟手问题解决方案设计](design/2026-08-10-teleop-following-solution.md)**：防抖滤波与增益标定架构设计。
- 🛠️ **[跟手改进工程实现总结](design/2026-08-10-teleop-following-implementation.md)**：P0+P1 真机验证通过报告。
- 🖐️ **[HaMeR 3D 网格重建集成设计](design/2026-08-05-hamer-3d-integration-w1.md)**：基于 MANO 模型的真实三维手势回归。

### 4. 📋 计划与迭代日志 (Plans & Roadmaps)
- 📅 **[LeRobot 具身智能数据集采集集成计划](plans/2026-08-14-lerobot-integration-data-collection.md)**
- 📅 **[机械臂-灵巧手视觉遥操整合计划](plans/2026-08-11-arm-visual-teleop.md)**
- 📅 **[3D 视觉手势映射开发计划](plans/2026-08-05-hamer-3d-integration-w1.md)**

---

## 🛠️ 常用 CLI 快捷命令汇总

| 命令 | 用途 | 快速示例 |
| :--- | :--- | :--- |
| **`leap-teleop`** | 实时单目/深度手势遥操 | `leap-teleop --drive` |
| **`leap-teleop-3d`** | 3D MANO / 多源高精度遥操 | `leap-teleop-3d --drive` |
| **`leap-control`** | 交互式逐指/手势控制台 | `leap-control` |
| **`leap-calibrate`** | 零点校准与姿势录制 | `leap-calibrate --action 1` |
| **`leap-diagnose`** | 16 舵机物理限位扫描体检 | `leap-diagnose` |
| **`leap-latency`** | 视觉遥操跟手时延测量 | `leap-latency` |
