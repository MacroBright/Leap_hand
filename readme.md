# 🖐️ LEAP Hand — Dexterous Hand Control Hub & Visual Teleoperation

> 基于 [LEAP Hand](https://github.com/leap-hand/LEAP_Hand_API)（RSS 2023）的分支仓库，定位为**中医按摩灵巧手**：在保留原版 SDK 全部能力的基础上，新增 Python 灵巧手控制 Hub 与「手部关键点识别 → 灵巧手 / 机械臂」的视觉遥操作。

> 中文文档见 [CLAUDE.md](CLAUDE.md)（项目总纲）、[docs/](docs/)（设计 / 计划 / 手册）。

---

## 中文摘要

本项目在原 [LEAP Hand](https://github.com/leap-hand/LEAP_Hand_API) 开源 SDK 之上新增了两大功能：

| 功能 | 说明 |
|------|------|
| **灵巧手控制 Hub** | Python 控制中心（`main.py` / LeapNode）：PID 控制、姿势预设录制与回放、逐指 / 手势交互控制、16 电机直连。 |
| **手部关键点识别 → 视觉遥操** | 摄像头（RealSense D455）实时捕捉人手关键点（MediaPipe 21 点 / HaMeR 3D）→ 映射 LEAP 16-DOF 关节角驱动灵巧手；并可进一步把**人手 6DOF**（手腕位置 + 手掌姿态）映射为**机械臂末端位姿**，实现整臂视觉遥操作。 |

> 当前进度：灵巧手 16-DOF 手势映射已真机验证；机械臂视觉遥操基于仿真验证、差分速度 / 末端位姿跟随已实现（真机接线需 STM32 固件支持末端 FK 反馈，见 [demo_arm_teleop.py](python/gesture_mapping/demo_arm_teleop.py)）。LeRobot BYOH 数据采集集成处于规划阶段。

---

## 1. Overview

**LEAP Hand** is a low-cost, efficient, and anthropomorphic 16-DOF dexterous hand designed for robot learning (Shaw et al., RSS 2023). Its original SDK ships USB-direct Dynamixel control with Python / C++ / ROS / ROS2 APIs.

This repository is a **fork** built for a *Traditional Chinese Medicine (TCM) massage* use case. It keeps the upstream SDK intact (`cpp/`, `ros_module/`, `ros2_module/`, `useful_tools/`) and adds a project-specific layer on top:

```
┌──────────────────────── Python Control Hub (python/) ────────────────────────┐
│  main.py (LeapNode) · calibrate.py (姿势录制) · interactive_control.py        │
│      ↑ ID 0-15 位置-电流模式 PID 控制（Dynamixel SDK 4.0.5, USB 直连）            │
├────────────────────── Visual Teleoperation (python/gesture_mapping/) ────────┤
│  Camera(RealSense D455/MediaPipe) → 21 关键点                                   │
│    → JointMapper（人手 → LEAP 16DOF，伪 3D / HaMeR 3D） → 灵巧手驱动器          │
│    → WristTracker（腕 3D + 掌姿态 6DOF）→ 位置/姿态环 → 机械臂 end_event        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### What's new in this fork

| Area | Upstream (LEAP_Hand_API) | This fork |
|------|--------------------------|-----------|
| Hand control | Raw demos + examples | Pose preset DB (`poses.json`), calibration & recording, per-finger / gesture interactive control, PID tuning |
| Vision input | — | MediaPipe Hands (21 kp) + RealSense depth; HaMeR 3D as alternative source |
| Hand→LEAP mapping | MANO→LEAP tool only | Real-time human-hand→LEAP 16-DOF mapping (4 fingers + thumb, incl. thumb wiring fix) |
| Arm teleoperation | — | Human wrist 6-DOF → arm end-effector pose (position/orientation loop → `end_event` for full IK) |
| Robot learning | — | LeRobot BYOH integration planned (roadmap) |

---

## 2. Hardware

- **16 × Dynamixel XC330-M288** servos, Protocol 2.0, **4 Mbaud**, USB direct to PC (`/dev/ttyUSB0`), no STM32 in the loop.
- Position–current control mode; current limit 300 mA (Lite) / up to 550 mA (Full).
- Right hand only; motor IDs 0–15. See the joint mapping table in [CLAUDE.md](CLAUDE.md#五关节映射速查).

> ⚠️ Thumb wiring is opposite to the other fingers: ID 12 = MCP-flex, ID 13 = MCP-abduction (the four fingers use ID x = abduction, x+1 = flexion).

---

## 3. Quick Start

Environment: conda env `leap_hand` (Python 3.14, `opencv-contrib`, `mediapipe`, `pyrealsense2`, `dynamixel-sdk 4.0.5`, `sounddevice`/`pyserial`).

```bash
conda activate leap_hand
python python/main.py                          # Core control hub (requires hardware)
cd python && python calibrate.py               # Calibrate & record poses
cd python && python interactive_control.py     # Per-finger / gesture interaction
```

**Hand gesture mapping → LEAP hand:**

```bash
cd python && python gesture_mapping/demo_realtime.py [--drive]   # MediaPipe (pseudo-3D) real-time mapping
cd python && python gesture_mapping/demo_hamer3d.py [--drive]    # Swappable 3D source (hamer / world / pseudo-3D)
```

**Vision teleoperation of an arm** (simulation first — run the MuJoCo arm in another terminal):

```bash
cd python && python gesture_mapping/demo_arm_teleop.py --port socket://localhost:5555
```

See the [arm visual teleop operation manual](docs/2026-08-12-arm-visual-teleop-operation-manual.md) for camera placement, hand→arm motion table, and key bindings (H clutch, M record home pose, R reset, K hand-eye calibration wizard, Y e-stop).

---

## 4. Project Layout

```
Leap_Hand/
├── python/                      # Added: control hub + vision teleoperation
│   ├── main.py                  # LEAP 16-DOF PID control, pose presets, safety gate
│   ├── calibrate.py             # Pose recording → poses.json
│   ├── interactive_control.py   # Per-finger / gesture interactive control
│   ├── poses.json               # Calibrated pose DB (source of truth)
│   ├── leap_hand_utils/         # Dynamixel client wrapper (upstream-derived)
│   └── gesture_mapping/         # Added: media/gesture/telop pipeline
│       ├── hand_tracker.py      # MediaPipe 21-kp tracking
│       ├── hamer_3d.py          # HaMeR 3D hand mesh source
│       ├── joint_mapper.py      # Human hand → LEAP 16-DOF mapping
│       ├── wrist_tracker.py     # Wrist 3D + palm 6-DOF pose
│       ├── handeye_calib.py     # Hand–eye rotation calibration (Euler / Procrustes / wizard)
│       └── demo_*.py            # demo_realtime / demo_hamer3d / demo_arm_teleop
├── cpp/  ros_module/  ros2_module/  useful_tools/   # Upstream SDK (unchanged)
├── CAD/                                              # Upstream CAD files
├── docs/                        # Added: design / plans / operation manuals (zh)
└── CLAUDE.md                    # Project guideline (zh): ADRs, workstreams, conventions
```

---

## 5. Documentation

- [CLAUDE.md](CLAUDE.md) — project guideline in Chinese: tech stack, ADRs (001–005), workstream split, joint mapping.
- [docs/design/](docs/design/) — design docs: HaMeR 3D integration, teleop-following investigation & implementation, arm visual teleop design.
- [docs/plans/](docs/plans/) — implementation plans (incl. LeRobot integration & data collection).
- [docs/2026-08-12-arm-visual-teleop-operation-manual.md](docs/2026-08-12-arm-visual-teleop-operation-manual.md) — arm teleop operator manual.
- Original SDK usage: upstream [LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API) — Python / C++ / ROS / ROS2 setup links live in the upstream repo.

---

## 6. Upstream Usage Notes (unchanged, from LEAP Hand)

- Connect **5 V power** to the hand; connect the **Micro-USB** cable (avoid extensions).
- On Ubuntu, find the hand at `/dev/serial/by-id/*` (persistent). `sudo chmod 666 /dev/serial/by-id/(your_id)` for permissions.
- **Do not** keep Dynamixel Wizard open while using the API (the port will be busy).
- Latency tips: set *Return Delay Time* (register 9) to **0 µs**; lower P/D values for jitter, raise for a weak hand.
- Full hand: raise the current limit from 300 mA to 550 mA for extra strength.

### Troubleshooting (upstream)

| Symptom | Fix |
|---------|-----|
| Motor off by 90°/180°/270° | Remount the horn |
| No motors show up | Check serial permissions |
| Some motors missing | Verify IDs and U2D2 connections |
| Overload (motors flash red) | Power cycle; if frequent, lower current limits |
| Jittery / inaccurate motors | Lower / raise P/D values |

---

## 7. License & Citation

- **Code:** MIT License — see [LICENSE](LICENSE).
- **CAD:** CC BY-NC-SA (non-commercial use with attribution).
- Provided **as-is**, without warranty.

If you use LEAP Hand in research, please cite the original work:

```bibtex
@article{shaw2023leaphand,
  title={LEAP Hand: Low-Cost, Efficient, and Anthropomorphic Hand for Robot Learning},
  author={Shaw, Kenneth and Agarwal, Ananye and Pathak, Deepak},
  journal={Robotics: Science and Systems (RSS)},
  year={2023}
}
```

👉 **More info:** [LEAP Hand Website](http://leaphand.com/) · [Upstream repo](https://github.com/leap-hand/LEAP_Hand_API)