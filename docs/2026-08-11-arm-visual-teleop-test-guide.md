# 机械臂视觉遥操 — 测试与使用教程

> 2026-08-11 | 位置跟随控制 · D455 深度 · 仿真先行
> 关联：[设计文档](design/2026-08-11-arm-visual-teleop-design.md) · [实现计划](plans/2026-08-11-arm-visual-teleop.md)

---

## 1. 系统概览

人手 → 机械臂末端的**视觉遥操**管线，控制范式为**位置跟随**（参考 teleop_gesture_toolbox 的 TeleoperationByDrawing）：

```
┌─ Leap_Hand PC (conda leap_hand) ─────────────────────────────┐
│ D455(color+depth) → MediaPipe 21点 → wrist 2D 像素            │
│   → rs.align 深度反投影 → wrist 3D(相机系, mm)                │
│   → 深度时域中值(滚动5) → 手眼旋转R → 基座系                  │
│   → delta = wrist − 参考(按住H时捕获)                         │
│   → target_ee = anchor_ee + delta·scale_pos (目标末端位置)    │
│   → 位置环 v = k·(target − ee_current) + 死区 (P控制)         │
│   → remote_event @30Hz                                        │
└───────────────┬──────────────────────────────────────────────┘
                │ serial (真机) / socket://localhost:5555 (仿真)
┌───────────────▼──────────────────────────────────────────────┐
│ Arm-robot_VLA (conda smolvla)                                 │
│  真机: STM32 robot_pid_remote (速度积分+IK+PID) / J5/J6直驱    │
│  仿真: mujoco_sim.py --ik (阻尼伪逆 Jacobian, get_ee 反馈)     │
└──────────────────────────────────────────────────────────────┘
```

**控制范式**：按住 H 时，手相对锚点的位移 → 末端**目标位置** → 位置环把臂"走到位停住"（手停臂停、手移臂移）；松开 H 重锚定（走哪停哪）。这与"手偏移→速度、回中才停"的速率控制本质不同，控制性更好。

**关节映射**（真机实测语义）：

| 关节 | 物理行为 | 控制信号 |
|------|---------|---------|
| J1-J4 | 位置链 | 位置环 vx/vy/vz → IK |
| J5 | 末端上下 | 手腕俯仰角 → 目标角 + 角度环 |
| J6 | 末端旋转 | 掌滚转角 → 目标角 + 角度环 |

---

## 2. 环境与硬件

| 项 | 要求 |
|----|------|
| conda env（Leap_Hand 侧） | `leap_hand`（D455/MediaPipe/手势管线） |
| conda env（机械臂侧） | `smolvla`（MuJoCo 仿真） |
| 硬件 | Intel RealSense D455 ×1（必须） |
| 分支 | 两仓库均 `feat/arm-visual-teleop` |

两个仓库路径：
- Leap_Hand：`/home/bright/win_office/ubantu_files/project/Leap_Hand`
- Arm-robot_VLA：`/home/bright/win_office/ubantu_files/project/Arm-robot_VLA`

---

## 3. 快速启动（仿真）

开**两个终端**：

### 终端 A — 仿真臂

```bash
conda activate smolvla
cd /home/bright/win_office/ubantu_files/project/Arm-robot_VLA
python scripts/mujoco_sim.py --ik --viewer
```

> `--viewer` 显示 3D 机械臂；Wayland 上崩溃则用 `--ik --no-camera` 无头模式（靠终端 B 的 HUD 和仿真每 2s 的 J1-J6 日志观察）。
> 仿真已对齐固件语义：`remote_event vx/vy/vz` 为基座系线速度（满速 150mm/s），p2/p3 直驱 J5/J6（满速 30°/s，KV 已提至 3.0）。

### 终端 B — 视觉遥操

```bash
conda activate leap_hand
cd /home/bright/win_office/ubantu_files/project/Leap_Hand/python
python gesture_mapping/demo_arm_teleop.py --port socket://localhost:5555
```

### 按键速查

| 按键 | 功能 |
|------|------|
| **H**（按一下切换） | 离合器：按住状态跟随，再次按下重锚定（走哪停哪） |
| **K** | 手眼标定向导（见 §4） |
| **SPACE** / **1-6** | 校准向导内：确认挥动 / 选基座方向码 |
| **Z** | 校准验证模式：翻转 Z 方向 |
| **C** | 重载 `handeye_calib.json` |
| **Y** | e_stop 急停 |
| **Q / ESC** | 退出（try/finally 安全回收） |

### HUD 行解读

```
CLUTCH:ON/OFF             离合器状态
v=(vx,vy,vz) J5=.. J6=..  当前命令（vx/vy/vz∈[-1,1], J5/J6∈[-1,1]）
J5tgt=.. J6tgt=..         位置环目标角（度）
depth=NNNmm roll=+Xdeg pitch=+Ydeg   wrist 深度(应随靠近/远离变化) + 手姿态诊断
tgt=(tx,ty,tz)mm err=(ex,ey,ez)mm    末端目标位置 + 位置误差（err→0 表示已到位）
```

---

## 4. 手眼标定（K 向导）

相机系(X右/Y下/Z前) 与机械臂基座系(X/Y水平/Z上) 三轴不同，必须标定旋转 R。**差分模式只需旋转**（平移在位移差中抵消）。

### 4.1 步骤

按 **K** 进入向导，依次 3 步：

1. 手沿**画面向右**移一段（按住 H）→ **SPACE** → 按 **1-6** 告诉"臂应去的基座方向"
2. 手沿**画面向上**移一段 → **SPACE** → 选方向
3. 手**向前（靠近相机）**移一段 → **SPACE** → 选方向

> **方向码**：`1=+X 2=-X 3=+Y 4=-Y 5=+Z(上) 6=-Z(下)`。选"臂实际应去的方向"，如手向上选 5。

求解后进入**验证模式**：
- 手向相机移动 → 看臂方向；反了按 **Z** 翻转 Z 映射（自动重存），对了按 **SPACE** 完成
- 完成自动写入 `python/gesture_mapping/handeye_calib.json` 并生效

### 4.2 标定文件

```json
{"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
```
`R` 为 3×3 旋转矩阵（相机系→基座系，列向量 `v_base = R @ v_cam`）。缺文件时用单位旋转（仅测试，方向必然错位）。

### 4.3 提示

- 标定过程臂仍受旧 R 驱动属正常；完成后需重新按 H 锚定才恢复遥操
- 若 X/Y 对但 Z 不对 → 用验证模式 Z 翻转，或重做第 3 步
- 若三个方向都乱 → 相机安装方式变了，重做 K 向导

---

## 5. 测试流程（M2 验证清单）

> 前置：完成 §4 标定。每项按"操作 → 预期 → 判定"执行。

### 5.1 位置跟随（X/Y/Z）

| # | 操作 | 预期 |
|---|------|------|
| 1 | 按住 H，手右移 50mm **停住** | 臂末端右移对应距离后**停住**，HUD `err=(ex,0,0)` 收敛到 ~0 |
| 2 | 手微调 ±10mm | 臂微动跟随，不抖动（死区 8mm 内不触发） |
| 3 | 松开 H | 臂停住当前位（下次按 H 从当前位置继续） |
| 4 | 手快速移动 | 臂以 150mm/s 追目标（误差大跑快、误差小减速） |
| 5 | 手**靠近/远离**相机 | 臂沿对应基座轴走到位（HUD `depth=` 应变小/变大） |

### 5.2 姿态跟随（J5/J6）

| # | 操作 | 预期 |
|---|------|------|
| 6 | 手腕**上翘/下压** | J5（末端上下）目标角跟随，臂转动 |
| 7 | 手腕**翻转**（旋前/旋后） | J6（末端旋转）目标角跟随，臂转动 |
| 8 | 姿态回正 | J5tgt/J6tgt 回到锚定值附近 |

### 5.3 安全

| # | 操作 | 预期 |
|---|------|------|
| 9 | 手移出视野 | 速度清零，臂保持 |
| 10 | 无手超时（>0.3s） | 仿真/固件归零 |
| 11 | 按 Y | e_stop 生效 |
| 12 | 按 Q/ESC | 安全退出（finally 回收 remote_disable/串口） |

### 5.4 判定标准

- 位置跟随：手到哪臂到哪，**手停臂停**，`err` 收敛到死区内
- 姿态：J5/J6 目标角随手腕俯仰/滚转变化，臂能走到目标角
- 全程无电机冲击、无方向错乱

---

## 6. 参数调优

`demo_arm_teleop.py` 中构造 `WristTracker(...)` 的参数：

| 参数 | 默认 | 含义 | 调法 |
|------|------|------|------|
| `scale_pos` | 1.0 | 手位移(mm)→末端目标(mm) 比例 | 手大范围→臂小范围：调小到 0.5；反之调大 |
| `k_pos` | 0.01 | 位置环增益 (1/mm)，100mm 误差≈0.9 满速 | 臂太慢调大(0.02)，太冲调小(0.005) |
| `deadzone_pos_mm` | 8.0 | 位置死区(mm)，防末端抖动 | 抖动调大(12)，不敏感调小(5) |
| `scale_ang` | 1.0 | 手俯仰/滚转(deg)→J5/J6目标(deg) | 按手感 |
| `k_ang` | 0.02 | 角度环增益 (1/deg) | 同 k_pos |
| `deadzone_ang_deg` | 3.0 | 角度死区(deg) | 同 deadzone_pos |
| `j5_range` | (0,90) | J5 目标钳制范围(度) | 勿超机械限位 |
| `j6_range` | (0,360) | J6 目标钳制范围(度) | 勿超机械限位 |
| `min_cutoff` | 2.0 | OneEuro 位置滤波截止(Hz) | 越大越跟手、越小越稳 |
| `beta` | 0.02 | OneEuro 速度系数 | 同上 |

**仿真 vs 真机差异**：仿真满速 150mm/s（`REMOTE_LIN_GAIN`），真机固件满速 20mm/s。同样 `k_pos` 下真机臂慢 7.5 倍——**真机需按手感重调 `k_pos`/`scale_pos`**。真机无 `get_ee` 命令时位置环自动降级为差分模式（用锚点作反馈），需固件支持或 PC 端 FK 后才闭环（见 §8）。

---

## 7. 故障排查

| 现象 | 可能原因 | 处置 |
|------|---------|------|
| 三轴全乱、臂乱跑 | 标定 R 是单位旋转（未做 K 向导） | 按 §4 做标定 |
| X/Y 对、Z 不对 | Z 映射反/被污染 | 重做 K 向导第 3 步，或用验证模式 Z 翻转 |
| 臂完全不动 | 未按 H / 手不在视野 / depth 无效 | 看 HUD `CLUTCH`/`depth`；手移到画面内 |
| 靠近/远离不跟 | wrist 深度噪声 / 深度流问题 | 看 HUD `depth=` 是否随距离变化；若跳变，深度滤波已内置(时域中值)仍不行则报告 |
| J6 翻转不跟 | J6 速度慢 / 掌法线噪声 | 确认 HUD `roll=` 随翻转变化；仿真已提速(KV 3.0) |
| 臂抖动 | 位置环死区太小 / 滤波太轻 | 调大 `deadzone_pos_mm` / 调小 `min_cutoff` |
| 手停臂不停(漂移) | 位置环收敛慢 / err 未到死区 | 调大 `k_pos` / 调小 `deadzone_pos_mm` |
| 方向整体相反 | R 方向码选反 | 重做 K 向导对应步骤 |
| `depth=0` 或不变 | D455 深度无效（太近<0.28m / 边缘） | 手移到相机 0.3-1.5m 工作区中部 |

### 诊断小技巧

- HUD `depth=`：**wrist 深度**，靠近/远离时应变小/变大——这是"深度流是否正常"的第一信号
- HUD `roll=/pitch=`：手腕翻转/俯仰时是否变化——判定姿态测量是否正常
- HUD `tgt=/err=`：`err` 是位置环误差，**err 大臂快跑、err≈0 臂停**——理解臂为什么在动

---

## 8. 架构与文件

### 8.1 Leap_Hand 侧模块

| 文件 | 职责 |
|------|------|
| `python/gesture_mapping/camera.py` | RealSense D455 color+depth 对齐 + 内参 |
| `python/gesture_mapping/handeye_calib.py` | 手眼 R：欧拉角/Procrustes/solve_handeye/存取 |
| `python/gesture_mapping/wrist_tracker.py` | **核心**：反投影、掌参考系、角度、位置环 `WristTracker` |
| `python/gesture_mapping/arm_client.py` | 机械臂串口薄客户端（remote_event/get_state/get_ee/e_stop） |
| `python/gesture_mapping/demo_arm_teleop.py` | 主循环 + 校准向导 + HUD |
| `python/tests/test_wrist_tracker.py` 等 | 单测（位置环/校准/客户端） |

### 8.2 Arm-robot_VLA 侧模块

| 文件 | 职责 |
|------|------|
| `scripts/remote_semantics.py` | `remote_event` 固件语义纯函数（与 robot_cmd.c 逐字节一致） |
| `scripts/mujoco_sim.py` | MuJoCo 仿真：位置 IK + J5/J6 直驱，get_ee 反馈 |
| `scripts/verify_remote_semantics.py` | 逐轴验证 vx/vy/vz/J5/J6 方向 |
| `firmware/src/robot_cmd.c` | 真机固件 `remote_event` 解析（语义基准） |

### 8.3 关键数据流

```
HandResult.landmarks(归一化) → build_palm_pts → (21,3)相机系mm
  → WristTracker.update(pts, ee_mm, j5_deg, j6_deg)
      depth时域中值 → R·wrist → delta → target = anchor + delta·scale
      位置环 v = k·(target−ee) → (vx,vy,vz)
      姿态目标角 + 角度环 → (j5_cmd, j6_cmd)
  → ArmClient.remote_event(vx,vy,vz,j5,j6)
      → p0=-vx p1=vy p2=j6 p3=-j5 p4=vz p5=-vz
  → 仿真/固件 → 电机
```

### 8.4 测试运行

```bash
cd /home/bright/win_office/ubantu_files/project/Leap_Hand/python
conda activate leap_hand
python -m pytest tests/ -q          # 全量（60 passed 2 skipped）
# 单测：位置环/校准/客户端
python -m pytest tests/test_wrist_tracker.py tests/test_arm_client.py tests/test_handeye_calib.py -v
```

---

## 9. 真机部署（M3 预告）

仿真验证通过后接真机 STM32：

1. **端口**：`--port /dev/ttyUSB0`（真机 USB 串口）
2. **get_ee 差异**：真机固件无 `get_ee` 命令 → 位置环自动降级为差分模式（用锚点作反馈）。要闭环需：
   - 方案 A：固件新增 `get_ee`（返回末端坐标，需固件内有 FK）
   - 方案 B：PC 端 FK（DH 参数已知，从 `get_state` 关节角算末端）
3. **满速差异**：真机 20mm/s vs 仿真 150mm/s → 按 §6 重调 `k_pos`/`scale_pos`
4. **安全**：J5/J6 目标已钳制在机械限位内；真机务必先低速试、`Y` 急停、e_stop 实测
5. **J5/J6 手感**：真机降级模式增益(0.01/死区8)与旧参数(0.008/15)不同，需重新确认

---

## 10. 参考

- 设计：`docs/design/2026-08-11-arm-visual-teleop-design.md`
- 计划：`docs/plans/2026-08-11-arm-visual-teleop.md`
- 参考项目：`GitHub_examples/mediapipe_dual_arm_control-main`、`teleop_gesture_toolbox-main`、`Dummy-Robot-main`
- 固件 `remote_event`：Arm-robot_VLA `firmware/src/robot_cmd.c`
- 仿真 `remote_event` 语义：Arm-robot_VLA `scripts/remote_semantics.py`
