# 视觉遥操"不跟手"问题 — 调研与方案制定任务交接

> 交接对象: 负责视觉遥操优化的 agent（新会话，独立上下文）
> 项目: LEAP Hand 中医按摩灵巧手 — W1 手势映射（视觉遥操）
> 任务: 调研"灵巧手动作不跟手"的根因，制定解决方案设计（**本轮只调研 + 方案，不实现**）
> 交接日期: 2026-08-10
> 交接人: 主会话（已通读全部源码 + 调试文档 + git 历史）

---

## 0. 一句话任务

视觉遥操中灵巧手动作不跟手（人手动、机器手跟得慢 / 跟不足 / 跟过头 / 抖动）。系统性调研整条驱动链路，逐一定位根因，产出**优先级化的解决方案设计文档**。本轮不做代码改动（除非极小验证性改动）。

## 1. 已知症状谱系（"不跟手"的可能表现，需逐一判定）

| 症状 | 描述 | 已有线索 |
|------|------|---------|
| **振幅不足（半握）** | 人手全攥，机器手只半握 | 有专门调试文档 §3，嫌疑已排序 |
| **跟随滞后** | 人手动了机器手慢半拍 | 延迟预算待查（§5.5） |
| **抖动/跳变** | 静止时角度颤动、偶发跳变 | 已有大量稳定性工作（§4） |
| **方向错** | 某指越弯越伸 / 侧摆方向反 | JOINT_DIR 已实测修正 |
| **手指独立性差** | 弯一指带动其他指 | 扇角/关键点链待查 |
| **拇指对掌不足** | 攥拳时拇指横跨掌心不到位 | `_OPP_*` 补偿权重待查 |
| **过冲/回弹** | 到位后晃两下 | 滤波参数待查 |

## 2. 系统管线（驱动链路）

```
相机(D455/OpenCV) → MediaPipe 21kp
  → 3D 源(伪3D / world-3D / hamer MANO)
  → JointMapper 算 16 关节角(屈曲 arccos + 侧摆 atan2)
  → Calibrator 减全开基线
  → OneEuro 1€滤波
  → JOINT_DIR×增益 → OPEN_POSE + 绝对位姿
  → motor_limits 裁剪 → Dynamixel 写入
```

### 2.1 硬件与 16-DOF（必读）

- 16 电机 ID 0-15，Dynamixel XC330-M288，USB 直连 `/dev/ttyUSB0`，4M baud，位置-电流模式（`curr_lim=550mA`）
- 人手→机器手: 食指→0-3，中指→4-7，**小指→8-11（LEAP 无名指）**，拇指→12-15；**人手无名指舍弃**
- ⚠ **拇指物理接线与其他指相反: ID 12=mcp, 13=side**（其余四指 ID x=side, x+1=mcp）
- **JOINT_DIR**: mcp/pip/dip 为 -1（负值弯向手心），**拇指 mcp(ID12) 例外 +1**；四指 side 为 -1

### 2.2 映射算法（JointMapper，[joint_mapper.py](../../python/gesture_mapping/joint_mapper.py)）

- 屈曲角 = 相邻三点空间夹角（`_compute_flexion`, arccos 点积）
- 侧摆角 = 手指近端段相对"中指方向"的平面内带符号扇角（`_compute_fan_angle`, atan2）
- 拇指对掌补偿: `_OPP_MCP_WEIGHT=0.6` / `_OPP_PIP_WEIGHT=0.6` 叠加到拇指 mcp/pip
- 输出裁剪 `_ANGLE_MIN=-0.5` / `_ANGLE_MAX=2.8` → 乘每关节增益
- 默认增益 `_JOINT_GAIN`: 侧摆 0.4×（阻尼过冲），mcp/pip/dip 1.5×（补偿伪3D 压扁），拇指 mcp 1.5× / side 0.6×

### 2.3 校准（Calibrator，[calibrator.py](../../python/gesture_mapping/calibrator.py)）

- SPACE 记录全开手基线，之后输出 = `clip(当前角 − 基线, -0.3, 2.8)`
- **两套基线槽位分离**: 3D 路径 `_baseline_points`，伪3D 路径 `_baseline`，不可混用
- 基线持久化 `calibration_3d.json`（启动自动加载）

### 2.4 滤波（[filter.py](../../python/gesture_mapping/filter.py)）

- `angle_filter`: OneEuro 16 维（min_cutoff≈0.5-1.0, beta≈0.005-0.007）——**min_cutoff 越低越平滑但延迟越大**，是"跟手滞后"的嫌疑点
- 关键点级 smoother: `world_smoother`(63维) / `pseudo_smoother`(63维) / `frame_smoother`(9维)
- 帧级质量门控 `_MIN_VIS=0.55`（低可见度保持上一帧好角度）

### 2.5 电机限位裁剪

- `motor_limits.json`（实测 min/max 16 电机）写入前: `pose = clip(OPEN_POSE + JOINT_DIR*angles, min, max)`
- 若测量偏小（尤其 DIP 回绕关节）会**提前封顶屈曲** → 半握嫌疑之一

### 2.6 安全（必须遵守）

- 真机启动**不上电**，张开手按 SPACE（校准全开基线）→ **才连接上电**
- 涉及硬件操作前核对 `~/.coagent-knowledge/safety/` 与 hardware-safety-checklist 技能（CLAUDE.md §八）
- 改动 2+ 文件前按 CLAUDE.md §九 备份

## 3. 已有调试资产（必读）

**`docs/debug/2026-08-05-half-clench-handoff.md`** — "人手全攥→机器手半握"的完整调试交接，包含: 控制映射原理、5 步排查流程（Step2 先判"角度够不够"再查增益/限位/3D源/方向）、已知坑、完成标准。新会话先读它，再决定哪些已验证、哪些需重新验证。

半握根因嫌疑排序（该文档给出）:
1. 增益未放大（恒等 1.0× 未补偿 LEAP 行程）
2. 电机限位裁剪提前封顶
3. 校准基线不准
4. 3D 源攥拳时关键点退化
5. 方向/接线错

## 4. 当前状态速查（2026-08-10 已核对源码，勿轻信旧文档）

- **主 demo = [demo_hamer3d.py](../../python/gesture_mapping/demo_hamer3d.py)**，默认 3D 源 = **伪3D**（`source_mode=2`，代码注释"实机跟手最佳"；M 键循环 hamer → world-3D → 伪3D）
- ⚠ **增益分离的坑**: demo_hamer3d 里三种源**共用 `joint_gain_3d.json`**（不是 joint_mapper 默认 `_JOINT_GAIN`）。该文件已调大（ID3/ID13=3.0），但**伪3D 源没有吃到默认的 1.5× 放大**——伪3D 的"相机压扁"补偿是否充分，需重点核实
- `joint_gain_3d.json` 当前值: `[0.44, 1.84, 0.89, 3.0,  1.0, 1.47, 0.81, 2.92,  0.10, 1.88, 1.50, 2.88,  0.60, 3.0, 0.8, 2.8]`
- `motor_limits.json`（实测）、`calibration_3d.json`（持久化基线）均已存在
- 8 个姿势（7 手势 + 揉法）已录制于 `poses.json`
- 另一 demo `demo_realtime.py`（伪3D 路径）用 `_JOINT_GAIN` 默认 + 加载 `joint_gain.json`（当前不存在）——与 demo_hamer3d 增益来源不同
- **已完成稳定性工作**（git log，勿重复做）: 手丢失平滑回 OPEN、持久化校准基线、掌参考系时域平滑、bbox EMA、关键点 OneEuro、延迟上电、限位裁剪、MediaPipe world-3D 源
- hamer 3D 需独立 `hamer` env（py3.10 + GPU）；`leap_hand` env 无 torch/hamer

## 5. 调研方向（研究线索，需验证/扩展）

1. **增益策略（半握根因#1）**: 区分"角度没算够"vs"电机没走够"。核实 demo_hamer3d 在伪3D 源下是否充分放大屈曲；评估 per-3D-source 增益设计是否合理
2. **限位封顶（#2）**: DIP 回绕关节限位是否偏小；攥拳时 `OPEN+DIR*angles` 是否触 max
3. **3D 源攥拳退化（#4）**: 攥拳时伪3D / world-3D / hamer 三源角度对比（M 键切换 / `--img` 单图 / 离线 test_compare）
4. **重定向保真度**: 现为"直接角映射"（arccos），未做手指尺寸归一化/运动学重定向。人手段长比例 ≠ LEAP。参考官方 `docs/reference/dex-hand-teleop-main`、AnyDexRetarget 的 retargeting 方法（`docs/knowledge/discovery-candidates.md` #1/#2）
5. **延迟预算**: MediaPipe VIDEO 模式 + OneEuro min_cutoff + Dynamixel 写入频率，各自延迟贡献；"跟手滞后"是否可感知
6. **拇指对掌**: `_OPP_*` 权重是否够；攥拳时拇指 mcp/pip 屈曲实测值
7. **手指独立性**: 扇角 `_FAN_SIGN`、`_palm_frame` 参考系稳定性（已有平滑，但手快速转动时是否串扰）
8. **尺寸校准**: workstream 01 的 TODO"人手-灵巧手尺寸校准"
9. **角度定义边界**: `_ANGLE_MAX=2.8` 裁剪、`_compute_flexion` 对退化关键点的处理

## 6. 关键文件地图

| 文件 | 作用 |
|------|------|
| [joint_mapper.py](../../python/gesture_mapping/joint_mapper.py) | 16-DOF 角度计算核心（`_JOINT_GAIN`/`_ANGLE_*`/`_FINGER_MAP`/`map_points_to_leap`/`_compute_flexion`/`_compute_fan_angle`）|
| [calibrator.py](../../python/gesture_mapping/calibrator.py) | 全开基线（`calibrate_points`/`map_points`/`_baseline_points`）|
| [demo_hamer3d.py](../../python/gesture_mapping/demo_hamer3d.py) | 主 demo: JOINT_DIR / 驱动写入 / SPACE上电 / 限位裁剪 / 增益调参键 / 3D源切换 |
| [demo_realtime.py](../../python/gesture_mapping/demo_realtime.py) | 伪3D demo（旧）：`_MOTOR_DIAG`/`find_best_camera`/`draw_hud` 被 demo_hamer3d import |
| [hand_tracker.py](../../python/gesture_mapping/hand_tracker.py) | MediaPipe VIDEO 模式跟踪（Landmark 字典、world 关键点）|
| [hamer_3d.py](../../python/gesture_mapping/hamer_3d.py) | hamer MANO 3D（GPU，可选源）|
| [filter.py](../../python/gesture_mapping/filter.py) | OneEuro / EMA 滤波 |
| [camera.py](../../python/gesture_mapping/camera.py) | RealSense D455 / OpenCV 回退 |
| [main.py](../../python/main.py) | LeapNode 硬件驱动（`set_leap`/`OPEN_POSE`/PID/Dynamixel 初始化/安全门）|
| [measure_motor_limits.py](../../python/gesture_mapping/measure_motor_limits.py) | 手动推指实测电机限位 |
| `gesture_mapping/motor_limits.json` | 实测限位表（min/max）|
| `gesture_mapping/joint_gain_3d.json` | 3D 路径增益（demo_hamer3d 三源共用）|
| `gesture_mapping/calibration_3d.json` | 持久化 3D 校准基线 |
| [tests/](../../python/tests/) | 合成手点云单测 / hamer 离线集成 / 三源对比 |
| docs/debug/2026-08-05-half-clench-handoff.md | 半握问题调试交接（必读）|

## 7. 环境与命令

| 用途 | 环境 | 命令 |
|------|------|------|
| 实机驱动（伪3D/world-3D）| `leap_hand` | `cd python && python gesture_mapping/demo_hamer3d.py --drive` |
| 需要 hamer 3D 源 | `hamer` | 同上（M 键切源）|
| 无相机单图离线 | `leap_hand` | `python gesture_mapping/demo_hamer3d.py --img <路径>` |
| 限位测量 | 任一 | `python gesture_mapping/measure_motor_limits.py [--motor N]` |
| 单元测试 | `hamer` | `cd python && python -m pytest tests/ -q`（GPU 相关自动 skip）|

激活方式: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate leap_hand`

## 8. 工作方法（证据驱动）

- **先读代码再下结论**: 文档/注释可能滞后，一切以源码为准（尤其增益来源、默认 3D 源、滤波参数）
- **症状优先判定**: 先分清"不跟手"是振幅 / 滞后 / 抖动 / 方向哪一类，再深入；不要一开始就怀疑所有环节
- **可复现证据**: 每个根因结论需带证据（HUD/终端角度、限位表、单图离线输出、git 历史对照）
- **区分真机 vs 无真机**: 无真机时用 `--img` 离线 + 纯代码级分析；有真机时按 half-clench 文档 Step 2 先判"角度够不够"
- **不重复已做工作**: §4 列出的稳定性改动已完成，先假设有效，除非有反证

## 9. 约束与注意

1. 本轮只**调研 + 方案设计**，不实现（允许极小验证性改动，但需在报告中说明）
2. 真机操作前核对安全清单（CLAUDE.md §八）；SPACE 才上电
3. 改动 2+ 文件前备份（CLAUDE.md §九）
4. 不要动 vendor 目录（cpp/、ros_module/、ros2_module/、docs/reference/）
5. `motor_limits.json`、`joint_gain_3d.json`、`calibration_3d.json` 是**机器相关运行时数据**，方案若涉及改它们需说明兼容性

## 10. 交付物

1. **调研报告**: 按 §1 症状谱系逐一判定（哪些存在/不存在、严重度），每个根因带证据链
2. **方案设计**: 优先级（P0 最可能/最低成本见效 → P1 → P2），每个方案含: 改哪个文件、原理、预期效果、验证方法、风险与回滚
3. 写入 `docs/design/` 或 `docs/plans/`，命名 `2026-08-10-teleop-following-<方案名>.md`，并在 CLAUDE.md §六 更新状态（若结论影响阶段判定）
4. 若调研发现现有方案无法根治（如需要运动学重定向），给出评估与取舍（自研 vs 参考 AnyDexRetarget 移植）

---
*交接完。新会话从 docs/debug/2026-08-05-half-clench-handoff.md 和本简报 §2 开始。*
