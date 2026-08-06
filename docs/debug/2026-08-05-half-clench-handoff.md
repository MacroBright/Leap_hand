# 灵巧手实机调试交接文档 — 攥拳跟随不足（人手全攥，灵巧手半握）

> 交接对象: 负责实机调试的 agent
> 项目: LEAP Hand 中医按摩灵巧手 — 手势映射 W1 workstream
> 故障: 人手攥拳（全弯），灵巧手只能半握 — 跟随程度/屈曲幅度不足
> 日期: 2026-08-05

---

## 0. 一句话定位

**驱动链路**：`相机 → MediaPipe 手检测 → 3D 关键点(world-3D) → JointMapper 算 16 关节角 → 校准减基线 → 1€滤波 → 符号/增益 → 绝对位姿 → 电机限位裁剪 → Dynamixel 写入`。

"半握" = 链路中**某个环节把屈曲幅度压小了**，需要逐段定位是"角度没算够"还是"电机没走到位"。

---

## 1. 控制映射原理（调试 agent 必读）

### 1.1 硬件与 16-DOF 布局

LEAP Hand 14-DOF 右手 + 拇指 2 个附加 = 16 电机（ID 0-15），Dynamixel XC330-M288，USB 直连 `/dev/ttyUSB0`（协议 2.0，4M baud）。

| 电机 ID | 手指 | 关节 | 说明 |
|---------|------|------|------|
| 0 | 食指 | MCP侧摆 | 沿中指轴线张开/并拢 |
| 1,2,3 | 食指 | MCP/PIP/DIP | 屈曲 |
| 4 | 中指 | MCP侧摆 | **固定轴线，不动**（设计如此，限位=OPEN）|
| 5,6,7 | 中指 | MCP/PIP/DIP | |
| 8 | 小指 | MCP侧摆 | |
| 9,10,11 | 小指 | MCP/PIP/DIP | |
| 12 | 拇指 | MCP | **注意：12=mcp，13=side（拇指物理接线与其他指相反）** |
| 13 | 拇指 | 侧摆 | |
| 14,15 | 拇指 | PIP/DIP | |

**人手→灵巧手映射**：食指→ID0-3，中指→ID4-7，**小指→ID8-11**，拇指→ID12-15；**无名指丢弃**。`joint_mapper.py:_FINGER_MAP`。

### 1.2 关节角怎么算（JointMapper）

- 输入：21 个 3D 关键点（**world-3D**：MediaPipe 自带规范 3D 手模型，米制，绕旋转稳定；备选 hamer MANO 3D / 伪3D）。
- 每个手指 4 关节角：
  - **屈曲角**（mcp/pip/dip）= 相邻三点的空间夹角（`_compute_flexion`，arccos 点积）。
  - **侧摆角**（abd）= 手指第一段与"中指参考方向"的平面夹角（`_compute_fan_angle`），中指自身≈0。
- 输出 16 维相对角度 → **`_ANGLE_MIN=-0.5` / `_ANGLE_MAX=2.8` rad** 裁剪 → **乘每关节增益 `_JOINT_GAIN`**。

### 1.3 增益 `_JOINT_GAIN`（关键调试点）

`joint_mapper.py:43-48` 默认值：
```
ID 0,4,8 (侧摆): 0.4×   ← 压低，侧摆易过冲
ID 1-3,5-7,9-11 (mcp/pip/dip): 1.5×  ← 放大屈曲（当初为补偿 MediaPipe 伪z的"相机压扁"）
ID 12 (拇指mcp): 1.5×    ID 13 (拇指侧): 0.6×
```

⚠️ **重要**：demo_hamer3d.py 用的是**独立增益文件 `joint_gain_3d.json`**，且默认（无文件时）为**恒等 1.0×**（因为 world-3D 是真 3D，弯曲值已准）。这就是"半握"的首要嫌疑——**1.0× 没有放大屈曲**。真机要全握，需要把屈曲关节（mcp/pip/dip）增益调大（1.2-2.0×）。demo 支持 **TAB 选关节 + `[`/`]` 调增益 + S 保存**。

### 1.4 符号 JOINT_DIR（demo_hamer3d.py）

```
JOINT_DIR = [-1,-1,-1,-1, -1,-1,-1,-1, -1,-1,-1,-1,  1,-1,-1,-1]
             食指            中指            小指       拇指(12=+1)
```
含义：绝对位姿 = `OPEN_POSE + JOINT_DIR * angles`。mcp/pip/dip 负值 = 向手心弯；**拇指 mcp(ID12) 例外 +1**（接线方向不同）。方向反了会"越弯越伸"——调试时先确认方向对。

### 1.5 校准（Calibrator）

- SPACE 记录"全开手"基线（`_baseline_points`，3D 路径），之后 `map_points` 输出 `clip(当前角度 − 基线, -0.3, 2.8)`。
- **两套基线槽位分离**：3D 路径用 `_baseline_points`，伪3D 用 `_baseline`，不可混用（demo 已按源自动选）。
- 基线持久化到 `calibration_3d.json`（启动自动加载）。

### 1.6 滤波

- `angle_filter`（OneEuro 16 维，min_cutoff≈1.0, beta≈0.007）：输出角度时域平滑。
- `world_smoother`（OneEuro 63 维，min_cutoff=1.5）：world 关键点平滑。
- 帧级质量门控 `_MIN_VIS=0.55`：可见度低 → 保持上一帧好角度。

### 1.7 电机限位裁剪

- `motor_limits.json`（你实机测量）min/max 16 电机。写入前：`pose = clip(OPEN_POSE + JOINT_DIR*angles, min, max)`。
- 若测量值偏小（尤其 DIP 回绕关节），裁剪会**提前封顶屈曲** → 也是"半握"嫌疑。

### 1.8 延迟上电（安全）

真机启动**不上电**。张开手对准相机 → 按 **SPACE**（记录全开基线）→ **才连接上电**（写到全开位）。HUD 显示红色提示。

---

## 2. 故障分析与排查优先级

"人手全攥 → 灵巧手半握" = 屈曲幅度被压小。嫌疑按概率排序：

| # | 嫌疑环节 | 证据 | 排查方法 |
|---|---------|------|---------|
| 1 | **增益=1.0 未放大** | world-3D 屈曲角真实但 LEAP 手机械上需要更大行程才"看着全握" | 攥拳时看 HUD 角度：若 mcp/pip/dip 已经 ≥1.5 rad 但仍半握 → 就是增益不够 → 调大增益 |
| 2 | **电机限位裁剪提前封顶** | 屈曲关节 max 测小了 | 攥拳时对照 motor_limits.json 的 max 与 HUD 角度；若 `OPEN+DIR*angle` 触到 max → 裁剪生效，重测/放宽该关节限位 |
| 3 | **校准基线不准** | 全开基线≠真全开 → 相对角度偏小 | 重新张开手按 SPACE 校准；检查 baseline 是否接近 0 |
| 4 | **映射角度本身偏小** | world-3D 在攥拳时关键点退化 | 攥拳时对比 world-3D vs hamer 源的角度（M 键切换）；若 hamer 明显更大 → world-3D 攥拳退化 |
| 5 | **方向/接线错** | 某指越弯越伸 | 逐指弯一下看是否方向正确 |

**判定逻辑**：先确认**角度对不对**（HUD/终端 `print_angles_table`），再确认**电机走没走到**。

---

## 3. 分步调试流程

### Step 1 — 环境与启动

```bash
conda activate leap_hand        # 实机日常用 leap_hand（轻量，world-3D 默认）
cd ~/office/Leap_Hand/python
python gesture_mapping/demo_hamer3d.py --drive
```
启动后 HUD 红字提示未上电。**张开手对准相机 → 按 SPACE**（校准+上电）。确认 HUD 显示 `3D: WORLD 3D`、`CALIBRATED`。

### Step 2 — 测"攥拳"角度是否够大

人手攥拳，看终端（每 20 帧打印一次 `print_angles_table`）或 HUD：
- 关注 mcp/pip/dip 列。
- **若 ≥1.5 rad**（尤其 mcp/pip）：角度够了 → 问题在**增益不够放大**或**电机限位** → 去 Step 3。
- **若 <1.0 rad**：角度没算够 → 问题在映射/3D 源 → 去 Step 4。

### Step 3 — 调增益放大屈曲（最可能解决）

demo 实时调参：
- **TAB**：切换当前关节（0-15）
- **[** / **]**：当前关节增益 −0.05 / +0.05
- **S**：保存到 `joint_gain_3d.json`
- **R**：当前关节增益复位 1.0

做法：攥拳姿态，把**食指/中指/小指的 mcp/pip/dip（ID 1,2,3, 5,6,7, 9,10,11）增益逐步加到 1.5-2.0×**，看灵巧手能否从半握到全握。同时确认不超 `_ANGLE_MAX=2.8`。调到满意后 S 保存，重跑确认持久化。

> 若 HUD 角度已经 ≥1.5 rad 但仍半握，优先加增益——这是该故障最常见根因（恒等 1.0× 未补偿 LEAP 手行程）。

### Step 4 — 若角度偏小：查映射/3D 源

- 按 **M** 切到 `HAMER 3D`（需 `conda activate hamer` 环境）对比攥拳角度。
- 若 hamer 明显更大 → world-3D 在攥拳时关键点退化，考虑默认源换 hamer 或调 `_JOINT_GAIN`。
- 若两者都偏小 → 检查 `_compute_flexion` 与关键点链（`_FINGER_CHAIN`）是否正确；确认手离相机距离合适（30-60cm）。

### Step 5 — 查电机限位封顶

攥拳时终端对比：`OPEN_POSE + JOINT_DIR*angles` 是否触及 `motor_limits.json` 的 max。若封顶：
- 重测该关节限位：`python gesture_mapping/measure_motor_limits.py --motor <ID>`（手动推指法）。
- 或临时注释掉 demo 里的 `motor_limits` 裁剪验证是否放开后能全握（确认后重测限位）。

### Step 6 — 确认方向与逐指

逐指单独弯曲，确认每指方向正确（ID12 拇指 mcp 是 +1）。方向错会掩盖增益问题。

---

## 4. 关键文件

| 文件 | 作用 |
|------|------|
| `python/gesture_mapping/joint_mapper.py` | 16-DOF 角度计算核心（`_JOINT_GAIN`/`_ANGLE_*`/`_FINGER_MAP`/`map_points_to_leap`）|
| `python/gesture_mapping/calibrator.py` | 全开基线校准（`calibrate_points`/`map_points`，`_baseline_points`）|
| `python/gesture_mapping/demo_hamer3d.py` | 实时 demo：JOINT_DIR / 驱动写入 / SPACE上电 / 限位裁剪 / 增益调参键 |
| `python/gesture_mapping/hand_tracker.py` | MediaPipe VIDEO 模式跟踪（world 关键点来源）|
| `python/gesture_mapping/hamer_3d.py` | hamer MANO 3D（可选源）|
| `python/gesture_mapping/filter.py` | OneEuro/EMA 滤波 |
| `python/main.py` | LeapNode 硬件驱动（`set_leap`/`OPEN_POSE`/Dynamixel 初始化）|
| `python/gesture_mapping/measure_motor_limits.py` | 手动推指实测电机限位 |
| `python/gesture_mapping/motor_limits.json` | 实测限位表（min/max，写入前裁剪）|
| `python/gesture_mapping/joint_gain_3d.json` | 3D 路径增益（默认恒等 1.0，S 保存）|
| `python/gesture_mapping/calibration_3d.json` | 持久化校准基线 |

---

## 5. 环境与命令

| 用途 | 环境 | 命令 |
|------|------|------|
| 实机驱动（world-3D）| `leap_hand` | `python gesture_mapping/demo_hamer3d.py --drive` |
| 需要 hamer 3D 源 | `hamer` | 同上（M 键切换源）|
| 限位测量 | `leap_hand` 或 `hamer` | `python gesture_mapping/measure_motor_limits.py [--motor N]` |
| 全量测试 | `hamer` | `cd python && python -m pytest tests/ -q` |

**注意**：`hamer` env 有 torch/hamer（GPU）；`leap_hand` 无。demo 在 leap_hand 下 hamer 源自动跳过（`h3d.available=False`），world-3D 照常。

---

## 6. 已知坑 / 注意事项

1. **SPACE 才上电**：调试时手没张开就按 SPACE，基线错 → 角度全偏。务必手全开再按。
2. **motor_limits.json 是实测值**：DIP(回绕)关节若测量不准会提前封顶；电机 4 固定轴限位=OPEN。
3. **增益文件分离**：3D 路径用 `joint_gain_3d.json`（不是 `joint_gain.json`），别改错文件。
4. **方向**：ID12 拇指 mcp 符号是 +1，其他 mcp/pip/dip 是 -1；改 JOINT_DIR 前先确认。
5. **速度/力度**：XC330 位置-电流模式，`curr_lim=550mA`；若电机憋停/异响，可能是增益过大撞限位，先查限位表。
6. **HUD 角度看绝对值**：`print_angles_table` 打印的是相对角度（已减基线），攥拳时 mcp/pip 应明显 >1.0 rad。

---

## 7. 调试完成标准

- 人手攥拳 → 灵巧手能到**接近全握**（视觉对齐），且 mcp/pip/dip 不触发异常（无异响/过热/憋停）。
- 张手/半握/单指弯曲跟随自然，无抖动跳变。
- `S` 保存的 `joint_gain_3d.json` 在重跑后仍然生效。
- 电机 4（中指侧摆）保持不动，食指/小指沿轴线张开并拢正确。
