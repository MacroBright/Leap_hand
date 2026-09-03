# 视觉遥操"不跟手"解决方案设计

> 设计日期: 2026-08-10
> 前置调研: `docs/design/2026-08-10-teleop-following-investigation.md`（症状判定 + 证据链）
> 优先级: **P0**（最低成本/最可能见效）→ **P1**（结构性）→ **P2**（长期/重定向）
> 约束: 本轮只做设计，不实现。真机改动前按 CLAUDE.md §八 核对安全、§九 备份。

---

## 0. 设计总览

| 优先级 | 方案 | 改的文件 | 核心收益 | 成本/风险 |
|--------|------|---------|---------|----------|
| **P0-1** | 滤波重新平衡（抬 min_cutoff + beta） | `demo_hamer3d.py` 1-2 行 | 滞后 400→250ms、幅度 30%→46% | 抖动回潮风险，需真机门槛 |
| **P0-2** | 拇指驱动修复 | `joint_gain_3d.json` + `joint_mapper.py` 常量 + 重测限位 | 攥拳拇指 mcp ≥1.0 rad（对掌足） | 撞 ID14 限位，需先重测 |
| **P0-2b** | 对掌补偿升级为 per-指缩放捏合（借鉴重定向论文①） | `joint_mapper.py`（`_thumb_opposition` + `_OPP_*`） | 对掌 + 捏合同时改善（W2 按摩受益） | 与现 `_OPP_` 需替换而非叠加 |
| **P0-3** | 三源增益/基线解耦 | `demo_hamer3d.py` + 新 json | 切源不跳变 | 需各源重新校准 |
| **P1-1** | 默认源切 world-3D | `demo_hamer3d.py:290` | 根治折叠噪声源 → 滤波可再抬 | world 攥拳屈曲需 per-source 增益 |
| **P1-2** | 折叠 landmark 源头降噪（含指尖朝向回退，借鉴②） | `joint_mapper.py` / 新增 tip-collapse 检测 | 折叠稳定，允许低延迟滤波 | 实现中等复杂度 |
| **P1-3** | "跟手"量化评估脚本（四误差指标，借鉴③） | 新增验证脚本 + LEAP V1 FK | 每次调参有数值验收，不再目测 | 需 V1 FK，只作相对对比 |
| **P2-1** | 运动学重定向（指尖保真） | 新模块 | 按摩/捏合指尖位姿准 | 大工作量，见 §4 评估 |
| **P2-2** | 人手尺寸归一化 | `joint_mapper.py` | 人手 vs LEAP 比例适配 | 低 |

**推荐实施顺序**：P0-1 + P0-2 + P0-3 并行试（都在真机上快速验证），若 P0-1 抖动回潮 → 立即上 P1-1（换 world 源）。P2 等 W2 按摩手势需要指尖保真时再做。

---

## 1. P0 — 快速见效

### P0-1 滤波重新平衡（治"慢半拍"+ 动态幅度衰减）

- **改哪个文件**：`python/gesture_mapping/demo_hamer3d.py:212`。
  ```
  当前:  angle_filter = OneEuroFilter(n_joints=16, min_cutoff=0.5, beta=0.005)
  目标:  angle_filter = OneEuroFilter(n_joints=16, min_cutoff=1.0, beta=0.02)   # 起步值
  ```
- **原理**：仿真证明（30fps，kp_smoother 1.0Hz 级联）——**min_cutoff 是主导杠杆**，beta 单独几乎无效（0.5Hz/beta 0.03 仅降 33ms，因遥操角速度 1-5 rad/s 时 `cutoff=0.5+0.03×4≈0.62Hz` 仍太低）。抬到 1.0Hz：快速斜坡滞后 400→267ms、1Hz 正弦幅度 0.30→0.46；抬到 1.5Hz：滞后 233ms、幅度 0.54。
- **预期效果**：跟手明显改善（400ms→~250ms），动态幅度提升 ~50%。
- **验证方法**：
  1. 复跑 `/tmp/leap_lag_sim.py` 对照 0.5/1.0/1.5Hz 三档数字；
  2. 真机快速张握 10 次：目测跟手 + 听 DIP 是否抖动/异响；
  3. 真机折叠姿势静止 5s：HUD 角度是否颤动（抖动回潮门槛）。**若抖动回潮 → 回退 0.8Hz 并转 P1-1**。
- **风险与回滚**：唯一风险是折叠抖动回潮（0.5Hz→1.0Hz 静态平滑减半，DIP 增益 3.0 放大噪声）。回滚 = 一行改回 0.5。**不单独做**，需与 P1-1（换源/降噪）或 DIP 增益微降配合。

### P0-2 拇指驱动修复（治"对掌不足"）

- **改哪个文件**：
  1. `python/gesture_mapping/joint_gain_3d.json`：ID12 0.60→**1.3**，ID14 0.8→**1.2**（拇指 mcp/pip 屈曲提升 2 倍）。
  2. `python/gesture_mapping/joint_mapper.py:102-103`：`_OPP_MCP_WEIGHT` / `_OPP_PIP_WEIGHT` 0.6→**0.9**（对掌补偿有效增量 0.36→0.54 rad）。
  3. `measure_motor_limits.py --motor 14`：**重测 ID14 限位**（见下）。
  4. SPACE 校准时**拇指必须完全伸直**（否则基线虚高，吃掉屈曲行程）。
- **原理**：调研证据——ID12 增益 0.60 + 基线 0.306 + OPP 有效 0.36 rad，把攥拳时拇指 mcp 输出压到 ≈0（离线 test5 raw 0.83 → CAL≈0.19）。提高增益与补偿权重后有效屈曲应回到 1.0+ rad。
- **预期效果**：攥拳时拇指横跨掌心（对掌足），视觉与"全握拳"姿势一致。
- **验证方法**：
  1. 离线：`/tmp/leap_offline_probe.py` 看 test5 thumb CAL 是否 >0.8；
  2. 真机攥拳 HUD：thumb mcp 列 ≥1.0 rad；
  3. 视觉：拇指 TIP 触到食指/中指根。
- **风险与回滚**：⚠️ **ID14 拇指 PIP motor 可用屈曲仅 0.81 rad**（OPEN=2.157, min=1.351）——增益提高可能直接撞限位（表现为拇指 PIP 憋住）。**必须先重测 ID14 限位**，若行程真小，则拇指 PIP 只能接受 0.81 rad，靠 mcp+side 补偿对掌。回滚 = 恢复 json/常量。数据文件改动是机器相关运行时数据，改动前记录现值。

### P0-2b 对掌补偿升级为 per-指缩放捏合（借鉴重定向论文①）

> 来源：《Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation》（LEAP 手消融实验）结论——**捏合距离（pinch distance）是决定性目标**，去掉则拇指-指尖无法合拢、捏合成功率骤降；且**必须用缩放版**（A2 消融：原始捏合距离因人手跟踪误差而失败）。

- **改哪个文件**：`python/gesture_mapping/joint_mapper.py`——替换 `_thumb_opposition`（TIP→掌心距离）+ `_OPP_MCP/PIP_WEIGHT` 常量。
- **原理**：现对掌补偿只量"拇指 TIP 离掌心多近"，叠加固定 0.6 屈曲，**不区分与哪根指捏合**。升级为：对每个映射指（食指/中指/小指）计算**拇指 TIP → 该指 TIP 的间隙**，除以手掌宽度归一化（缩放，对伪3D 噪声鲁棒）；间隙小 → 该指 mcp/pip 与拇指 mcp/pip **同步**加强屈曲（soft 权重，非硬阈值）。
- **预期效果**：对掌（攥拳拇指跨掌）**和**捏合（拇指对食指/中指捏）同时改善——W2 按摩的"捏/揉"直接受益；比现启发式更稳（归一化吸收人手尺寸与伪3D 压缩）。
- **验证方法**：
  1. 离线：合成捏合点云（拇指 TIP→食指 TIP），确认两指屈曲同步增大（参照 `tests/test_joint_mapper.py::test_thumb_tip_to_palm_increases_flexion` 扩写）；
  2. 真机：拇指对食指/中指捏合，HUD 两指屈曲联动；对掌攥拳仍达 P0-2 指标。
- **风险与回滚**：**与现 `_OPP_` 必须替换而非叠加**（否则双重补偿过度内收）；伪3D 下折叠时 TIP 塌陷会让捏合间隙噪声大——依赖 P1-1/P1-2 源头降噪 + 滤波。回滚 = 恢复 `_thumb_opposition` 实现。

### P0-3 三源增益/基线解耦（治"M 键切源跳变"）

- **改哪个文件**：`demo_hamer3d.py`（增益加载 `:216-220`、SPACE 校准保存 `:479-484`）+ 新增每源配置文件（如 `joint_gain_world3d.json`、`calibration_world3d.json`）。
- **原理**：调研发现三源**共用** `joint_gain_3d.json`（为伪3D 压扁调的 DIP 3.0）与 `calibration_3d.json` 基线。world-3D 是真 3D 不需要 3.0 放大，切源必跳变。按 source_mode 分区加载/保存即可。
- **预期效果**：M 键切源角度连续，各源用各自最优增益。
- **验证方法**：M 键切换，HUD/终端角度无台阶。
- **风险与回滚**：需要为各源重新 SPACE 校准基线。改动集中在 demo 内增益/基线 key 分区，回滚容易。

---

## 2. P1 — 结构性修复（根治抖动-滞后循环）

### P1-1 默认源切 world-3D

- **改哪个文件**：`demo_hamer3d.py:290` `source_mode = 2` → `source_mode = 1`。
- **原理**：调研 §3.1 的核心——抖动滞后循环的**源头是伪3D 折叠噪声**（DIP 增益 3.0 放大 → 0.5Hz 压噪 → 滞后）。world-3D 是 MediaPipe 米制规范手，z 稳定、无折叠压扁。换源后折叠 landmark 稳定，滤波就允许抬回 1.0Hz+。
- **预期效果**：折叠静止不抖 + 跟手响应同时改善；伪3D 的"压扁/放大"负担消失。
- **验证方法**：M 键对比伪3D vs world-3D 攥拳角度（离线 `/tmp/leap_offline_probe.py` 已证两者发散）；真机攥拳 HUD ≥1.5 rad 判定。
- **风险**：handoff 记录 world-3D 攥拳屈曲不足（这就是当初换伪3D 的原因）——需用 per-source 增益（P0-3 的 world 增益档）修正；手快速旋转/离远时 world landmark 也可能退化。**这是本设计里唯一需要真机实测权衡的点**，若 world 实测更差则保留伪3D + P1-2。
- **回滚**：一行改回 `source_mode=2`。

### P1-2 折叠 landmark 源头降噪（备选根治）

- **改哪个文件**：`python/gesture_mapping/joint_mapper.py`（或 demo 内预处理）。新增**折叠检测**：某指 TIP 与 DIP 距离 < 阈值（折叠塌陷）时，该指 dip 屈曲改用"tip→base 归一化距离"或回退 mcp 角度，抑制 raw flexion 微小抖动。
- **原理**：MediaPipe 在折叠时 DIP/TIP landmark 相互吸附，raw arccos 对 ±几像素抖动极敏感。检测塌陷态并降噪，抖动源头消失 → 滤波可保持在 0.5-1.0Hz 低延迟。
  **指尖朝向回退（借鉴重定向论文②）**：《Analyzing Key Objectives...》结论——**指尖朝向项缺失 → 手弯得不像人**（A3：挂钩子任务失败）。我们 DIP 用末两段 arccos，折叠时 TIP 塌陷 → DIP 朝向不可靠（调研证据：DIP raw 在折叠时 0.01~0.44 乱跳）。因此折叠塌陷时，DIP 屈曲**回退用近端段（mcp→pip）朝向估计**，而不是死信末段 arccos——这就是"指尖朝向保真"的轻量实现。
- **预期效果**：与 P1-1 等效的"源头降噪"，但不依赖换源。
- **风险**：实现需防新假象（误判塌陷导致手指不跟）。验证：折叠姿态 HUD 静止稳定。
- **回滚**：该逻辑可开关。

### P1-3 "跟手"量化评估脚本（四误差指标，借鉴重定向论文③）

> 来源：论文用 **4 个指标**评估重定向——指尖位置误差（全局 / 相对腕 / 相对拇指）+ 指尖朝向误差，在 LEAP 手上做了真实对比。这正好把"视觉上像不像跟手"变成可测数字。

- **改哪个文件**：新增验证脚本（如 `python/gesture_mapping/measure_following.py` 或 `python/tests/`），配套需要 **LEAP V1 正向运动学（FK）**。
- **原理**：录一组人手姿势 → MediaPipe 人手指尖 → 当前映射出 16 角 → LEAP FK 算出机器手指尖位置/朝向 → 计算与目标指尖的 4 类误差。跨 3D 源 / 跨增益档 / 跨滤波参数对比，得到数值化的"跟手"评分。
- **FK 来源**：官方参考 `docs/reference/LEAP_Hand_API/useful_tools/mano_to_leap_mapping.py`（MANO→LEAP 官方角映射，作者自注"需 tune offsets/scaling"——印证我们结论）；V2 Adv API 的 **Telekinesis Node（SDLS 指尖 IK + PyBullet URDF）** 是官方现成方法，但面向 **V2 17-DOF**，移植到 V1 16-DOF 需自建/改 URDF FK。**四误差指标的具体实现可直接参考 `/tmp/retargeting/src/retargeting/evaluation/robot_metrics.py`**（论文官方代码，含 position/orientation/relative/relative-to-wrist error；见 §4.1）。
- **预期效果**：P0-1/P1-1 每次调参有数值验收（如"快速张握指尖误差 -30%"）；跨源对比支撑"默认源选伪3D 还是 world-3D"的真机决策。
- **验证方法**：本身即验证工具；在真机录 3-5 组姿势（张/握/捏/比耶/OK）回放对比。
- **风险**：人手→LEAP 尺度归一化会引入校准误差——**只作改动前后的相对对比**，不作绝对目标；V1 FK 需小工作量自建。

---

## 3. P2 — 长期（重定向与尺寸）

### P2-2 人手-灵巧手尺寸归一化（低风险先行）

- **改哪个文件**：`joint_mapper.py`（`_compute_flexion` 前对指尖做指长归一化，或映射后加每指屈曲压缩因子）。
- **原理**：人手各指长度比例 ≠ LEAP。直接角映射在指尖"到达"上无保证。按人手指长归一化到 LEAP 比例，或对 mcp/pip/dip 用每指屈曲比，改善指尖位置。
- **预期效果**：捏/按动作指尖更到位。
- **风险**：中低；需真机校准每指比例。

### P2-1 运动学重定向（指尖保真）——评估见 §4

---

## 4. 重定向评估：自研 vs 移植 AnyDexRetarget

**现状**：`joint_mapper.py` 直接角映射（arccos 屈曲 + atan2 扇角），无指尖位置优化、无尺寸归一化。

**为什么需要重定向**（W2 按摩手势前可先不做）：
- 论文《Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation》（2026，LEAP 手消融实验）结论：**指尖位置/朝向目标**是决定性项——去掉它捏合成功率骤降；只比关节角不够。
- DexPilot（NVIDIA/CMU）不用关节角，用 DART 跟踪 + **指尖位置**重定向（人手与 Allegro 关节轴完全不同，直接比角无意义）。
- 本项目按摩（按/揉/捏）本质是**指尖位姿任务**，现方案在指尖到达上无保证。

**两方案对比**：

| 维度 | 自研轻量 IK（推荐路径） | 移植 AnyDexRetarget |
|------|----------------------|--------------------|
| 原理 | 以拇指+食指指尖 3D 位置为目标，用 LEAP 运动学做少量迭代 IK/优化 | 通用多手运动学重定向框架（优化式） |
| 依赖 | 无新增（参考 `docs/reference/LEAP_Hand_V2_Adv_API` 官方 IK） | PyTorch + 优化库 + 框架本身 |
| 性能 | 每帧少量迭代，30fps 可达 | 优化式每帧求解，**延迟风险与 P0 滞后目标冲突** |
| 范围 | 只做需要的（拇指+食指），可控 | 全手通用，对本单右手单相机场景偏重 |
| 风险 | 需要 LEAP URDF/运动学参数 + 指尖目标归一化，工作量中等 | 需核对 **LEAP V1 支持**（框架多面向 V2/其他手）；License 需确认；调参黑盒 |
| 何时做 | W2 按摩手势需要指尖保真时 | 若人力有限且需全手保真 |

**建议**：**P0/P1 先行**——不解决滤波滞后与拇指驱动，重定向再好也被 300ms 滞后淹没。到 P2 阶段用**轻量自研**（指尖位置目标 + 官方 IK 参考），不移植通用框架；AnyDexRetarget 仅作备选，若走它先做 LEAP V1 支持与 License 可行性验证。

### 4.1 参考实现实测：Mingrui-Yu/retargeting（论文官方代码，2026-07 重构）

> 已克隆 `/tmp/retargeting` 核查（2026-08-10）。这是论文作者代码，**比 AnyDexRetarget 更适合本项目**。

| 项 | 实测结论 |
|----|---------|
| LEAP 支持 | ✅ `panda_leap_paxini.urdf` 的 `actuated_joints = joint_0…joint_15`（16-DOF，与我们 ID 0-15 完全对应），含 `assets/meshes/leap_hand` 网格 + `leaphand_ros2_module` |
| 目标项实现 | ✅ `src/retargeting/core/retargeter.py`：`thumb_primary_dist`（缩放捏合 + sigmoid 权重）、指尖位置/朝向、`world_thumb`、时间正则化；含 A1-A8 消融开关 |
| 离线验证 | ✅ `python -m retargeting_apps.main app=offline_retarget …`，MuJoCo + Viser 网页查看，**无硬件可跑** |
| 四指标评估 | ✅ `src/retargeting/evaluation/robot_metrics.py`（position/orientation/relative/relative-to-wrist error）——**正是 P1-3 要建的，可直接借鉴** |
| 实时性 | ⚠️ 每帧 nlopt 优化，`command_hz=20.0`（输出有 `max_joint_speed/command_hz` 限速）；**优化式延迟风险仍与 P0 滞后目标冲突** |
| License | ⚠️ **无 LICENSE 文件**（仅有 Citation）。论文已公开（arXiv 2506.09384），但代码无显式许可——**复制实现前须向作者确认**；借公式/思路无碍 |
| 依赖 | ⚠️ pinocchio(conda) + torch + nlopt + mujoco + mr_utils 子模块，Python 3.10，需独立 env |

**对本项目 P2-1 的用法**（保持"轻量自研"路线）：
1. **P1-3 直接受益**：`robot_metrics.py` 四指标 → 我们做"跟手"量化评估时可参考其实现（不复制，重写）；
2. **P2-1 的自研模板**：用该 repo 的**目标项公式**（缩放捏合 + 指尖位置/朝向）作为自研 IK 的目标函数，参考 `LEAP_Hand_V2_Adv_API` 的 SDLS 思路做轻量求解，而非移植整框架；
3. **LEAP 手资产**：其 URDF/网格可对照确认我们 motor_limits/OPEN_POSE 的几何一致性（不直接搬）。
4. **不建议**：整个移植（license 缺失 + 依赖重 + 优化式延迟 + 是"臂+手"装配，需剥出单手）。

---

## 5. 真机验证清单（P0/P1 落地时的最小验收）

1. 快速张握 ×10：跟手（无 0.3s+ 延迟）、无 DIP 异响/回弹。
2. 折叠姿势静止 5s：HUD 角度稳定（抖动门槛）。
3. 攥拳：HUD mcp/pip ≥1.5 rad（四指）、thumb mcp ≥1.0 rad、拇指 TIP 触掌心。
4. M 键切源：角度无台阶。
5. 保存后重跑：`joint_gain_*.json` / `calibration_*.json` 生效。

---

## 6. 方案改动涉及的数据文件（机器相关运行时数据）

| 文件 | P0-2 是否改 | 兼容性说明 |
|------|-----------|-----------|
| `joint_gain_3d.json` | ✅ ID12/ID14 | 改动前备份现值；P0-3 引入分源后可能拆成多个文件 |
| `calibration_3d.json` | 间接（重校准） | 拇指伸直重校准会覆盖 |
| `motor_limits.json` | ✅ 重测 ID14 | 确认 0.81 rad 行程真伪 |
| `poses.json` | ❌ 不动 | OPEN_POSE 权威值保持 |
