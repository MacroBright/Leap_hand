# 视觉遥操优化实施记录 — Phase 1 + Phase 2（真机已验证）

> 日期: 2026-08-10
> 前置: 调研 `docs/design/2026-08-10-teleop-following-investigation.md` / 方案 `docs/design/2026-08-10-teleop-following-solution.md`
> 状态: **Phase 1+2 已实现且真机验证通过（防抖+跟手显著改善）**; Phase 3 已实现但真机效果未达预期, 按用户决定搁置
> 备份: `python/backups/2026-08-10_phase1-2-verified/`（真机验证版）/ `python/backups/2026-08-10_teleop-following-phase1/`

---

## 0. 一句话

为治"视觉遥操不跟手"（调研确认: 滤波滞后 300-500ms + 幅度衰减 + 拇指欠驱 + 折叠噪声），实施 P0 快赢 + P1 结构性修复，真机反馈**防抖与跟手均显著改善**；重定向模块（Phase 3）离线验证可行但真机不佳，暂搁置。

---

## 1. Phase 1 — P0 快赢（4 项改动 + 1 项基建）

### 1.1 P0-1 滤波重新平衡（治"慢半拍"）
- **文件**: `gesture_mapping/demo_hamer3d.py`
- **改动**: `angle_filter` OneEuro `min_cutoff 0.5→1.0Hz, beta 0.005→0.02`; `pseudo_smoother`(63d) `1.0→1.5Hz`
- **原理**: 原 0.5Hz 为压折叠噪声, 致级联滞后 + 动态幅度衰减; 抬到 1.0Hz + 高 beta → 静态平滑、快速跟手
- **效果**(仿真): 阶跃63% 500→333ms, 90% 967→633ms; 斜坡滞后 400→300ms; 1Hz 幅度 30%→46%

### 1.2 P0-2 拇指驱动修复（治"对掌不足"）
- **文件**: `joint_gain_3d.json` + `joint_mapper.py`
- **改动**: ID12 拇指mcp 增益 0.60→**1.3**, ID14 拇指pip 0.8→**1.2**; `_OPP_MCP/PIP_WEIGHT` 0.6→**0.9**
- **效果**(离线 probe): test5 拇指 mcp 输出 -0.07→**+0.41**, pip →+0.47

### 1.3 P0-2b 对掌补偿升级为 per-指缩放捏合
- **文件**: `joint_mapper.py`
- **改动**: `_thumb_opposition`(仅 TIP→掌心) → `_opposition_pinch`(跨掌对掌 + 各指捏合间隙, 掌宽归一, 取 max)
- **原理**: 不区分与哪指捏合; 缩放归一对伪3D 噪声鲁棒; 攥拳跨掌与指尖对捏都覆盖
- **注意**: per-指协同驱动留给 Phase 3, 未并入直映(保隔离单测)

### 1.4 P0-3 三源增益/基线解耦（治"M 键切源跳变"）
- **文件**: `demo_hamer3d.py`（`_apply_source_config()`）
- **改动**: 三种 3D 源(伪3D/world/hamer)各用独立增益+基线文件, M 切源换载
- **效果**: 切源不跳变; world/hamer 用恒等 1.0

### 1.5 LEAP FK + 四误差评估（基建）
- **新增**: `leap_fk.py`(numpy 手写 LEAP V1 正运动学, URDF 几何, q=0=开手验证) + `measure_following.py`(四误差: 指尖位置全局/相对腕/相对拇指 + 朝向, 掌心规范化可比)
- **发现**: ① LEAP 手指相对掌宽远短于人手指(0.9 vs 2+ 掌宽) → 四误差只作相对对比; ② 参考 URDF 拇指对掌行程受限(Phase 3 限制)

---

## 2. Phase 2 — P1 结构性修复（P1-1 按用户要求跳过）

### 2.1 P1-2 折叠 landmark 降噪（指尖朝向回退）
- **文件**: `joint_mapper.py` — `_FOLD_PIP_THRESHOLD=0.8` / `_FOLD_DIP_COUPLING=0.6`
- **改动**: 手指折叠(PIP>0.8)时 DIP = 0.6·PIP + 0.4·DIP_raw（末段 arccos 病态时回退近端朝向）
- **效果**(确定性验证):
  - 折叠 DIP 恢复: test1 中指 DIP 0.33→**1.26**（塌陷欠驱修复 → 帮"跟不足"）
  - 噪声降低 **60%**（300 次 TIP 噪声注入, std 0.545→0.218 → 帮"抖动"）
  - 四误差**无回归**

---

## 3. 真机验证结果（用户反馈）

> **Phase 1+2 上机后: 防抖和跟手都有显著改善, 基本满足要求 → ✅**

---

## 4. Phase 3 状态（重定向模块, 已实现, 真机搁置）

- **新增**: `retarget_obs.py`(reach-fraction + 掌心基底观测) + `retarget_mapper.py`(手写 numpy LM 求解器 + fast FK)
- **集成**: `demo_hamer3d.py --retarget`（替代直映, 无需校准基线）
- **离线验证**: 求解收敛(reach 误差<0.01)、实时(20ms/帧)、四误差优于直映(global -11%, rel_thumb -27%)
- **真机反馈**: 效果不好, 灵巧手与人手真实动作相差很大 → **按用户决定暂停优化, 模块保留但不启用**
- **可能方向**(未投入): 人手↔LEAP 掌心基底符号/朝向对齐、URDF 与真机运动学差异标定、目标权重调优

---

## 5. 当前代码状态

| 项 | 状态 |
|----|------|
| 改动文件 | `demo_hamer3d.py`, `joint_mapper.py`, `joint_gain_3d.json` |
| 新增文件 | `leap_fk.py`, `measure_following.py`, `retarget_obs.py`, `retarget_mapper.py`, `tests/test_leap_fk.py`, `tests/test_retarget.py` |
| 单测 | 34 passed（2 skip=GPU/hamer） |
| 备份 | `backups/2026-08-10_phase1-2-verified/`（真机验证版） |
| Phase 3 | `--retarget` 未启用时完全不影响直映路径 |

## 6. 待办/后续

- [ ] 若继续 Phase 3: 诊断重定向真机失真（基底符号对齐 + FK-真机运动学标定 + 权重）
- [ ] P2-2 人手-灵巧手尺寸归一化（LEAP 手指比例差, 评估/重定向的底层问题）
- [ ] 滤波可从 1.0Hz 尝试 1.5Hz（真机折叠抖动门槛允许时）
