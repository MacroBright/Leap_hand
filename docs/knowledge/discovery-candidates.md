# LEAP Hand 中医按摩灵巧手 — 技能/资料检索候选清单

> 生成日期: 2026-08-03
> 检索人: Discoverer (CoAgent 流水线)
> 检索方向: 灵巧手控制 / MediaPipe 视觉 / 具身智能与模仿学习 / 运动学 / Dynamixel 伺服
> 来源覆盖: GitHub API ✓ / skills.sh API ✓ / anthropics ✓ / Claude Marketplace ✓ / HuggingFace ✓

## 一、候选总表

类型: `skill`(Claude 技能) | `docs`(代码库/文档参考)
来源: `github` | `skills.sh` | `anthropics` | `marketplace` | `huggingface`

### 方向 1: 灵巧手 / 机器人手控制

| # | 类型 | 名称 | 来源 | URL | 相关度 | 说明 |
|---|------|------|------|-----|--------|------|
| 1 | docs | dex-hand-teleop (yzqin) | github | https://github.com/yzqin/dex-hand-teleop | 高 | RA-L/IROS2022「单相机人手遥操作+模仿学习」开源，思路与 MediaPipe→LEAP 映射高度一致，可直接借鉴手姿到机器手映射与数据采集 |
| 2 | docs | AnyDexRetarget (qqsq12321) | github | https://github.com/qqsq12321/AnyDexRetarget | 高 | 多灵巧手（含 LEAP）高精度手姿重定向，支持相机/Vision Pro/Quest 输入，正对应 W1 的 21 关键点→16DOF 重定向需求 |
| 3 | docs | Bidex_Manus_Teleop (leap-hand) | github | https://github.com/leap-hand/Bidex_Manus_Teleop | 高 | LEAP Hand 官方手套遥操作实现，可作真实设备端遥操作与控制回读参考 |
| 4 | docs | rwr_system (srl-ethz) | github | https://github.com/srl-ethz/rwr_system | 中 | 灵巧手遥操作+数据采集+推理全链路课程代码，覆盖单相机遥操作与数据管线 |

### 方向 2: MediaPipe / 计算机视觉

| # | 类型 | 名称 | 来源 | URL | 相关度 | 说明 |
|---|------|------|------|-----|--------|------|
| 5 | skill | mediapipe-usage (liuchiawei) | skills.sh | https://skills.sh/skills/liuchiawei/agent-skills/mediapipe-usage | 中 | MediaPipe 使用通用技能（47 安装），辅助 W1 快速上手 landmark 提取，偏通用需裁剪 |
| 6 | skill | hand-gesture-recognition (omer-metin) | skills.sh | https://skills.sh/skills/omer-metin/skills-for-antigravity/hand-gesture-recognition | 中 | 实时手势识别技能（31 安装），对应按摩手法手势分类需求，可作分类层参考 |
| 7 | skill | hand-tracking (openclaw-graph) | skills.sh | https://skills.sh/skills/alphaonedev/openclaw-graph/hand-tracking | 中 | 手部跟踪技能（56 安装），含 landmark 连续跟踪，适合实时驱动场景 |

### 方向 3: 具身智能 / 模仿学习 (LeRobot / VLA)

| # | 类型 | 名称 | 来源 | URL | 相关度 | 说明 |
|---|------|------|------|-----|--------|------|
| 8 | docs | LeRobot (huggingface) | github | https://github.com/huggingface/lerobot | 高 | 项目指定 BYOH 集成框架本体，26.3k stars，含 robot 适配器/数据格式/训练推理全链路 |
| 9 | docs | LeRobot BYOH 集成文档 | huggingface | https://huggingface.co/docs/lerobot/integrate_hardware | 高 | W2 直接目标文档：自定义硬件接入 LeRobot 的标准流程与适配器规范 |
| 10 | docs | openarmx_vla (openarmx) | github | https://github.com/openarmx/openarmx_vla | 中 | LeRobot 兼容 VLA 部署包，直接支持 SmolVLA/Pi0/Pi0.5，给出真机 VLA 推理参考实现 |
| 11 | docs | Lerobot-Mujoco (laohao78) | github | https://github.com/laohao78/Lerobot-Mujoco | 中 | LeRobot+MuJoCo 中文教程，完整复现 ACT/pi0/SmolVLA 数据采集-训练-部署，贴合本项目仿真+真机路线 |
| 12 | skill | NVIDIA/skills (Physical AI) | github | https://github.com/NVIDIA/skills | 中 | NVIDIA Agent Skills 集（2.8k stars），含 Isaac Sim/Lab、LeRobot 可视化(i4h-lerobot-viz)与模仿学习微调工作流 |

### 方向 4: 运动学 (关节映射 / IK / 手部模型)

| # | 类型 | 名称 | 来源 | URL | 相关度 | 说明 |
|---|------|------|------|-----|--------|------|
| 13 | docs | LEAP_Hand_API (leap-hand) | github | https://github.com/leap-hand/LEAP_Hand_API | 高 | LEAP Hand v1 官方控制 API（170 stars），本项目 Dynamixel 底层/关节 ID 协议的权威参考 |
| 14 | docs | LEAP_Hand_V2_Adv_API (leap-hand) | github | https://github.com/leap-hand/LEAP_Hand_V2_Adv_API | 高 | LEAP v2 进阶 API 含官方 IK 代码，手部运动学求解的现成参考，可移植用于 16DOF 映射验证 |
| 15 | skill | mujoco (coolbeevip) | skills.sh | https://skills.sh/skills/coolbeevip/mujoco-skills/mujoco | 中 | MuJoCo 建模/仿真技能，辅助 6-DOF 臂+手联合仿真场景搭建 |
| 16 | skill | manipulation-ik (isaac-sim) | skills.sh | https://skills.sh/skills/isaac-sim/isaacsim/manipulation-ik | 中 | Isaac Sim 操作 IK 技能，若沿用 Isaac 系仿真可作为 IK 求解对照 |

### 方向 5: Dynamixel / 伺服控制

| # | 类型 | 名称 | 来源 | URL | 相关度 | 说明 |
|---|------|------|------|-----|--------|------|
| 17 | docs | openhand-software (grablab) | github | https://github.com/grablab/openhand-software | 中 | Dynamixel 直连 Python 控制库（OpenHand 系列），与本项目「无 STM32、PC 直连」架构同构，可参考位置/力矩控制模式 |
| 18 | skill | robot-control (dora-rs) | skills.sh | https://skills.sh/skills/dora-rs/dora-skills/robot-control | 低 | 数据流式机器人控制框架技能，仅当 W3 采集管线需异步数据流时参考 |

## 二、来源覆盖记录

| 来源 | 状态 | 结论 |
|------|------|------|
| GitHub (API) | ✓ 完成 8 个查询 | 灵巧手/LeRobot/MediaPipe/Dynamixel/LEAP 官方均有高质量候选 |
| skills.sh (API) | ✓ 完成 5 个关键词查询 | 找到 4-5 个中相关技能；无官方机器人技能，社区为主 |
| anthropics/skills | ✓ 已遍历仓库树 | **无**：官方技能均为 docx/pptx/pdf/webapp 等文档办公类，无机器人/视觉技能 |
| anthropics/claude-cookbooks | ✓ 已遍历仓库树 | 仅 multimodal/vision notebook，无 MuJoCo/机器人/模仿学习教程 |
| Claude Marketplace | ✓ 检查 wshobson/agents 等 | **无**机器人专用插件（仅 vision-sft 等 LLM 微调类）；通用市场如 wshobson/agents (38k stars) 可留作后续检索入口 |
| HuggingFace | ✓ 文档站不可直连(curl/WebFetch 均被拒) | 以 GitHub 仓库 + 已知官方文档 URL 收录 |

## 三、建议安装/使用方式

- **skill clone**（#5,6,7,12,15,16,18）: `claude skill add <owner>/<repo>`（skills.sh 提供的源仓库）；低相关(#18)建议仅阅读不安装。
- **docs clone**（#1,2,3,4,8,10,11,13,14,17）: `git clone` 至 `docs/reference/` 作为实现参考，重点精读 #8/#9/#13/#14。
- **docs 在线**（#9）: 直接访问 HuggingFace 文档页。

## 四、风险与备注

- #2 AnyDexRetarget 与 #1 dex-hand-teleop 均需自行评估 License 与对 LEAP V1 的适配成本。
- #3 Bidex_Manus_Teleop 使用手套遥操作，与本项目 MediaPipe 视觉方案不同，仅参考设备端控制/回读逻辑。
- skills.sh 检索到的 MediaPipe/手势类技能多为社区小项目，质量参差，建议仅作参考模板，核心映射逻辑仍需项目自研。
- WebSearch 工具在本环境不可用，skills.sh 与 marketplace 均改由 API/curl 完成检索，覆盖等价。
