# LEAP Hand — 项目总纲 (CLAUDE.md)

> 中医按摩灵巧手项目。LEAP Hand 14-DOF 右手作为末端执行器，集成到 Arm-robot_VLA 管线。

---

## 一、项目定位

LEAP Hand 在完整系统中的角色：
1. **按摩执行器**：接收 SmolVLA 输出的 16DOF 动作，执行按/揉/推/捏等按摩手法
2. **独立测试平台**：脱离机械臂时可独立调试手势与力控
3. **手势映射终端**：接收 MediaPipe 人手关键点 → 实时驱动灵巧手

---

## 二、技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| **硬件** | Dynamixel XC330-M288 ×16 | 4M baud, Protocol 2.0, 位置-电流模式 |
| **通信** | USB 直连 PC (`/dev/ttyUSB0`) | 无 STM32 中间层 |
| **SDK** | Dynamixel SDK 4.0.5 (Python) | dynamixel_client.py 封装 |
| **控制** | LeapNode (main.py) | PID 控制, 姿势预设, 交互脚本 |
| **映射** | MediaPipe Hands / HaMeR 3D → LEAP 16DOF | W1 负责 (`gesture_mapping/`) |
| **框架** | LeRobot (HuggingFace) | BYOH 集成 (W2 负责) |
| **模型** | SmolVLA-450M | 臂+手联合推理 |
| **环境** | conda `leap_hand` (Python 3.14) | opencv-contrib 5.0.0, mediapipe 1.0.0, pyrealsense2, dynamixel-sdk 4.0.5, sounddevice(+portaudio), pyserial |

### 运行方式

```bash
conda activate leap_hand            # 不建 .venv; 首次需 source ~/miniconda3/etc/profile.d/conda.sh
python python/main.py               # 核心控制 (需接硬件)
cd python && python calibrate.py            # 校准 & 录制姿势
cd python && python interactive_control.py  # 逐指/手势交互
cd python && python gesture_mapping/demo_realtime.py [--drive]  # 实时手势映射 (伪3D)
cd python && python gesture_mapping/demo_hamer3d.py [--drive]   # 3D 源可切换 (hamer/world/伪3D)
```

> sounddevice 依赖 portaudio（已装于环境内）；如需重装: `conda install -c conda-forge portaudio`。

---

## 三、架构决策记录 (ADR)

### ADR-001: PC 直连 USB, 不经 STM32
- **决策**: 灵巧手沿用 Dynamixel SDK 的 PC 直连方式
- **原因**: LEAP Hand 开源项目原生为 PC 直连；STM32 仅负责机械臂的力控安全
- **影响**: 灵巧手无独立安全网关, 按摩力度控制需在 PC 端通过电流反馈实现

### ADR-002: 分层 VLA 输出 (臂 → 手)
- **决策**: SmolVLA 先输出臂 6DOF 到达目标位, 再输出手 16DOF 执行按摩
- **原因**: 训练复杂度低, 可分别验证臂和手的行为
- **影响**: 需两阶段推理

### ADR-003: 实物数据采集 (无仿真)
- **决策**: 手势识别 + 姿态映射的实物采集方式
- **原因**: LEAP Hand 柔性手指建模复杂, 真机数据更可靠
- **影响**: 需真机调试, 采集速度受限

### ADR-004: 仅右手
- **决策**: 本项目仅复刻右手, 电机 ID 固定 0-15

### ADR-005: 姿势数据以 poses.json 为准 (源码回退)
- **决策**: 姿势录制存 `python/poses.json`（`calibrate.py` 写入）；`main.py` 的 `POSES` 字典作为回退值
- **原因**: 真机需反复校准, 数据文件比源码更易更新; 读取失败的全零/常数坏数据由 `_is_valid_pose` 拦截
- **影响**: 当前 8 个姿势（7 手势 + 揉法）; 全开位数据无效时安全门拒绝驱动电机

---

## 四、多窗口并行工作流

| # | 窗口 | 简报 | 职责 |
|---|------|------|------|
| 1 | 🤚 手势映射 | `.claude/workstreams/01-gesture-mapping.md` | MediaPipe/hamer 3D → LEAP 16DOF; 手腕6DOF (TODO) |
| 2 | 🎮 按摩手势 | `.claude/workstreams/02-leap-massage.md` | 按摩手法库 + LeRobot 适配器 |
| 3 | 🔗 采集管线 | `.claude/workstreams/03-data-pipeline.md` | 臂+手同步录制 → LeRobot 格式 |
| 4 | 📝 文档穴位 | `.claude/workstreams/04-docs-acupoint.md` | CLAUDE.md + 穴位检测 + 实验追踪 |

**用法**: 新开 Claude Code 窗口 → `加载 .claude/workstreams/0X-xxx.md，当前任务: [描述]`

**依赖**: W1 → W2 (映射接口), W2 → W3 (LeapRobot), W4 → 全部 (文档)。

---

## 五、关节映射速查

| ID | 手指 | 关节 | 全开位 (LEAP rad) |
|----|------|------|-------------------|
| 0 | 食指 | MCP侧摆 | 3.1385 |
| 1 | 食指 | MCP前后 | 4.9133 |
| 2 | 食指 | PIP | 4.5390 |
| 3 | 食指 | DIP | 1.7089 |
| 4 | 中指 | MCP侧摆 | 3.0894 |
| 5 | 中指 | MCP前后 | 3.2858 |
| 6 | 中指 | PIP | 3.0204 |
| 7 | 中指 | DIP | 4.8903 |
| 8 | 无名指 | MCP侧摆 | 3.1170 |
| 9 | 无名指 | MCP前后 | 3.2428 |
| 10 | 无名指 | PIP | 3.0925 |
| 11 | 无名指 | DIP | 4.6296 |
| 12 | 拇指 | MCP前后 (mcp) | 3.5282 |
| 13 | 拇指 | MCP侧摆 (side) | 3.6386 |
| 14 | 拇指 | PIP | 2.1568 |
| 15 | 拇指 | DIP | 5.0667 |

> ⚠ **拇指物理接线与其他指相反**: ID 12=mcp, 13=side（其余四指为 ID x=侧摆, x+1=前后）。
> **人手→电机映射**: 食指→0-3, 中指→4-7, **小指→8-11 (LEAP 无名指)**, 拇指→12-15; 人手无名指舍弃。
> 全开位角度随校准变化, 以 `python/poses.json` 为准（`main.py` 硬编码值为回退）。

---

## 六、当前阶段

**Phase 2**: 手势映射 + 实物数据采集
- [x] Python SDK 可用, 交互脚本工作
- [x] 8 个姿势录制 (7 手势 + 揉法), 4 窗口拆分完成
- [x] W1: 手势识别 → 关节映射（MediaPipe 21kp → LEAP 16DOF, 含 hamer 3D 增强）
- [ ] W1 收尾: 手腕 6DOF 空间定位 + hamer 实机验证/增益调优
  - 遥操"不跟手"调研+方案: `docs/design/2026-08-10-teleop-following-investigation.md` / `-solution.md`
  - **P0+P1 已实现且真机验证通过**（防抖+跟手改善）: `docs/design/2026-08-10-teleop-following-implementation.md`
  - Phase 3 重定向模块已实现（`--retarget`）但真机效果不佳, 搁置; 直映路径不受影响
- [ ] W2: 按摩手势库 + LeRobot 适配
- [ ] W3: 同步采集管线
- [ ] W4: 文档 + 穴位优化

## 七、参考

- LEAP Hand RSS 2023: Shaw et al.
- Arm-robot_VLA: `/home/bright/office/Arm-robot_VLA/`
- LeRobot BYOH: huggingface.co/docs/lerobot/integrate_hardware
- SmolVLA: huggingface.co/papers/2506.01844
- XC330: e manual.robotis.com/docs/en/dxl/x/xc330-m288/

---

## 八、CoAgent 流水线

用 CoAgent 流水线推进：/plan → /implement → /test → /review → /integrate
出问题时用 /debug（独立上下文、证据驱动）。状态用 /status 查看。

领域知识：通用规范与踩坑记录在 `~/.coagent-knowledge/`；本项目计划与设计在 `docs/plans/` 与 `docs/design/`。
涉及硬件操作（电机上电、烧录、真机使能）前，必须核对 `~/.coagent-knowledge/safety/` 与 hardware-safety-checklist 技能。

---

## 九、改动前备份规则

**大改动前必须先备份当前版本**（改动 2+ 文件 / 算法重写 / 硬件映射调整）：

1. 备份到 `python/backups/<YYYY-MM-DD_描述>/`，复制**全部将改动文件**（含整个相关包）
2. 备份后确认文件已成功复制，再开始编辑
3. 小改动（单文件几行）不强制备份
4. 备份目录不入库（`.gitignore` 已忽略 `python/backups/`）

