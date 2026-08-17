# Window 3: 同步采集管线 (🔗)

## Scope
| 文件/模块 | 用途 |
|-----------|------|
| `python/data_pipeline/sync_recorder.py` | 臂+手同步录制 |
| `python/data_pipeline/convert_to_lerobot.py` | 原始数据 → LeRobot v2 格式 |
| `python/data_pipeline/episode_manager.py` | Episode 管理与元数据 |
| `python/data_pipeline/calibration_recorder.py` | 校准数据录制 |

## Current State
- [x] Arm-robot_VLA 有 sim 采集管线 (record_sim.py + convert_to_lerobot.py)
- [x] LEAP Hand 可独立控制
- [ ] 臂+手同步通信协议
- [ ] 联合同步录制脚本
- [ ] LeRobot v2 格式转换 (22DOF = 臂6 + 手16)
- [ ] 数据版本管理

## Next Tasks (prioritized)
1. 🔴 设计同步协议: 臂 STM32 串口 + 手 Dynamixel USB, 统一时间戳
2. 🔴 实现 sync_recorder.py:
   - 开两个连接: SerialProtocol(臂) + LeapNode(手)
   - 录 arm 6DOF + hand 16DOF 关节角
   - 录相机帧
   - 统一时间戳, 存为 episode 目录
3. 🔴 改造 convert_to_lerobot.py 支持 22DOF (6 arm + 16 hand)
4. 🟡 Episode 命名规则与元数据管理
5. 🟡 数据质量检查 (关节范围、帧率一致性)
6. 🟢 数据集版本管理 (massage_v1, v2, ...)

## Interfaces
### Output → SmolVLA 训练
- LeRobot v2 格式数据集 (22DOF action + camera frames + state)

### Input ← W2
- `LeapRobot` 实例 (send_action, get_observation)
- `massage_gestures` 手势库

### Input ← Arm-robot_VLA
- `MassageRobot` 实例
- `SerialProtocol` 串口协议
- `convert_to_lerobot.py` 参考实现

## Session Start
```
加载 .claude/workstreams/03-data-pipeline.md，当前任务: [描述]
```

## References
- Arm 采集管线: `/home/bright/office/Arm-robot_VLA/scripts/record_sim.py`
- Arm 转换脚本: `/home/bright/office/Arm-robot_VLA/scripts/convert_to_lerobot.py`
- LeRobot 数据格式: huggingface.co/docs/lerobot/datasets
- LEAP Hand: `main.py` LeapNode
