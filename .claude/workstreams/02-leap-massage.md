# Window 2: LEAP Hand 按摩手势库 + LeRobot 适配 (🎮)

## Scope
| 文件/模块 | 用途 |
|-----------|------|
| `python/massage_gestures.py` | 按摩手法定义 (按/揉/推/捏/点/滚/拨) |
| `python/leap_lerobot/leap_robot.py` | LeapRobot(Robot) LeRobot 适配器 |
| `python/leap_lerobot/config_leap.py` | LeapRobotConfig 配置类 |
| `python/leap_lerobot/__init__.py` | 包导出 |
| `python/leap_lerobot/pyproject.toml` | 包安装配置 |

## Current State
- [x] main.py 有 LeapNode 类支持 set_pose/set_leap/set_joint
- [x] 7 个静态姿势录制完成
- [ ] 按摩手法关节序列定义
- [ ] LeRobot BYOH 适配器
- [ ] 与 W1 手势映射联调

## Next Tasks (prioritized)
1. 🔴 定义按摩手法: 每种手法 = 16DOF 时间序列 (轨迹)
   - 按(压): 手指伸直 + 控制力度
   - 揉: 拇指+食指/中指画圆
   - 推: 四指并拢前后滑动
   - 捏: 拇指+食指开合
   - 点: 单指指尖施压
   - 滚: 四指依次波浪
   - 拨: 拇指侧向拨动
2. 🔴 实现 massage_gestures.py (可被 interactive_control.py 和 LeRobot 调用)
3. 🔴 LeapRobot(Robot) 适配器:
   - connect(): 开串口 + 开扭矩
   - send_action(): 发 16DOF 目标角度
   - get_observation(): 返回关节位置 + 电流
   - calibrate(): 回到全开位
4. 🟡 与 W1 映射模块对接测试
5. 🟡 力控按摩: 基于电流反馈调节力度
6. 🟢 按摩参数化 (力度、频率、持续时间)

## Interfaces
### Output → W3
- `massage_gestures.py` 手势生成器 → 数据采集
- `LeapRobot.send_action()` → 同步录制管线

### Input ← W1
- `joint_mapper.map_keypoints_to_leap(keypoints)` → 动态手势驱动

## Session Start
```
加载 .claude/workstreams/02-leap-massage.md，当前任务: [描述]
```

## References
- main.py LeapNode: `set_pose()`, `set_leap()`, `set_joint()`
- 已录姿势: `main.py` POSES 字典
- LeRobot BYOH: huggingface.co/docs/lerobot/integrate_hardware
- Arm MassageRobot 参考: `/home/bright/office/Arm-robot_VLA/lerobot_robot_massage/massage_robot.py`
