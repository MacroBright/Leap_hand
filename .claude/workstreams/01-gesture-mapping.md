# Window 1: 手势识别 → 关节映射 (🤚)

## Scope
| 文件/模块 | 用途 |
|-----------|------|
| `python/gesture_mapping/hand_tracker.py` | MediaPipe Hands 人手 21 关键点实时追踪 |
| `python/gesture_mapping/joint_mapper.py` | 关键点 → LEAP Hand 16DOF 角度映射 |
| `python/gesture_mapping/wrist_tracker.py` | 手腕空间位置 → 机械臂末端 6DOF |
| `python/gesture_mapping/calibrator.py` | 零位校准 + 手指识别 |
| `python/gesture_mapping/filter.py` | 1€ 时域滤波 (去抖) |
| `python/gesture_mapping/demo_realtime.py` | 实时 demo (camera→detect→map→drive) |
| `python/gesture_mapping/__init__.py` | 包导出 |
| `python/gesture_mapping/models/` | MediaPipe TFLite 模型 |

## Current State
- [x] LEAP Hand Python SDK 可用 (main.py)
- [x] 7 个静态手势已录制 (poses.json)
- [x] MediaPipe Hands 安装与验证
- [x] 关键点 → LEAP 关节映射算法
- [x] 零位校准 + 手指识别
- [x] 1€ 时域滤波去抖
- [x] 逐关节增益校准 (_JOINT_GAIN)
- [x] 关节方向校正 (JOINT_DIR, 已实测修正, 拇指 12=mcp/13=side)
- [x] 电机 ID → landmark 诊断显示 (按 D 切换)
- [ ] 手腕 6DOF 空间定位
- [ ] 复杂手势精度优化 (待测试后调参)

## Next Tasks (prioritized)
1. ✅ **3D 手部网格重建（解决 3D 失效）**：已实现 hamer 3D MANO 集成（MediaPipe bbox → hamer kp3d，跳过 ViTPose）。
   - 模块: `python/gesture_mapping/hamer_3d.py` + `demo_hamer3d.py`
   - 运行: `python gesture_mapping/demo_hamer3d.py [--drive]`（hamer env; 默认 MediaPipe bbox → hamer MANO kp3d → JointMapper）
   - 验证: `tests/test_hamer3d_offline.py` 投影对齐中位距 21.3px；`tests/test_compare_3d_sources.py` 首轮对比 test1 攥拳 hamer 正确报出而 MediaPipe 低估
2. 🟡 实机验证 hamer 3D（no-drive 叠加 → --drive 驱动真手，倾斜/旋转/攥紧）
3. 🟡 继续调 _JOINT_GAIN 优化手指弯曲灵敏度 (当前: abd×0.4, mcp/pip/dip×1.5)
4. 🟡 手腕 3D 位置估计 (MediaPipe Pose/Holistic)
5. 🟡 手腕位置 → 机械臂 IK 解算
6. 🟢 人手-灵巧手尺寸校准

## HaMeR 环境速查 (2026-08-05 搭好)

- conda env `hamer`（py3.10，torch 2.13.0+cu130，勿 conda install）；`_DATA` 权重在 NTFS 软链
- 冒烟测试：`cd /home/bright/office/hamer && python smoke_test.py --img example_data/test1.jpg --bench 10`
- 迁移记录：`/home/bright/office/win2uban_condainstal/migration-log.md` Task 4

## ⚠️ 关节方向 (实测确认)

手势映射输出语义: **"正值 = 向手心弯曲"** (mcp/pip/dip) / **"负值 = 向拇指方向"** (index/middle/ring side)

⚠ 拇指电机顺序与其他指不同: **ID 12 = mcp, ID 13 = side** (2026-08-04 更正编号)
⚠ 关节方向 (2026-08-04 真机实测): index/middle/ring 全为 -1; 拇指仅 mcp 为 +1 (正值弯向手心)

| ID | 关节 | DIR |
|----|------|-----|
| 0 | 食指 side | -1 |
| 1 | 食指 mcp | -1 |
| 2 | 食指 pip | -1 |
| 3 | 食指 dip | -1 |
| 4 | 中指 side | -1 |
| 5 | 中指 mcp | -1 |
| 6 | 中指 pip | -1 |
| 7 | 中指 dip | -1 |
| 8 | 无名指 side | -1 |
| 9 | 无名指 mcp | -1 |
| 10 | 无名指 pip | -1 |
| 11 | 无名指 dip | -1 |
| 12 | 拇指 mcp | **+1** (真机实测: 正值弯向手心) |
| 13 | 拇指 side | -1 |
| 14 | 拇指 pip | -1 |
| 15 | 拇指 dip | -1 |

```python
JOINT_DIR = np.array([-1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  1, -1, -1, -1])
target = OPEN_POSE + JOINT_DIR * gesture_angle
```

## 逐关节增益 (_JOINT_GAIN)

解决侧摆过灵敏、弯曲不够灵敏的问题:

```python
_JOINT_GAIN = np.array([
    0.4, 1.5, 1.5, 1.5,   # index:  abd↓  mcp↑  pip↑  dip↑
    0.4, 1.5, 1.5, 1.5,   # middle
    0.4, 1.5, 1.5, 1.5,   # pinky→ring
    0.6, 1.2, 1.2, 1.2,   # thumb:  abd↓  mcp↑  pip↑  dip↑
])
```

## 手指映射 (5→4, 已修正标签)

```
人手      →  LEAP Hand      →  电机 ID
拇指      →  拇指            →  12-15
食指      →  食指            →  0-3
中指      →  中指            →  4-7
无名指    →  ✂️ 舍弃
小指      →  无名指 (LEAP ring) →  8-11
```

## side 扇角映射 (2026-08-04 新方案)

**平面内带符号扇角** (in-plane fan)，替代原"近端指骨抬离手掌平面"的旧外展算法。

- **固定基准**：中指近端方向 (投影到手掌平面) 为 0°，并拢时各指扇角 ≈ 0
- **拇指 side**：相对**食指近端方向**计算（用户选定）
- **符号**：`lateral` 轴指向食指/拇指侧，`atan2(seg·l_ref, seg·ref_dir)` → "向拇指方向张开"为正
- **四指符号**：`_FAN_SIGN` 表（默认全 +1），真机发现某指反时改 -1
- **关节方向衔接**：扇角正 → `JOINT_DIR[side]=-1` → 电机向拇指方向

| 手型 | 扇角 | side 电机 |
|------|------|----------|
| 并拢 | index/middle/ring ≈ 0 | 不动 |
| 全张 | index +、middle ≈0、ring/pinky − | 食指朝拇指、无名指离拇指扇开 |

⚠️ `_FAN_SIGN` 与拇指方案需真机验证 (待测试后调参)；`_JOINT_GAIN` 的 side 阻尼 0.4/0.6 可能需随真角重调。

## 管线架构

```
Camera → HandTracker.detect()
       → JointMapper.map_keypoints_to_leap()  [+ _JOINT_GAIN]
       → Calibrator.map()                     [zero-offset]
       → OneEuroFilter()                      [anti-jitter]
       → JOINT_DIR * angles                   [direction fix]
       → OPEN_POSE + result → leap.set_leap()
```

## 操作方式

```bash
cd ~/office/Leap_Hand/python && ../venv/bin/python gesture_mapping/demo_realtime.py --drive
```

| 按键 | 功能 |
|------|------|
| 空格 | 零位校准 (手全张开) |
| D | 切换电机 ID 诊断面板 |
| Q/ESC | 退出 |

## References
- LEAP Hand 关节映射: `main.py` 中 POSES 字典
- 电机 ID 表: ID 0-3=食指, 4-7=中指, 8-11=无名指, 12-15=拇指
- MediaPipe Hands: developers.google.com/mediapipe/solutions/vision/hand_landmarker
- Arm-robot_VLA: `/home/bright/office/Arm-robot_VLA/`
