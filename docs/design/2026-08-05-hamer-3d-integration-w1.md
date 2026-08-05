# HaMeR 3D MANO 集成 W1 — 设计文档

> 日期: 2026-08-05
> 目标: 解决"手倾斜/旋转/攥紧时 3D 失效" —— 用 hamer 的 MANO 真 3D 关键点替代 MediaPipe 伪 3D z 坐标。
> 关联: W1 workstream `.claude/workstreams/01-gesture-mapping.md` Next Task #1

---

## 1. 背景与根因

当前 W1 管线:

```
Camera → HandTracker(MediaPipe 21kp) → Calibrator.map()
       → JointMapper.map_keypoints_to_leap()   [+ _JOINT_GAIN]
       → OneEuroFilter → JOINT_DIR → OPEN_POSE + → leap.set_leap()
```

`JointMapper._build_point_cloud()` 用 MediaPipe 的 z 坐标 `lm.z * w` 构造 (21,3) 点云。
MediaPipe 的 z 是**伪深度**（由 2D 模型回归, 单位无量纲, 相对腕部）。当手倾斜/旋转/攥紧时:

- 手指在 2D 上重叠（遮挡）→ 伪 z 不可靠
- 弯曲（弯向/弯离相机）的深度分量无法从 2D 恢复
- 结果: 倾斜/旋转/攥紧时 flexion 角度漂移或失效

**修复**: 用 hamer 回归 MANO 参数 → 得到真 3D 关键点 kp3d 与网格 verts。hamer 是完整 3D 参数化手模型, 对旋转/攥紧稳健。

## 2. 已确认的决策

| 决策 | 选择 |
|------|------|
| 运行环境 | `hamer` conda 环境装齐依赖 (pip: mediapipe/dynamixel_sdk/pyrealsense2/pyserial) |
| 交付形态 | 独立模块 `hamer_3d.py` + 独立 demo `demo_hamer3d.py`; 保留 demo_realtime.py 作回退 |
| 驱动范围 | 本步即支持 `--drive` 驱动真手 (默认不开) |
| 集成方式 | 方案 A: 3D 点云替换 (MANO kp3d → 复用 JointMapper 核心) |

## 3. 关键事实（探索确认）

### 3.1 hamer kp3d 顺序与 MediaPipe 索引 1:1 兼容

`hamer/models/mano_wrapper.py`: `mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]`

- 基础 MANO 16 关节: 0=wrist, 1-15=四指 mcp/pip/dip + 拇指 mcp/pip/dip
- `extra_joints` 取自 `vertex_ids['mano']` 指尖顶点: thumb→744, index→320, middle→443, ring→554, pinky→671
- 经 joint_map 重排后的 21 关节:

| idx | hamer 语义 | MediaPipe idx | 对齐 |
|-----|-----------|---------------|------|
| 0 | wrist | 0 | ✓ |
| 1-4 | thumb [mcp, pip, dip, tip] | 1-4 [cmc, mcp, ip, tip] | 链式兼容 |
| 5-8 | index [mcp, pip, dip, tip] | 5-8 | ✓ |
| 9-12 | middle | 9-12 | ✓ |
| 13-16 | ring | 13-16 | ✓ |
| 17-20 | pinky | 17-20 | ✓ |

→ **无需重排**, hamer kp3d 可直接按 MediaPipe 索引喂给 JointMapper 核心。

⚠ 差异: MANO 无独立拇指 CMC 关节。h[1] 是拇指 mcp（基底部）。作为链底仍有效, 仅轻微解剖偏移。

### 3.2 kp3d 尺度

- hamer kp3d 为**米制** MANO 手空间坐标（根在腕部附近）, 绝对位置/尺度不影响角度计算（`_compute_flexion`/`_compute_fan_angle` 用点积/叉积, 尺度不变）。
- 故 `map_points_to_leap()` 直接接收原始米制坐标, 不需按像素缩放。

### 3.3 环境分叉

| 包 | leap_hand env | hamer env |
|----|--------------|-----------|
| mediapipe | 1.0.0 ✓ | ✗ |
| dynamixel_sdk | ✓ | ✗ |
| pyrealsense2 | ✓ | ✗ |
| serial/pyserial | ✓ | ✗ |
| torch | ✗ | 2.13.0+cu130 ✓ |
| hamer | ✗ | ✓ |
| detectron2 | ✗ | 0.6 ✓ |

→ 决策: 在 hamer env pip 安装 4 个缺失轻量包。torch/hamer/detectron2 不动。

### 3.4 显存与速度

- fp16 加载 ~1.5GB VRAM; 跳过 detectron2（MediaPipe 提供 bbox）→ 6GB 无压力
- hamer forward 稳态 ~30ms → 全管线 ~20-25fps; 可选 `--skip N` 降频

## 4. 架构与数据流

```
Camera (RealSense/OpenCV)
  → HandTracker.detect()                    [MediaPipe 始终跑: 给 bbox + 回退源]
  → hand_bbox_from_landmarks()              [方形裁剪 + margin, 全帧坐标]
  → crop → HaMeR3D.regress(crop)            [hamer fp16+autocast, 单手 batch=1]
        → kp3d (21,3) 米制 [顺序 1:1 MediaPipe]
        + verts (778,3)  + cam_t (弱透视)
  → JointMapper.map_points_to_leap(kp3d)    [新增核心, 复用 _palm_frame/flexion/fan/gain/clip]
  → Calibrator.map_points / FingerIdentifier.identify_points
  → OneEuroFilter → JOINT_DIR → OPEN_POSE + → leap.set_leap()   [--drive]
  → 叠加: MediaPipe 21kp(绿) + MANO kp3d投影(青) + mesh 线框(可选) + HUD
```

## 5. 模块设计

### 5.1 `python/gesture_mapping/hamer_3d.py`（新增）

```python
class HaMeR3D:
    available: bool        # False if import hamer fails / no CUDA
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, device="cuda", fp16=True):
        # 惰性 import hamer; 加载 fp16 (复用 smoke_test load_hamer_half); 失败→available=False
        pass

    def regress(self, crop_bgr: np.ndarray) -> "HaMeR3DResult" | None:
        # 单手 batch=1; torch.no_grad + autocast(fp16); 校验 kp3d finite
        # 返回 result 或 None(失败)

    def project(self, pts3d, cam_t, crop_box, frame_size) -> np.ndarray:
        # 弱透视: fx*(xy+cam_t.xy)/(z+cam_t.z) + 主点 → 全帧像素 (N,2)

class HaMeR3DResult:
    kp3d: np.ndarray   # (21,3) 米制, MediaPipe 索引顺序
    verts: np.ndarray  # (778,3) 米制
    cam_t: np.ndarray  # (3,)
    # 供叠加/诊断
```

辅助函数:
- `hand_bbox_from_landmarks(hand_result, image_shape, margin=1.5, square=True) -> (x0,y0,x1,y1)`:
  - 从 MediaPipe 21kp 取 min/max + margin; 扩为方形（hamer 输入 192x256 纵横比容忍）
  - clamp 到帧边界; 面积过小(< 32x32)返回 None
- `to_mediapipe_order(kp3d) -> kp3d`: 显式恒等映射, 注明 1:1, 便于未来改动

### 5.2 `JointMapper` 增补（纯加法, 不动现有行为）

- 新增 `map_points_to_leap(pts: np.ndarray(21,3)) -> np.ndarray(16)`:
  - 直接使用 pts, 跳过 `_build_point_cloud` 的归一化缩放
  - 复用现有 `_palm_frame` / `_compute_flexion` / `_compute_fan_angle` / `_JOINT_GAIN` / clip
- 新增 `map_points_to_leap_dict(pts) -> Dict[str, Dict[str, float]]`:
  - 与 `map_keypoints_to_leap_dict` 同构, 供 `FingerIdentifier.identify_points` 与 HUD 显示
- `map_keypoints_to_leap(hand_result, image_shape)` 重构为:
  ```python
  pts = self._build_point_cloud(hand_result, image_shape)
  return self.map_points_to_leap(pts)
  ```
- `map_keypoints_to_leap_dict(hand_result, image_shape)` 重构为委托 `map_points_to_leap_dict`

### 5.3 `Calibrator` 增补

- `calibrate_points(pts)` / `map_points(pts)`: 与现有 `calibrate/map` 同逻辑, 但用 `map_points_to_leap`
- ⚠ 基线必须用**同一 3D 源**捕获（hamer 米制 vs MediaPipe 像素伪 z 尺度不同, 不可混用）

### 5.4 `FingerIdentifier` 增补

- `identify_points(pts) -> (bent_finger, scores)`: 基于 `map_points_to_leap_dict` 的 points 版
- 可顺带复用 `identify_geometry` 的几何逻辑（纯 3D 几何, 尺度不变）

### 5.5 `demo_hamer3d.py`（新增）

- 复用 demo_realtime 基建: `find_best_camera` / `open_realsense` / `_OpenCVCamera` / `draw_hud` / `_MOTOR_DIAG`
- 每帧流程见第 4 节
- 按键:
  | 键 | 功能 |
  |----|------|
  | 空格 | 3D 校准 (手全张开, 用 kp3d 基线) |
  | D | 叠加诊断开关 (hamer kp3d + mesh) |
  | M | 切换显示源 (hamer 3D / MediaPipe 伪3D) |
  | S | 保存增益 |
  | Q/ESC | 退出 |
- 参数: `--drive` `--camera` `--hand` `--skip N` `--no-display`
- HUD 顶部显示 3D 源状态: `HAMER 3D` / `MP FALLBACK`

## 6. 错误处理与回退

| 场景 | 行为 |
|------|------|
| `import hamer` 失败 / 无 CUDA | `HaMeR3D.available=False` → demo 纯 MediaPipe (等同 demo_realtime), 打印警告 |
| 单帧 kp3d 非 finite | 该帧回退 MediaPipe 伪 3D |
| MediaPipe 无手 | 无 bbox → 跳过 hamer → `leap.set_open()` |
| bbox 太小/越界 | 跳过 hamer, 该帧回退 MediaPipe |
| hamer 推理异常 (显存不足等) | try/except → 回退 MediaPipe, 累计计数告警 |

## 7. 验证计划

1. **环境**: hamer env pip 装 4 包 → 验证 import + 全栈 import 冒烟
2. **静态回归**: `example_data/test1.jpg` + 一组倾斜/旋转/攥紧图 → 断言 kp3d finite、(21,3) 形状、顺序与 MediaPipe 索引对齐（逐点距离对比）
3. **离线角度对比**: 同帧下 MediaPipe vs hamer flexion 角度——旋转/攥紧时 hamer 应稳定、MediaPipe 漂移（量化记录）
4. **实机**: 先 no-drive 观察叠加跟手（倾斜/旋转/攥紧）→ 再 `--drive` 驱动真手, 校验 JOINT_DIR 方向与力反馈

## 8. 风险与缓解

| 风险 | 缓解 |
|------|------|
| mediapipe pip 进 hamer env 的 protobuf/absl 冲突 | 装后立即验证; 必要时锁版本 |
| 镜像翻转 → MediaPipe handedness 标签翻转 | 默认取 results[0]; 实机验证后加 `--hand` |
| fps ~20-25 (hamer ~30ms) | 可接受; 可选 `--skip N`; OneEuroFilter 已平滑 |
| 拇指 CMC 缺失 | 用拇指 mcp 作链底, 链式有效 |
| 基线尺度混用 | Calibrator points 版与 MediaPipe 版严格分离 |

## 9. 明确不做（YAGNI）

- 不做 detectron2/ViTPose 人员检测（MediaPipe bbox 足够）
- 不做双进程 IPC
- 不做 MANO hand_pose → LEAP 直接角度 (方案 B)
- 不做 pyrender 全网格渲染（本步用 kp3d 投影 + 可选线框）
- 不做 GPU 推理与真机驱动的跨进程隔离
