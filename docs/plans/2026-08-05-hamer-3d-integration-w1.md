# HaMeR 3D MANO 集成 W1 — 实施计划

> 设计文档: `docs/design/2026-08-05-hamer-3d-integration-w1.md`（已提交 f86dd88）
> 目标: 用 hamer 的 MANO 真 3D 关键点替代 MediaPipe 伪 3D z，解决"手倾斜/旋转/攥紧时 3D 失效"

---

**For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 hamer 3D MANO 回归模块并接入 W1 手势映射管线（MediaPipe bbox → hamer kp3d → JointMapper），解决倾斜/旋转/攥紧时 3D 失效，支持驱动真手。

**Architecture:** MediaPipe 始终提供 21 关键点（bbox + 回退源）→ 由关键点算方形 bbox → ViTDetDataset 裁剪后喂 hamer（fp16+autocast）→ 得到 MANO kp3d(21,3)（顺序与 MediaPipe 索引 1:1）→ 复用 JointMapper 核心计算角度 → 1€ 滤波 → 驱动。hamer 失败时逐帧回退 MediaPipe 伪 3D。

**Tech Stack:** Python 3.10 (conda env `hamer`), torch 2.13+cu130, hamer, mediapipe 1.0.0, detectron2 (仅 ViTDetDataset 裁剪用), OpenCV, numpy, dynamixel_sdk, pyrealsense2, pytest。

## Global Constraints

- 所有步骤在 `hamer` conda 环境执行（`conda activate hamer`）；**只用 pip 装包，禁用 conda install**（该环境已有 pip 安装的 torch）。
- hamer kp3d 顺序与 MediaPipe 索引 **1:1 兼容，禁止重排**（`hamer/models/mano_wrapper.py` joint_map 已确认）。
- **纯加法改动**：不得改变现有 `demo_realtime.py` / `map_keypoints_to_leap` 行为；新路径全部新增。
- hamer 前向必须 `torch.no_grad() + torch.autocast(fp16)`；模型 fp16 加载（~1.5GB VRAM，6GB 卡可运行）。
- 跳过 detectron2/ViTPose 人员检测，bbox 一律来自 MediaPipe 关键点。
- 实现文件遵循 black + isort；项目包结构 `python/gesture_mapping/`。
- 实机驱动前必须备份（项目 CLAUDE.md 第九节：改动 2+ 文件先备份到 `python/backups/`）。

---

### Task 1: 环境准备（hamer env 依赖 + 设计文档签名对齐）

**Files:**
- 修改: `docs/design/2026-08-05-hamer-3d-integration-w1.md`（§5.1 签名修正，见 Step 3）

**Interfaces:**
- Produces: `hamer` env 具备 mediapipe/dynamixel_sdk/pyrealsense2/pyserial/pytest；设计文档 §5.1 更新为 `regress(frame_bgr, bbox_xyxy)`（裁剪由 ViTDetDataset 内部完成）。

- [ ] **Step 1: 在 hamer env 安装缺失依赖**

```bash
conda activate hamer
pip install mediapipe==1.0.0 dynamixel-sdk pyrealsense2 pyserial pytest
```

- [ ] **Step 2: 验证全栈 import（含 torch/hamer 未被破坏）**

```bash
cd /home/bright/office/Leap_Hand/python
python -c "import mediapipe, cv2, numpy, torch, dynamixel_sdk, pyrealsense2, serial, pytest, hamer; print('ALL IMPORTS OK')"
```

Expected: `ALL IMPORTS OK`

若 mediapipe 破坏了 hamer（numpy/protobuf 冲突，出现 `import hamer` 失败），回退：`pip install "numpy==2.2.6" "protobuf>=4"`，再重跑本步；仍不行则记录版本并暂停回报。

- [ ] **Step 3: 设计文档签名对齐**

在 `docs/design/2026-08-05-hamer-3d-integration-w1.md` §5.1 中，把 `regress(self, crop_bgr)` 改为 `regress(self, frame_bgr, bbox_xyxy)`，并在该方法上方加一行注释：`# 裁剪/缩放由 hamer 的 ViTDetDataset 内部完成 (镜像 smoke_test 路径); bbox_xyxy 为全帧像素坐标`。同时 §4 数据流中 `HaMeR3D.regress(crop)` 改为 `HaMeR3D.regress(frame, bbox)`。

- [ ] **Step 4: 提交**

```bash
cd /home/bright/office/Leap_Hand
git add docs/design/2026-08-05-hamer-3d-integration-w1.md
git commit -m "docs(design): refine HaMeR3D.regress to take full frame + bbox (ViTDetDataset crops internally)"
```

---

### Task 2: JointMapper 增补 3D 点云路径（TDD）

**Files:**
- Create: `python/tests/test_joint_mapper.py`
- Modify: `python/gesture_mapping/joint_mapper.py`

**Interfaces:**
- Produces:
  - `JointMapper.map_points_to_leap(pts: np.ndarray(21,3)) -> np.ndarray(16)`（米制/任意尺度均可，角度尺度不变）
  - `JointMapper.map_points_to_leap_dict(pts) -> Dict[str, Dict[str,float]]`
  - `map_keypoints_to_leap(hand_result, image_shape)` 重构为委托 `map_points_to_leap`
  - `map_keypoints_to_leap_dict(hand_result, image_shape)` 重构为委托 `map_points_to_leap_dict`

- [ ] **Step 1: 备份现有代码**

```bash
mkdir -p /home/bright/office/Leap_Hand/python/backups/2026-08-05_hamer3d
cp /home/bright/office/Leap_Hand/python/gesture_mapping/joint_mapper.py \
   /home/bright/office/Leap_Hand/python/gesture_mapping/calibrator.py \
   /home/bright/office/Leap_Hand/python/backups/2026-08-05_hamer3d/
```

- [ ] **Step 2: 写失败测试** — 创建 `python/tests/test_joint_mapper.py`：

```python
"""Tests for JointMapper.map_points_to_leap (3D point-cloud path)."""
import numpy as np
import pytest
from types import SimpleNamespace

from gesture_mapping import JointMapper
from gesture_mapping.hand_tracker import HandResult


# ─── Synthetic hand builders ──────────────────────────────────────
# A flat open hand: wrist at origin, fingers as straight rays → all flexion ≈ 0.

_JCHAIN = {
    "thumb":  (1, 2, 3, 4),
    "index":  (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring":   (13, 14, 15, 16),
    "pinky":  (17, 18, 19, 20),
}
_JJOINT = {"mcp": 0, "pip": 1, "dip": 2}
_SEG = {  # per-finger segment lengths [wrist→MCP, MCP→PIP, PIP→DIP, DIP→TIP]
    "thumb":  [0.45, 0.35, 0.30, 0.30],
    "index":  [0.50, 0.50, 0.40, 0.40],
    "middle": [0.50, 0.50, 0.40, 0.40],
    "ring":   [0.50, 0.50, 0.40, 0.40],
    "pinky":  [0.50, 0.50, 0.40, 0.40],
}
_DIRS = {
    "thumb":  np.array([0.95, 0.30, 0.0]),
    "index":  np.array([0.35, 0.94, 0.0]),
    "middle": np.array([0.12, 0.99, 0.0]),
    "ring":   np.array([-0.10, 0.99, 0.0]),
    "pinky":  np.array([-0.30, 0.95, 0.0]),
}


def _open_hand_pts() -> np.ndarray:
    pts = np.zeros((21, 3), dtype=np.float64)
    for name, chain in _JCHAIN.items():
        d = _DIRS[name] / np.linalg.norm(_DIRS[name])
        seg = _SEG[name]
        for j, idx in enumerate(chain):
            pts[idx] = d * sum(seg[:j + 1])
    return pts


def _rot(p, pivot, axis, theta):
    v = p - pivot
    k = axis / np.linalg.norm(axis)
    c, s = np.cos(theta), np.sin(theta)
    return pivot + v * c + np.cross(k, v) * s + k * np.dot(k, v) * (1 - c)


def _bend_joint(pts, finger, joint, theta) -> np.ndarray:
    """Rotate the part of `finger` distal to `joint` by theta out of the palm plane."""
    ch = _JCHAIN[finger]
    j = _JJOINT[joint]
    pivot = pts[ch[j]]
    proximal = pts[0] if j == 0 else pts[ch[j - 1]]
    axis = np.cross(pts[ch[j]] - proximal, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return pts.copy()
    axis = axis / n
    out = pts.copy()
    for k in range(j + 1, len(ch)):
        out[ch[k]] = _rot(pts[ch[k]], pivot, axis, theta)
    return out


def _synthetic_hand_result(pts):
    lms = [SimpleNamespace(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in pts]
    return HandResult(landmarks=lms, handedness="Right")


# ─── Tests ────────────────────────────────────────────────────────

def test_open_hand_has_low_flexion():
    mapper = JointMapper()
    a = mapper.map_points_to_leap(_open_hand_pts())
    flexion_ids = [1, 2, 3, 5, 6, 7, 9, 10, 11]  # index/middle/pinky mcp+pip+dip
    assert np.all(np.abs(a[flexion_ids]) < 0.05)


def test_bending_mcp_increases_index_mcp_only():
    mapper = JointMapper()
    open_a = mapper.map_points_to_leap(_open_hand_pts())
    bent = _bend_joint(_open_hand_pts(), "index", "mcp", 0.8)
    a = mapper.map_points_to_leap(bent)
    assert a[1] > 0.5        # index mcp
    assert abs(a[2]) < 0.05  # index pip unchanged
    assert abs(a[3]) < 0.05  # index dip unchanged


def test_bending_pip_increases_index_pip_only():
    mapper = JointMapper()
    open_a = mapper.map_points_to_leap(_open_hand_pts())
    bent = _bend_joint(_open_hand_pts(), "index", "pip", 0.8)
    a = mapper.map_points_to_leap(bent)
    assert a[2] > 0.5
    assert abs(a[1]) < 0.05
    assert abs(a[3]) < 0.05


def test_bending_dip_increases_index_dip_only():
    mapper = JointMapper()
    open_a = mapper.map_points_to_leap(_open_hand_pts())
    bent = _bend_joint(_open_hand_pts(), "index", "dip", 0.8)
    a = mapper.map_points_to_leap(bent)
    assert a[3] > 0.5
    assert abs(a[1]) < 0.05
    assert abs(a[2]) < 0.05


def test_map_keypoints_delegates_to_map_points():
    """Refactor guard: both entry points must agree on the same point cloud."""
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, size=(21, 3))
    hr = _synthetic_hand_result(pts)
    mapper = JointMapper()
    assert np.allclose(
        mapper.map_keypoints_to_leap(hr, (1, 1)),
        mapper.map_points_to_leap(pts),
    )


def test_map_points_to_leap_dict_roundtrip():
    mapper = JointMapper()
    pts = _open_hand_pts()
    arr = mapper.map_points_to_leap(pts)
    d = mapper.map_points_to_leap_dict(pts)
    assert set(d.keys()) == {"index", "middle", "pinky", "thumb"}
    # standard fingers: group order [abd, mcp, pip, dip]
    for i, name in enumerate(["index", "middle", "pinky"]):
        assert d[name]["abduction"] == pytest.approx(arr[i * 4 + 0])
        assert d[name]["mcp"] == pytest.approx(arr[i * 4 + 1])
        assert d[name]["pip"] == pytest.approx(arr[i * 4 + 2])
        assert d[name]["dip"] == pytest.approx(arr[i * 4 + 3])
    # thumb physical order differs: [mcp, abd, pip, dip]
    assert d["thumb"]["mcp"] == pytest.approx(arr[12])
    assert d["thumb"]["abduction"] == pytest.approx(arr[13])
    assert d["thumb"]["pip"] == pytest.approx(arr[14])
    assert d["thumb"]["dip"] == pytest.approx(arr[15])


def test_map_points_rejects_bad_shape():
    mapper = JointMapper()
    with pytest.raises(ValueError):
        mapper.map_points_to_leap(np.zeros((20, 3)))
```

- [ ] **Step 3: 跑测试确认失败**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_joint_mapper.py -v
```

Expected: FAIL（`AttributeError: 'JointMapper' object has no attribute 'map_points_to_leap'`）

- [ ] **Step 4: 实现 `map_points_to_leap` 核心**

在 `python/gesture_mapping/joint_mapper.py` 中：

1) 把 `map_keypoints_to_leap` 的方法体（原第 123-180 行）替换为委托，并在其后新增两个方法。改后 `map_keypoints_to_leap` 为：

```python
    def map_keypoints_to_leap(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Convert one detected hand to LEAP 16-DOF relative joint angles.

        Delegates to map_points_to_leap() after building the MediaPipe
        (21,3) point cloud.
        """
        pts = self._build_point_cloud(hand_result, image_shape)
        return self.map_points_to_leap(pts)
```

2) 新增（放在 `map_keypoints_to_leap` 之后）：

```python
    def map_points_to_leap(self, pts: np.ndarray) -> np.ndarray:
        """Map a (21,3) point cloud to LEAP 16-DOF relative joint angles.

        Accepts any metric/normalized coordinate frame — angle computation is
        scale-invariant. This is the hamer 3D entry point (real MANO kp3d,
        MediaPipe-index order) and the shared core for the MediaPipe path.
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (_NUM_LANDMARKS, 3):
            raise ValueError(
                f"expected a ({_NUM_LANDMARKS}, 3) point cloud, got {pts.shape}"
            )

        wrist_pt, palm_normal, mid_dir, lateral = self._palm_frame(pts)
        idx_dir = self._plane_dir(pts[Landmark["INDEX_MCP"]],
                                  pts[Landmark["INDEX_PIP"]], palm_normal)
        if np.linalg.norm(idx_dir) < 1e-9:
            idx_dir = mid_dir

        angles = np.zeros(_NUM_LEAP_DOF, dtype=np.float64)

        for human_finger, leap_start in _FINGER_MAP:
            chain = _FINGER_CHAIN[human_finger]
            kps = pts[chain]

            ref_dir = idx_dir if human_finger == "thumb" else mid_dir
            fan = self._compute_fan_angle(kps[0], kps[1], ref_dir, lateral,
                                          palm_normal)
            fan *= _FAN_SIGN.get(human_finger, 1.0)

            mcp = self._compute_flexion(wrist_pt, kps[0], kps[1])
            pip = self._compute_flexion(kps[0], kps[1], kps[2])
            dip = self._compute_flexion(kps[1], kps[2], kps[3])

            rel = (fan, mcp, pip, dip)
            order = (_THUMB_JOINT_ORDER if human_finger == "thumb"
                     else _STANDARD_JOINT_ORDER)
            for k, j in enumerate(order):
                angles[leap_start + k] = rel[j]

        return np.clip(angles * self.joint_gain, _ANGLE_MIN, _ANGLE_MAX)

    def map_points_to_leap_dict(
        self,
        pts: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Like map_points_to_leap() but returns a labeled dict."""
        angles = self.map_points_to_leap(pts)
        result = {}
        for i, name in enumerate(_OUTPUT_FINGER_KEYS):
            start = i * 4
            keys = _JOINT_KEYS_THUMB if name == "thumb" else _JOINT_KEYS_STANDARD
            result[name] = {k: float(angles[start + j]) for j, k in enumerate(keys)}
        return result
```

3) `map_keypoints_to_leap_dict` 方法体（原第 182-198 行）替换为：

```python
    def map_keypoints_to_leap_dict(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Like map_keypoints_to_leap() but returns a labeled dict."""
        pts = self._build_point_cloud(hand_result, image_shape)
        return self.map_points_to_leap_dict(pts)
```

- [ ] **Step 5: 跑测试确认通过**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_joint_mapper.py -v
```

Expected: 7 passed

- [ ] **Step 6: 提交**

```bash
cd /home/bright/office/Leap_Hand
git add python/gesture_mapping/joint_mapper.py python/tests/test_joint_mapper.py
git commit -m "feat(mapping): add map_points_to_leap 3D point-cloud path to JointMapper"
```

---

### Task 3: Calibrator + FingerIdentifier 3D 点云版（TDD）

**Files:**
- Create: `python/tests/test_calibrator.py`
- Modify: `python/gesture_mapping/calibrator.py`

**Interfaces:**
- Consumes: `JointMapper.map_points_to_leap` / `map_points_to_leap_dict`（Task 2）
- Produces:
  - `Calibrator.calibrate_points(pts) -> np.ndarray(16)` / `Calibrator.map_points(pts) -> np.ndarray(16)`
  - `FingerIdentifier.identify_points(pts) -> Tuple[Optional[str], Dict[str,float]]`

- [ ] **Step 1: 写失败测试** — 创建 `python/tests/test_calibrator.py`：

```python
"""Tests for Calibrator/FingerIdentifier 3D point-cloud entry points."""
import numpy as np

from gesture_mapping import JointMapper, Calibrator, FingerIdentifier
from test_joint_mapper import _open_hand_pts, _bend_joint


def test_calibrator_points_open_is_zero():
    mapper = JointMapper()
    cal = Calibrator(mapper)
    open_pts = _open_hand_pts()
    cal.calibrate_points(open_pts)
    assert cal.is_calibrated
    assert np.allclose(cal.map_points(open_pts), 0.0, atol=1e-9)


def test_calibrator_points_isolates_bend():
    mapper = JointMapper()
    cal = Calibrator(mapper)
    open_pts = _open_hand_pts()
    cal.calibrate_points(open_pts)
    bent = _bend_joint(open_pts, "index", "mcp", 0.9)
    a = cal.map_points(bent)
    assert a[1] > 0.5        # index mcp bends away from baseline
    assert abs(a[5]) < 0.05  # middle unaffected


def test_identify_points_open_is_none():
    mapper = JointMapper()
    fi = FingerIdentifier(mapper, bend_threshold=0.20)
    bent, scores = fi.identify_points(_open_hand_pts())
    assert bent is None


def test_identify_points_index_bent():
    mapper = JointMapper()
    fi = FingerIdentifier(mapper, bend_threshold=0.20)
    pts = _bend_joint(_open_hand_pts(), "index", "mcp", 0.9)
    pts = _bend_joint(pts, "index", "pip", 0.6)
    pts = _bend_joint(pts, "index", "dip", 0.4)
    bent, scores = fi.identify_points(pts)
    assert bent == "index"
    assert scores["index"] >= 0.20
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_calibrator.py -v
```

Expected: FAIL（`AttributeError: 'Calibrator' object has no attribute 'calibrate_points'`）

- [ ] **Step 3: 实现** — 在 `python/gesture_mapping/calibrator.py` 中新增：

在 `Calibrator.calibrate` 方法之后追加：

```python
    def calibrate_points(self, pts: np.ndarray) -> np.ndarray:
        """Record current 3D point cloud as the zero reference (call with hand fully open)."""
        self._baseline = self.mapper.map_points_to_leap(pts)
        self._calibrated = True
        return self._baseline

    def map_points(self, pts: np.ndarray) -> np.ndarray:
        """Zero-corrected joint angles from a (21,3) point cloud (hamer path)."""
        raw = self.mapper.map_points_to_leap(pts)
        if self._calibrated and self._baseline is not None:
            return np.clip(raw - self._baseline, -0.3, 2.8)
        return raw
```

在 `FingerIdentifier.identify` 方法之后追加：

```python
    def identify_points(
        self,
        pts: np.ndarray,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Identify most bent finger from a (21,3) point cloud (hamer path)."""
        d = self.mapper.map_points_to_leap_dict(pts)
        scores = {}
        for fname, joints in d.items():
            scores[fname] = (
                joints["mcp"] * 0.3 +
                joints["pip"] * 0.5 +
                joints["dip"] * 0.2
            )
        max_finger = max(scores, key=scores.get)
        return (
            (max_finger, scores) if scores[max_finger] >= self.threshold
            else (None, scores)
        )
```

- [ ] **Step 4: 跑测试确认通过**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_calibrator.py -v
```

Expected: 4 passed

- [ ] **Step 5: 回归 Task 2 测试**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/ -v
```

Expected: 11 passed

- [ ] **Step 6: 提交**

```bash
cd /home/bright/office/Leap_Hand
git add python/gesture_mapping/calibrator.py python/tests/test_calibrator.py
git commit -m "feat(calib): add points-based calibrate/map and finger identify (hamer path)"
```

---

### Task 4: hamer_3d.py 模块（纯函数 TDD + 类离线冒烟）

**Files:**
- Create: `python/gesture_mapping/hamer_3d.py`
- Create: `python/tests/test_hamer_3d.py`
- Create: `python/tests/test_hamer3d_offline.py`

**Interfaces:**
- Produces:
  - `hand_bbox_from_landmarks(pts_xy, image_shape, margin=1.5, square=True, min_size=32) -> Optional[Tuple[int,int,int,int]]`
  - `to_mediapipe_order(kp3d) -> np.ndarray`（恒等，显式声明 1:1）
  - `HaMeR3D(checkpoint=None, device=None, fp16=True)`，属性 `available: bool`
  - `HaMeR3D.regress(frame_bgr, bbox_xyxy) -> Optional[HaMeR3DResult]`
  - `HaMeR3D.project_to_frame(result, pts3d) -> np.ndarray(N,2)`（全帧像素）
  - `HaMeR3DResult` 字段: `kp3d(21,3)` / `verts(778,3)` / `cam_t(3,)` / `kp2d_patch(21,2)` / `box_center(2,)` / `box_size(float)`

- [ ] **Step 1: 写失败测试（纯函数）** — 创建 `python/tests/test_hamer_3d.py`：

```python
"""Tests for hamer_3d pure helpers (bbox + order mapping)."""
import numpy as np

from gesture_mapping.hamer_3d import hand_bbox_from_landmarks, to_mediapipe_order


def _spread_pts():
    # 21 points spanning x:[100,320] y:[100,340] in a 400x400 frame
    xs = np.linspace(100, 320, 21)
    ys = np.linspace(100, 340, 21)
    return np.stack([xs, ys], axis=1)


def test_to_mediapipe_order_is_identity():
    pts = np.arange(63, dtype=np.float64).reshape(21, 3)
    out = to_mediapipe_order(pts)
    assert out.dtype == np.float64
    assert np.array_equal(out, pts)


def test_bbox_square_and_clamped():
    pts = _spread_pts()
    bbox = hand_bbox_from_landmarks(pts, (400, 400), margin=1.5, square=True)
    assert bbox is not None
    x0, y0, x1, y1 = bbox
    assert x0 >= 0 and y0 >= 0 and x1 <= 400 and y1 <= 400
    assert (x1 - x0) == (y1 - y0)                       # square
    assert x0 <= pts[:, 0].min() and x1 >= pts[:, 0].max()
    assert y0 <= pts[:, 1].min() and y1 >= pts[:, 1].max()


def test_bbox_margin_expands():
    pts = _spread_pts()
    bbox1 = hand_bbox_from_landmarks(pts, (400, 400), margin=1.0, square=True)
    bbox2 = hand_bbox_from_landmarks(pts, (400, 400), margin=2.0, square=True)
    assert (bbox2[2] - bbox2[0]) > (bbox1[2] - bbox1[0])


def test_bbox_too_small_returns_none():
    pts = np.full((21, 2), 50.0)
    assert hand_bbox_from_landmarks(pts, (100, 100), margin=1.5) is None
```

- [ ] **Step 2: 跑测试确认失败**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_hamer_3d.py -v
```

Expected: FAIL（`ModuleNotFoundError: No module named 'gesture_mapping.hamer_3d'`）

- [ ] **Step 3: 写离线集成测试（GPU/hamer 才跑）** — 创建 `python/tests/test_hamer3d_offline.py`：

```python
"""Offline integration: MediaPipe bbox → hamer regress → projection alignment.

Skips when hamer/GPU/mediapipe model are unavailable.
Validates the projection formula empirically: projected MANO kp must land
near MediaPipe kp (median < 60px). If it fails, check box_center/box_size
convention in project_to_frame.
"""
import numpy as np
import pytest

from gesture_mapping import HandTracker
from gesture_mapping.hamer_3d import HaMeR3D, hand_bbox_from_landmarks

HAMER_IMG = "/home/bright/office/hamer/example_data/test1.jpg"


def test_regress_and_projection_alignment():
    try:
        h3d = HaMeR3D()
    except Exception:
        pytest.skip("hamer init raised")
    if not h3d.available:
        pytest.skip("no GPU / hamer unavailable")

    import cv2
    img = cv2.imread(HAMER_IMG)
    if img is None:
        pytest.skip(f"missing test image: {HAMER_IMG}")
    h, w = img.shape[:2]

    tracker = HandTracker(max_num_hands=1)
    results = tracker.detect(img)
    if not results:
        pytest.skip("MediaPipe found no hand in test image")
    hand = results[0]
    mp_pts = tracker.landmark_xy(hand, (h, w))
    bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
    assert bbox is not None

    hres = h3d.regress(img, bbox)
    assert hres is not None, "hamer regression returned None"
    assert hres.kp3d.shape == (21, 3)
    assert np.isfinite(hres.kp3d).all()
    assert hres.verts.shape == (778, 3)

    proj = h3d.project_to_frame(hres, hres.kp3d)
    dist = np.linalg.norm(proj - mp_pts, axis=1)
    assert np.median(dist) < 60.0, f"projection misaligned, median={np.median(dist):.1f}px"

    tracker.close()
```

- [ ] **Step 4: 实现 `hamer_3d.py`** — 创建文件，完整内容：

```python
"""HaMeR 3D MANO regression → MediaPipe-indexed kp3d for LEAP Hand W1.

Wraps hamer (fp16) so a MediaPipe-detected hand crop is turned into a real
MANO 3D hand model: 21 keypoints (order 1:1 with MediaPipe) + 778 verts.

Design: docs/design/2026-08-05-hamer-3d-integration-w1.md
"""

import io
import contextlib
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class HaMeR3DResult:
    """Container for one hamer regression output (single hand)."""

    def __init__(self, kp3d, verts, cam_t, kp2d_patch, box_center, box_size):
        self.kp3d = np.asarray(kp3d, dtype=np.float64)          # (21,3) metric, MP order
        self.verts = np.asarray(verts, dtype=np.float64)        # (778,3) metric
        self.cam_t = np.asarray(cam_t, dtype=np.float64)        # (3,) weak-persp translation
        self.kp2d_patch = np.asarray(kp2d_patch, dtype=np.float64)  # (21,2) patch pixels
        self.box_center = np.asarray(box_center, dtype=np.float64)  # (2,) full-frame px
        self.box_size = float(box_size)                          # full-frame px extent


class HaMeR3D:
    """Lazy-loaded hamer (fp16) regression from a full frame + hand bbox.

    available is False when torch/hamer are missing or no CUDA GPU — callers
    must fall back to MediaPipe pseudo-3D in that case.
    """

    def __init__(self, checkpoint: Optional[str] = None,
                 device: Optional[str] = None, fp16: bool = True):
        self.available = False
        self.fp16 = fp16
        self.model = None
        self.model_cfg = None
        self.image_size = None
        self.device = None
        try:
            import torch
        except Exception:
            return
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type != "cuda":
            return
        try:
            from hamer.configs import get_config
            from hamer.models import HAMER, DEFAULT_CHECKPOINT
        except Exception:
            return
        try:
            self._load(checkpoint or DEFAULT_CHECKPOINT, get_config, HAMER)
            self.available = True
        except Exception:
            self.model = None

    def _load(self, checkpoint, get_config, HAMER):
        ckpt_path = str(Path(checkpoint))
        model_cfg = get_config(str(Path(ckpt_path).parent.parent / "model_config.yaml"),
                               update_cachedir=True)
        if model_cfg.MODEL.BACKBONE.TYPE == "vit" and "BBOX_SHAPE" not in model_cfg.MODEL:
            model_cfg.defrost()
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            model_cfg.freeze()
        model = HAMER.load_from_checkpoint(ckpt_path, strict=False, cfg=model_cfg,
                                           map_location="cpu")
        self.model = model.half().to(self.device) if self.fp16 else model.to(self.device)
        self.model.eval()
        self.model_cfg = model_cfg
        self.image_size = int(model_cfg.MODEL.IMAGE_SIZE)

    def regress(self, frame_bgr: np.ndarray,
                bbox_xyxy: Tuple[int, int, int, int]) -> Optional[HaMeR3DResult]:
        """Run hamer on a hand crop. Returns None on failure.

        Cropping/resizing is handled inside by hamer's ViTDetDataset (the same
        path smoke_test uses). bbox_xyxy is full-frame pixel coords.
        """
        if not self.available:
            return None
        import torch
        try:
            from hamer.datasets.vitdet_dataset import ViTDetDataset
            from hamer.utils import recursive_to

            boxes = np.array([bbox_xyxy], dtype=np.float32)
            right = np.array([1], dtype=np.int32)  # LEAP is a right hand
            dataset = ViTDetDataset(self.model_cfg, frame_bgr, boxes, right,
                                    rescale_factor=2.0)
            loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0)
            with torch.no_grad(), torch.autocast(device_type="cuda",
                                                 dtype=torch.float16):
                # ViTDetDataset prints a debug line every item — suppress it.
                with contextlib.redirect_stdout(io.StringIO()):
                    batch = recursive_to(next(iter(loader)), self.device)
                if self.fp16:
                    for k, v in batch.items():
                        if torch.is_tensor(v) and v.is_floating_point():
                            batch[k] = v.half()
                out = self.model(batch)
        except Exception:
            return None

        kp3d = to_mediapipe_order(out["pred_keypoints_3d"][0].float().cpu().numpy())
        if not np.isfinite(kp3d).all():
            return None
        verts = out["pred_vertices"][0].float().cpu().numpy()
        cam_t = out["pred_cam_t"][0].float().cpu().numpy()
        kp2d_patch = out["pred_keypoints_2d"][0].float().cpu().numpy()
        box_center = batch["box_center"][0].float().cpu().numpy()
        box_size = float(batch["box_size"][0].item())
        return HaMeR3DResult(kp3d, verts, cam_t, kp2d_patch, box_center, box_size)

    def project_to_frame(self, result: HaMeR3DResult, pts3d: np.ndarray) -> np.ndarray:
        """Project 3D points (metric) to full-frame pixels via hamer's weak perspective.

        p_patch = (focal/IMAGE_SIZE) * (xy + cam_t.xy) / (z + cam_t.z), origin at
        patch top-left; then p_frame = box_center + (p_patch - IMAGE_SIZE/2) * box_size/IMAGE_SIZE.
        """
        pts = np.asarray(pts3d, dtype=np.float64)
        focal = float(self.model_cfg.EXTRA.FOCAL_LENGTH) / self.image_size
        z = pts[:, 2] + result.cam_t[2]
        p_patch = np.zeros((len(pts), 2), dtype=np.float64)
        nz = np.abs(z) > 1e-6
        p_patch[nz] = focal * (pts[nz, :2] + result.cam_t[:2]) / z[nz, None]
        s = self.image_size
        return result.box_center + (p_patch - s / 2) * (result.box_size / s)


def to_mediapipe_order(kp3d: np.ndarray) -> np.ndarray:
    """Return kp3d as-is.

    hamer's MANO joint order (wrist, thumb mcp/pip/dip/tip, index, middle,
    ring, pinky) is positionally 1:1 with MediaPipe indices, so no reordering
    is required. Kept explicit so a future MANO change is patched here only.
    """
    return np.asarray(kp3d, dtype=np.float64)


def hand_bbox_from_landmarks(
    pts_xy: np.ndarray,
    image_shape: Tuple[int, int],
    margin: float = 1.5,
    square: bool = True,
    min_size: int = 32,
) -> Optional[Tuple[int, int, int, int]]:
    """Square crop around a hand's 21 landmarks (pixel xy).

    Returns (x0, y0, x1, y1) clamped to the frame, or None if degenerate.
    """
    h, w = image_shape
    xs, ys = pts_xy[:, 0], pts_xy[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())

    if square:
        side = max(x1 - x0, y1 - y0) * margin
        if side < min_size:
            return None
        half = side / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, y0, x1, y1 = cx - half, cy - half, cx + half, cy + half
    else:
        padx = (x1 - x0) * (margin - 1) / 2
        pady = (y1 - y0) * (margin - 1) / 2
        x0, y0, x1, y1 = x0 - padx, y0 - pady, x1 + padx, y1 + pady

    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return (x0, y0, x1, y1)


# ─── Demo / Test ─────────────────────────────────────────────────

def smoke_on_image(image_path: str):
    """Run MediaPipe → hamer on a single image and print kp3d stats."""
    import cv2
    from gesture_mapping import HandTracker

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] cannot read {image_path}")
        return
    h, w = img.shape[:2]
    tracker = HandTracker(max_num_hands=1)
    results = tracker.detect(img)
    if not results:
        print("  (no hand detected)")
        return
    hand = results[0]
    mp_pts = tracker.landmark_xy(hand, (h, w))
    bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
    if bbox is None:
        print("  (bbox too small)")
        return
    h3d = HaMeR3D()
    if not h3d.available:
        print("  (hamer unavailable)")
        return
    res = h3d.regress(img, bbox)
    if res is None:
        print("  (hamer regression failed)")
        return
    print(f"  kp3d shape={res.kp3d.shape} finite={bool(np.isfinite(res.kp3d).all())}")
    print(f"  verts shape={res.verts.shape}")
    print(f"  kp3d range=[{res.kp3d.min():.3f}, {res.kp3d.max():.3f}] (meters)")
    proj = h3d.project_to_frame(res, res.kp3d)
    dist = np.linalg.norm(proj - mp_pts, axis=1)
    print(f"  projection median dist vs MediaPipe: {np.median(dist):.1f} px")
    tracker.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        smoke_on_image(sys.argv[1])
    else:
        print("usage: python hamer_3d.py <image_path>")
```

- [ ] **Step 5: 跑纯函数测试确认通过**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_hamer_3d.py -v
```

Expected: 4 passed

- [ ] **Step 6: 跑离线集成测试（GPU）**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_hamer3d_offline.py -v
```

Expected: 1 passed（若 hamer 加载失败/无 GPU → SKIPPED 视为通过）

若断言 `median dist < 60.0` 失败：说明投影公式的 `box_center/box_size` 约定需校正。排查顺序：(a) 打印 `res.box_center/box_size` 与手部像素范围比对；(b) 若整体偏移一个 `IMAGE_SIZE/2`，把 `project_to_frame` 中的 `- s / 2` 改为 `+ s / 2` 或去掉，重跑。

- [ ] **Step 7: 跑全量测试确认无回归**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/ -v
```

Expected: 全部通过（GPU 项跑或跳过）

- [ ] **Step 8: 提交**

```bash
cd /home/bright/office/Leap_Hand
git add python/gesture_mapping/hamer_3d.py python/tests/test_hamer_3d.py python/tests/test_hamer3d_offline.py
git commit -m "feat(3d): add HaMeR3D module (fp16 MANO regression + projection)"
```

---

### Task 5: demo_hamer3d.py 实时 demo

**Files:**
- Create: `python/gesture_mapping/demo_hamer3d.py`

**Interfaces:**
- Consumes:
  - `HandTracker` / `JointMapper` / `Calibrator` / `FingerIdentifier` / `OneEuroFilter`（已有）
  - `HaMeR3D` / `hand_bbox_from_landmarks`（Task 4）
  - `find_best_camera` / `_OpenCVCamera` / `draw_hud` / `print_motor_mapping` / `_MOTOR_DIAG` / `print_angles_table`（`gesture_mapping.demo_realtime`）
  - `open_realsense`（`gesture_mapping.camera`）
- Produces: 可运行 demo；`--img` 模式离线输出叠加图（可测试交付物）

- [ ] **Step 1: 创建 `demo_hamer3d.py`** — 完整内容：

```python
#!/usr/bin/env python3
"""Real-time demo: Camera → MediaPipe → hamer 3D MANO → JointMapper → LEAP.

Solves "3D fails when the hand tilts/rotates/clenches": hamer gives real
MANO 3D keypoints instead of MediaPipe's pseudo-3D z.

Controls:
    SPACE — calibrate zero-point (hold hand fully open)
    D     — toggle MANO 3D diagnostic overlay
    M     — toggle 3D source (hamer / MediaPipe pseudo-3D)
    S     — save gains
    Q/ESC — quit

Design: docs/design/2026-08-05-hamer-3d-integration-w1.md
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesture_mapping import HandTracker, JointMapper, Calibrator, FingerIdentifier
from gesture_mapping.filter import OneEuroFilter
from gesture_mapping.camera import open_realsense
from gesture_mapping.hamer_3d import HaMeR3D, hand_bbox_from_landmarks
from gesture_mapping.demo_realtime import (
    _OpenCVCamera, find_best_camera, draw_hud, print_motor_mapping,
    print_angles_table,
)


# MANO skeleton connectivity (MediaPipe-index) for the 3D overlay
_KP_CONN = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def _select_hand(results, hand: str):
    if hand == "first" or len(results) == 1:
        return results[0]
    for r in results:
        if r.handedness.lower() == hand:
            return r
    return results[0]


def _draw_hamer_overlay(frame, h3d, hres, mp_pts):
    """Draw projected MANO kp skeleton (yellow) + MediaPipe kp (green) for diagnosis."""
    kp2d = h3d.project_to_frame(hres, hres.kp3d)
    for a, b in _KP_CONN:
        cv2.line(frame,
                 (int(round(kp2d[a][0])), int(round(kp2d[a][1]))),
                 (int(round(kp2d[b][0])), int(round(kp2d[b][1]))),
                 (0, 255, 255), 2)
    for (x, y) in kp2d:
        cv2.circle(frame, (int(round(x)), int(round(y))), 3, (0, 200, 255), -1)
    # MediaPipe kp (green) for direct comparison
    for i in range(21):
        cv2.circle(frame, (int(round(mp_pts[i][0])), int(round(mp_pts[i][1]))),
                   2, (0, 255, 0), -1)
    return frame


def run_image(path, tracker, h3d, mapper):
    """Single-image offline run: detect → regress → angles → save overlay."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] cannot read {path}")
        return
    h, w = frame.shape[:2]
    results = tracker.detect(frame)
    if not results:
        print("  (no hand detected)")
        return
    hand = results[0]
    mp_pts = tracker.landmark_xy(hand, (h, w))
    bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
    if bbox is None:
        print("  (bbox too small)")
        return
    hres = h3d.regress(frame, bbox)
    if hres is None:
        print("  (hamer regression failed)")
        return
    angles = mapper.map_points_to_leap(hres.kp3d)
    print(f"  kp3d finite={bool(np.isfinite(hres.kp3d).all())} "
          f"range=[{hres.kp3d.min():.3f}, {hres.kp3d.max():.3f}] (m)")
    print_angles_table(angles, None, {})
    out = tracker.draw_landmarks(frame, [hand])
    out = _draw_hamer_overlay(out, h3d, hres, mp_pts)
    out_path = Path(path).with_name(Path(path).stem + "_hamer3d.jpg")
    cv2.imwrite(str(out_path), out)
    print(f"  overlay saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera index (default: auto-detect)")
    parser.add_argument("--drive", action="store_true", help="Drive LEAP Hand hardware")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--skip", type=int, default=0,
                        help="run hamer every (skip+1) frames (0 = every frame)")
    parser.add_argument("--hand", type=str, default="first",
                        choices=["first", "right", "left"],
                        help="which MediaPipe hand to track")
    parser.add_argument("--img", type=str, default=None,
                        help="run on a single image and exit")
    args = parser.parse_args()

    tracker = HandTracker(max_num_hands=2, min_detection_confidence=0.5)
    h3d = HaMeR3D()
    if h3d.available:
        print("[INFO] HaMeR 3D ready (fp16, MANO regression)")
    else:
        print("[WARN] hamer unavailable (no CUDA / not installed) → MediaPipe pseudo-3D fallback")

    mapper = JointMapper()
    calibrator = Calibrator(mapper)
    finger_id = FingerIdentifier(mapper, bend_threshold=0.20)
    angle_filter = OneEuroFilter(n_joints=16, min_cutoff=1.0, beta=0.007)

    gain_path = Path(__file__).resolve().parent / "joint_gain.json"
    if gain_path.exists():
        mapper.load_gain_from(str(gain_path))

    JOINT_DIR = np.array([-1, -1, -1, -1, -1, -1, -1, -1,
                          -1, -1, -1, -1,  1, -1, -1, -1])

    leap = None
    if args.drive:
        from main import LeapNode, OPEN_POSE
        try:
            leap = LeapNode()
            print("[INFO] LEAP Hand connected.")
        except OSError as e:
            print(f"[WARN] Cannot connect: {e}")

    if args.img:
        run_image(args.img, tracker, h3d, mapper)
        tracker.close()
        if leap is not None:
            leap.disconnect()
        return

    cam = open_realsense()
    if cam is not None:
        print("[INFO] Using RealSense SDK color stream (pyrealsense2)")
    else:
        cam_idx = args.camera
        if cam_idx < 0:
            print("[INFO] Auto-detecting camera (OpenCV)...")
            cam_idx = find_best_camera()
            if cam_idx is None:
                print("[ERROR] No working camera found. Check USB connection.")
                tracker.close()
                return
            print(f"[INFO] Using camera index {cam_idx}")
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {cam_idx}")
            tracker.close()
            return
        cam = _OpenCVCamera(cap)

    print("\n" + "=" * 50)
    print("  LEAP Hand — hamer 3D Gesture Mapper")
    print("  SPACE=calib | D=diag | M=source | S=save | Q=quit")
    print("=" * 50)
    print_motor_mapping()

    print("[INFO] Warming up camera (RealSense auto-calibration ~3s)...")
    warm_t0 = time.time()
    while time.time() - warm_t0 < 3.0:
        cam.read()
    print("[INFO] Camera warm. Starting control loop.")

    frame_count = 0
    show_diag = False
    hamer_on = True

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)   # mirror
            h, w = frame.shape[:2]
            results = tracker.detect(frame)

            if results:
                hand = _select_hand(results, args.hand)
                mp_pts = tracker.landmark_xy(hand, (h, w))
                frame = tracker.draw_landmarks(frame, [hand])

                hres = None
                if h3d.available and hamer_on and frame_count % (args.skip + 1) == 0:
                    bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
                    if bbox is not None:
                        hres = h3d.regress(frame, bbox)

                if hres is not None:
                    pts = hres.kp3d
                    angles = calibrator.map_points(pts)
                    bent, scores = finger_id.identify_points(pts)
                    if show_diag:
                        frame = _draw_hamer_overlay(frame, h3d, hres, mp_pts)
                    source = "HAMER 3D"
                else:
                    angles = calibrator.map(hand, (h, w))
                    bent, scores = finger_id.identify(hand, (h, w))
                    source = "MP FALLBACK"

                angles = angle_filter(angles)

                if leap is not None:
                    from main import OPEN_POSE
                    leap.set_leap(OPEN_POSE + JOINT_DIR * angles)

                draw_hud(frame, angles, calibrator, bent, scores, show_diag)
                cv2.putText(frame, f"3D: {source}", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 255) if source == "HAMER 3D" else (0, 120, 255), 2)

                if frame_count % 20 == 0:
                    print_angles_table(angles, bent, scores)
            else:
                if frame_count % 30 == 0:
                    print("  (no hand detected)")
                if leap is not None:
                    leap.set_open()

            if not args.no_display:
                if frame_count == 0:
                    cv2.namedWindow("LEAP Hand — hamer 3D Mapper", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("LEAP Hand — hamer 3D Mapper", 960, 720)
                cv2.imshow("LEAP Hand — hamer 3D Mapper", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord(" "):
                    if results:
                        if hres is not None:
                            baseline = calibrator.calibrate_points(hres.kp3d)
                        else:
                            baseline = calibrator.calibrate(hand, (h, w))
                        angle_filter.reset()
                        print(f"\n  *** CALIBRATED! baseline max: {baseline.max():.3f} rad ***\n")
                elif key == ord("d"):
                    show_diag = not show_diag
                    print(f"\n  Diagnostic overlay: {'ON' if show_diag else 'OFF'}\n")
                elif key == ord("m"):
                    hamer_on = not hamer_on
                    print(f"\n  3D source: {'hamer' if hamer_on else 'MediaPipe pseudo-3D'}\n")
                elif key == ord("s"):
                    mapper.save_gain(str(gain_path))

            frame_count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        if leap is not None:
            leap.set_open()
            leap.disconnect()
        tracker.close()
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] Demo stopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证离线单图模式（无摄像头/真手可测）**

```bash
cd /home/bright/office/Leap_Hand/python
python gesture_mapping/demo_hamer3d.py --img /home/bright/office/hamer/example_data/test1.jpg
```

Expected: 打印 kp3d finite + 角度表 + 保存 `test1_hamer3d.jpg`（打开图确认黄色 MANO 骨架与绿色 MediaPipe 关键点对齐，且 3D 骨架贴合手形）

- [ ] **Step 3: 提交**

```bash
cd /home/bright/office/Leap_Hand
git add python/gesture_mapping/demo_hamer3d.py
git commit -m "feat(demo): add demo_hamer3d.py (MediaPipe + hamer 3D MANO pipeline)"
```

---

### Task 6: 离线角度对比验证（倾斜/旋转/攥紧）

**Files:**
- Create: `python/tests/test_compare_3d_sources.py`
- Create: `python/images/`（测试图输出目录，入 .gitignore）

**Interfaces:**
- Consumes: `HaMeR3D` / `hand_bbox_from_landmarks`（Task 4）、`JointMapper.map_points_to_leap`（Task 2）、`HandTracker`（已有）
- Produces: 一组对比数值 + 对比叠加图；证明 hamer 3D 在旋转/攥紧下角度稳定、MediaPipe 漂移

- [ ] **Step 1: 收集测试图像**

从 W1 采集的倾斜/旋转/攥紧照片中挑 3-5 张，放入 `python/images/`。若手头没有，先用 `python gesture_mapping/demo_realtime.py --no-drive` 摆拍保存几帧。命名含姿态标签，如 `images/tilt_01.jpg`、`images/rotate_01.jpg`、`images/fist_01.jpg`。

- [ ] **Step 2: 写对比测试** — 创建 `python/tests/test_compare_3d_sources.py`：

```python
"""Compare MediaPipe pseudo-3D vs hamer 3D flexion under tilt/rotate/clench.

For each test image: run both sources, print both angle vectors. Skips on
missing GPU/hamer. Saves a comparison overlay to python/images/<stem>_compare.jpg.
"""
import glob
import os

import numpy as np
import pytest

from gesture_mapping import HandTracker, JointMapper
from gesture_mapping.hamer_3d import HaMeR3D, hand_bbox_from_landmarks
from gesture_mapping.demo_hamer3d import _draw_hamer_overlay

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")


def test_hamer_and_mediapipe_angles_for_pose_images():
    h3d = HaMeR3D()
    if not h3d.available:
        pytest.skip("no GPU / hamer unavailable")
    import cv2

    tracker = HandTracker(max_num_hands=1)
    mapper = JointMapper()

    imgs = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not imgs:
        pytest.skip("no images in python/images/")

    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        results = tracker.detect(img)
        if not results:
            print(f"  {os.path.basename(path)}: no hand detected")
            continue
        hand = results[0]
        mp_pts = tracker.landmark_xy(hand, (h, w))
        bbox = hand_bbox_from_landmarks(mp_pts, (h, w))
        hres = h3d.regress(img, bbox) if bbox is not None else None

        mp_angles = mapper.map_keypoints_to_leap(hand, (h, w))
        h3_angles = mapper.map_points_to_leap(hres.kp3d) if hres is not None else np.zeros(16)

        print(f"\n  === {os.path.basename(path)} ===")
        print(f"  {'DOF':>4s} {'MediaPipe':>10s} {'hamer':>10s}")
        for i in range(16):
            print(f"  {i:>4d} {mp_angles[i]:>+10.3f} {h3_angles[i]:>+10.3f}")

        out = tracker.draw_landmarks(img, [hand])
        if hres is not None:
            out = _draw_hamer_overlay(out, h3d, hres, mp_pts)
        out_path = os.path.join(IMAGES_DIR, os.path.splitext(os.path.basename(path))[0] + "_compare.jpg")
        cv2.imwrite(out_path, out)
        print(f"  overlay: {out_path}")

    tracker.close()
```

- [ ] **Step 3: 运行并人工判定**

```bash
cd /home/bright/office/Leap_Hand/python
python -m pytest tests/test_compare_3d_sources.py -v -s
```

Expected: 每张图打印两列角度 + 生成 `images/<stem>_compare.jpg`。
人工判定标准（对照 W1 目标）：
- 攥紧（fist）图：hamer 的 mcp/pip/dip 列明显增大（>1.0 rad），MediaPipe 列应漂移或过小。
- 倾斜/旋转图：hamer 角度在相邻帧间连续稳定；MediaPipe 抖动。
- 若某姿态 hamer 输出非 finite 或明显错误（如整手反折），在测试图中检查 bbox 是否盖住整只手；必要时调 `hand_bbox_from_landmarks` 的 `margin`。

判定不通过则记录并进入 Task 7 前的调参（增益/边距），调参后重跑本任务。

- [ ] **Step 4: 提交**

```bash
cd /home/bright/office/Leap_Hand
git add python/tests/test_compare_3d_sources.py python/images/
git commit -m "test(3d): add MediaPipe vs hamer angle comparison under tilt/rotate/clench"
```

---

### Task 7: 实机验证 + W1 workstream 收尾

**Files:**
- 修改: `.claude/workstreams/01-gesture-mapping.md`（勾选任务）

**Interfaces:**
- Consumes: 全部前序任务产物

- [ ] **Step 1: 实机无驱动验证（no-drive）**

```bash
cd /home/bright/office/Leap_Hand/python
python gesture_mapping/demo_hamer3d.py
```

检查项：
- HUD 顶部显示 `3D: HAMER 3D`（非 fallback）
- 倾斜手 → 手指弯曲角度仍正确反映；旋转手 → 角度不跳变；攥拳 → flexion 明显增大
- 按 D 开启叠加：黄色 MANO 骨架贴合手形并与绿色 MediaPipe 关键点基本对齐
- 按 M 切回 MediaPipe 伪 3D，对比同一姿态下角度差异（应明显劣于 hamer）

- [ ] **Step 2: 实机驱动验证（--drive）**

```bash
cd /home/bright/office/Leap_Hand/python
python gesture_mapping/demo_hamer3d.py --drive
```

检查项：
- 手张开 → 灵巧手张开；攥拳 → 灵巧手握拳（方向正确，JOINT_DIR 生效）
- 倾斜/旋转时灵巧手仍跟随（此为 3D 修复的核心验收）
- 无卡死/抖动，1€ 滤波工作正常
- 空格校准后各指零位正确

- [ ] **Step 3: 更新 W1 workstream 勾选状态**

在 `.claude/workstreams/01-gesture-mapping.md` 的 Next Tasks 中，把第 1 条改为已完成，并把"手腕 6DOF 空间定位"上移为下一条：

```markdown
- [x] 3D 手部网格重建（hamer 3D MANO 集成，倾斜/旋转/攥紧 3D 修复）
  - 模块: `python/gesture_mapping/hamer_3d.py` + `demo_hamer3d.py`
  - 运行: `python gesture_mapping/demo_hamer3d.py [--drive]`
- [ ] 手腕 6DOF 空间定位
```

- [ ] **Step 4: 提交收尾**

```bash
cd /home/bright/office/Leap_Hand
git add .claude/workstreams/01-gesture-mapping.md
git commit -m "docs(w1): mark hamer 3D MANO integration done in workstream"
```

---

## Self-Review 记录

- **Spec 覆盖**：设计 §4 数据流 → Task 4/5；§5.1 hamer_3d → Task 4；§5.2 JointMapper → Task 2；§5.3 Calibrator → Task 3；§5.4 FingerIdentifier → Task 3；§5.5 demo → Task 5；§6 回退 → Task 4/5；§7 验证 → Task 4 Step 6 / Task 6 / Task 7。全部覆盖。
- **占位符**：无 TBD/TODO；所有代码步骤含完整实现。
- **类型一致性**：`map_points_to_leap`(Task 2) → `calibrate_points/map_points/identify_points`(Task 3) → `HaMeR3D.regress/project_to_frame`(Task 4) → demo(Task 5) 全程签名一致；`HaMeR3DResult.kp3d` 均为 (21,3) MediaPipe 序。`hand_bbox_from_landmarks` 返回 4 元组 int，Task 4/5/6 一致。
- 唯一设计偏差：`regress(crop)` → `regress(frame, bbox)`（裁剪由 ViTDetDataset 处理），已在 Task 1 Step 3 同步设计文档。
