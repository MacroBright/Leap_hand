""""跟手"量化评估 — 四误差指标 (借鉴 Mingrui-Yu retargeting 评估方法).

人手 MediaPipe 21kp → 当前管线角度(直映/重定向) → LEAP FK 指尖 → 与"人手指尖
规范化描述"对比, 输出 4 类误差 (越小越跟手):
  1. registered global    — 掌心相似变换配准后, 人手→LEAP 指尖位置差 / LEAP 掌宽
  2. relative to wrist    — 相对腕的指尖向量 (掌心基底 + 掌宽归一化) 差
  3. relative to thumb    — 相对拇指 TIP 的指尖向量差
  4. orientation          — 指尖朝向 (DIP→TIP 段, 掌心基底) 夹角

规范化 (掌心基底 + 掌宽归一化) 使 人手尺度(米) 与 LEAP 尺度可比, 规避
"人手↔LEAP 绝对坐标系注册"难题。指标 2/3/4 只反映"指尖相对手掌的到达",
对 3D 源/增益/滤波/重定向 的参数对比最有用。

⚠ 口径注意: LEAP 手手指相对掌宽远短于人手 (FK 实测: 开手手指≈0.9 掌宽,
人手≈2+ 掌宽)。因此**绝对误差含"手指比例差"成分**, 伸直的姿态天然误差大。
本工具用于**改动前后的相对对比** (同一人手姿态, 对比直映/重定向/不同增益),
不作绝对"跟手度"评分。比例差的根治见 P2-2 尺寸归一化。

用法:
  python -m gesture_mapping.measure_following --images python/images/*.jpg
  (需 leap_hand env; 打印每姿态四误差 + 均值)

phase3 重定向对比: 传入自定义 angle_fn (pts→16角) 即可横向对比直映/重定向。
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gesture_mapping import HandTracker, JointMapper, Calibrator
from gesture_mapping.leap_fk import LeapFK

PROJ = Path(__file__).resolve().parent.parent.parent


# ─── 人手指尖/参考 landmark (MediaPipe 索引) ────────────────────────
_TIP_IDX = {"index": 8, "middle": 12, "pinky": 20, "thumb": 4}
_DIP_IDX = {"index": 7, "middle": 11, "pinky": 19, "thumb": 3}   # thumb 用 IP
_WRIST = 0
_IDX_MCP, _PKY_MCP = 5, 17
_FINGER_ORDER = ["index", "middle", "pinky", "thumb"]


class FollowEvaluator:
    """对一组人手点云, 用 angle_fn 算 16 角 → LEAP FK 指尖 → 四误差."""

    def __init__(self, angle_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.fk = LeapFK()
        # 默认 angle_fn = 直映路径 (当前 joint_gain_3d.json + calibration_3d.json)
        self.angle_fn = angle_fn if angle_fn is not None else self._direct_angles

    # ── 直映路径 (复刻 demo_hamer3d 伪3D 增益/基线, 不含滤波) ─────────
    @staticmethod
    def _direct_angles(pts: np.ndarray) -> np.ndarray:
        mapper = JointMapper(gain=FollowEvaluator._load_gain())
        cal = Calibrator(mapper)
        gm_dir = PROJ / "python/gesture_mapping"
        cal.load_points_baseline(str(gm_dir / "calibration_3d.json"))
        return cal.map_points(pts)

    @staticmethod
    def _load_gain() -> np.ndarray:
        p = PROJ / "python/gesture_mapping/joint_gain_3d.json"
        if p.exists():
            with open(p) as f:
                return np.array(json.load(f)["joint_gain"])
        return np.ones(16)

    # ── 规范化描述 ──────────────────────────────────────────────────
    @staticmethod
    def _human_palm(pts: np.ndarray):
        """人手掌参考系: (wrist, mid_dir, lateral, normal, palm_width)."""
        wrist = pts[_WRIST]
        idx_mcp, pky_mcp = pts[_IDX_MCP], pts[_PKY_MCP]
        palm_width = float(np.linalg.norm(idx_mcp - pky_mcp))
        v1, v2 = idx_mcp - wrist, pky_mcp - wrist
        normal = np.cross(v1, v2)
        n = np.linalg.norm(normal)
        normal = normal / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
        mid_dir = v1 - np.dot(v1, normal) * normal
        n = np.linalg.norm(mid_dir)
        mid_dir = mid_dir / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        lateral = np.cross(normal, mid_dir)   # 平面内横轴
        return wrist, mid_dir, lateral, normal, palm_width

    @staticmethod
    def _leap_palm():
        """LEAP 掌心参考: palm_lower 为原点, 手指沿 +x, 侧摆 +y, 法线 +z.
        掌宽 = FK 中 食指MCP→小指MCP 距离."""
        fk = LeapFK()
        jp = fk.joint_positions(np.zeros(16))
        idx_mcp = jp["mcp_joint"]
        pky_mcp = jp["mcp_joint_3"]
        palm_width = float(np.linalg.norm(idx_mcp - pky_mcp))
        return (np.zeros(3), np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), palm_width)

    @staticmethod
    def _canon(tips: np.ndarray, ref: np.ndarray, basis, palm_width: float) -> np.ndarray:
        """指尖向量规范化: (tip-ref) 投影到基底, 除以掌宽. (4,3)."""
        mid, lat, norm = basis
        rel = tips - ref
        return np.column_stack([rel @ mid, rel @ lat, rel @ norm]) / palm_width

    # ── 单姿态四误差 ────────────────────────────────────────────────
    def metrics_for_pose(self, pts: np.ndarray) -> Dict[str, float]:
        """pts: (21,3) 人手点云 (任意尺度, 规范化后可比)."""
        angles = self.angle_fn(pts)
        angles = np.asarray(angles, dtype=np.float64)

        fk_tips = np.array([self.fk.fingertip_positions(angles)[f] for f in _FINGER_ORDER])
        hum_tips = np.array([pts[_TIP_IDX[f]] for f in _FINGER_ORDER])

        hw, hmid, hlat, hnorm, hw = self._human_palm(pts)
        _lw, lmid, llat, lnorm, lw = self._leap_palm()

        # 2) relative to wrist
        hum_c = self._canon(hum_tips, pts[_WRIST], (hmid, hlat, hnorm), hw)
        mach_c = self._canon(fk_tips, np.zeros(3), (lmid, llat, lnorm), lw)
        rel_wrist = float(np.mean(np.linalg.norm(hum_c - mach_c, axis=1)))

        # 3) relative to thumb
        hum_ct = self._canon(hum_tips, hum_tips[3], (hmid, hlat, hnorm), hw)
        mach_ct = self._canon(fk_tips, fk_tips[3], (lmid, llat, lnorm), lw)
        rel_thumb = float(np.mean(np.linalg.norm(hum_ct - mach_ct, axis=1)))

        # 4) orientation: DIP→TIP 段方向夹角 (规范化基底内)
        orient_errs = []
        jp = self.fk.joint_positions(angles)
        dip_link = {"index": "dip", "middle": "dip_2",
                    "pinky": "dip_3", "thumb": "thumb_dip"}
        for i, f in enumerate(_FINGER_ORDER):
            hd = pts[_DIP_IDX[f]] - pts[_TIP_IDX[f]]
            hd = hd / (np.linalg.norm(hd) + 1e-9)
            md = self.fk.fingertip_positions(angles)[f] - jp[dip_link[f]]
            md = md / (np.linalg.norm(md) + 1e-9)
            hd_b = np.array([hd @ hmid, hd @ hlat, hd @ hnorm])
            md_b = np.array([md @ lmid, md @ llat, md @ lnorm])
            ang = np.arccos(np.clip(np.dot(hd_b, md_b) / (
                np.linalg.norm(hd_b) * np.linalg.norm(md_b) + 1e-9), -1.0, 1.0))
            orient_errs.append(float(ang))
        orient = float(np.mean(orient_errs))

        # 1) registered global: 相似变换 人手→LEAP (掌宽缩放 + 基底对齐)
        s = lw / hw
        R = np.column_stack([lmid, llat, lnorm]) @ np.column_stack([hmid, hlat, hnorm]).T
        hum_leap = s * ((hum_tips - pts[_WRIST]) @ R.T)
        mach_leap = fk_tips   # LEAP 腕即 palm_lower 原点
        global_err = float(np.mean(np.linalg.norm(hum_leap - mach_leap, axis=1)) / lw)

        return {"global": global_err, "rel_wrist": rel_wrist,
                "rel_thumb": rel_thumb, "orientation": orient}

    def evaluate(self, points_list: List[np.ndarray]) -> Dict[str, float]:
        """对一组点云求均值."""
        per = [self.metrics_for_pose(p) for p in points_list]
        if not per:
            return {}
        keys = per[0].keys()
        return {k: float(np.mean([p[k] for p in per])) for k in keys}


def _load_images():
    """读 python/images/*.jpg → [(name, human 21×3 点云)] (world-3D 优先)."""
    tracker = HandTracker(max_num_hands=1)
    out = []
    imgs = sorted(glob.glob(str(PROJ / "python/images/*.jpg")))
    imgs = [p for p in imgs if "_compare" not in p and "_hamer3d" not in p]
    for path in imgs:
        import cv2
        img = cv2.imread(path)
        if img is None:
            continue
        results = tracker.detect(img)
        if not results:
            print(f"  {os.path.basename(path)}: no hand")
            continue
        hand = results[0]
        if hand.world_landmarks is not None:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.world_landmarks],
                           dtype=np.float64)
        else:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmarks],
                           dtype=np.float64)
        out.append((os.path.basename(path), pts))
    tracker.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=None,
                    help="图像路径; 缺省用 python/images/*.jpg")
    args = ap.parse_args()

    ev = FollowEvaluator()
    samples = _load_images() if not args.images else []
    if args.images:
        import cv2
        tracker = HandTracker(max_num_hands=1)
        for p in args.images:
            img = cv2.imread(p)
            if img is None:
                continue
            r = tracker.detect(img)
            if r and r[0].world_landmarks is not None:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in r[0].world_landmarks])
                samples.append((os.path.basename(p), pts))
        tracker.close()

    if not samples:
        print("no samples with a detected hand")
        return

    print(f"{'image':12s} {'global':>7s} {'relWrist':>8s} {'relThumb':>8s} {'orient(°)':>9s}")
    acc = {k: [] for k in ("global", "rel_wrist", "rel_thumb", "orientation")}
    for name, pts in samples:
        m = ev.metrics_for_pose(pts)
        for k, v in m.items():
            acc[k].append(v)
        print(f"{name:12s} {m['global']:7.3f} {m['rel_wrist']:8.3f} "
              f"{m['rel_thumb']:8.3f} {np.degrees(m['orientation']):8.1f}")
    print("-" * 50)
    print(f"{'MEAN':12s} "
          f"{np.mean(acc['global']):7.3f} {np.mean(acc['rel_wrist']):8.3f} "
          f"{np.mean(acc['rel_thumb']):8.3f} {np.degrees(np.mean(acc['orientation'])):8.1f}")


if __name__ == "__main__":
    main()
