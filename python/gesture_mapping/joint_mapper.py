"""MediaPipe 21-keypoints → LEAP Hand 16-DOF joint angle mapper.

Finger mapping (5 → 4):
    Human thumb  → LEAP thumb   (ID 12-15)
    Human index  → LEAP index   (ID  0-3)
    Human middle → LEAP middle  (ID  4-7)
    Human ring   → DISCARDED
    Human pinky  → LEAP ring    (ID  8-11)

Joint mapping (per finger, 4 DOF):
    MCP侧摆  — in-plane fan angle (相对中指方向带符号, 指向食指/拇指侧为正)
    MCP前后  — MCP flexion angle
    PIP      — PIP flexion angle
    DIP      — DIP flexion angle

⚠ 拇指物理接线与其他指不同: ID 12=mcp, 13=side, 14=pip, 15=dip。
   (其余四指为 ID x=mcp侧摆, x+1=mcp前后, x+2=PIP, x+3=DIP)
⚠ side 扇角: 食指/中指/无名指以"中指近端方向"为 0° 基准;
   拇指 side 相对"食指近端方向"计算 (用户选定)。

All angles are RELATIVE to OPEN_POSE:  0 = fully open, >0 = flexed.

Workstream: W1 手势映射 — .claude/workstreams/01-gesture-mapping.md
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from .hand_tracker import HandResult, Landmark

# ─── Constants ────────────────────────────────────────────────────
_NUM_LANDMARKS = 21
_NUM_LEAP_DOF = 16

# Angle clipping (relative to OPEN_POSE, in radians)
_ANGLE_MIN = -0.5   # slight hyperextension
_ANGLE_MAX =  2.8   # max flexion

# Per-joint gain: compensate for camera perspective + MediaPipe sensitivity.
#   ID 0,4,8 (side):  0.4 × — dampen, these over-react to small finger spread
#   ID 1-3,5-7,9-11 (mcp/pip/dip): 1.5 × — amplify, camera flattens depth → flexion looks small
#   ID 13 (thumb side): 0.6 × — moderately dampened
#   ID 12 (thumb mcp):  1.5 × — amplify (thumb order differs: 12=mcp, 13=side)
_JOINT_GAIN = np.array([
    0.4, 1.5, 1.5, 1.5,   # index:  side↓  mcp↑  pip↑  dip↑
    0.4, 1.5, 1.5, 1.5,   # middle
    0.4, 1.5, 1.5, 1.5,   # pinky→ring
    1.5, 0.6, 1.5, 1.5,   # thumb:  mcp↑  side↓  pip↑  dip↑  (ID 12=mcp, 13=side)
], dtype=np.float64)

# MediaPipe landmark chain for each finger: [base, joint1, joint2, tip]
# base=CMC for thumb, base=MCP for others
_FINGER_CHAIN: Dict[str, List[int]] = {
    "thumb":  [Landmark["THUMB_CMC"], Landmark["THUMB_MCP"],
               Landmark["THUMB_IP"],  Landmark["THUMB_TIP"]],
    "index":  [Landmark["INDEX_MCP"], Landmark["INDEX_PIP"],
               Landmark["INDEX_DIP"], Landmark["INDEX_TIP"]],
    "middle": [Landmark["MIDDLE_MCP"], Landmark["MIDDLE_PIP"],
               Landmark["MIDDLE_DIP"], Landmark["MIDDLE_TIP"]],
    "ring":   [Landmark["RING_MCP"],  Landmark["RING_PIP"],
               Landmark["RING_DIP"],  Landmark["RING_TIP"]],
    "pinky":  [Landmark["PINKY_MCP"], Landmark["PINKY_PIP"],
               Landmark["PINKY_DIP"], Landmark["PINKY_TIP"]],
}

# Human → LEAP finger mapping (ring discarded)
# (human_name, LEAP_motor_start_idx)
_FINGER_MAP: List[Tuple[str, int]] = [
    ("thumb",  12),   # thumb  → LEAP thumb  (ID 12-15)
    ("index",   0),   # index  → LEAP index  (ID  0-3)
    ("middle",  4),   # middle → LEAP middle (ID  4-7)
    # ring DISCARDED
    ("pinky",   8),   # pinky  → LEAP ring   (ID  8-11)
]

# Dict key labels for map_keypoints_to_leap_dict().
# These name the HUMAN finger driving each LEAP motor group (order: 0-3, 4-7, 8-11, 12-15).
_OUTPUT_FINGER_KEYS = ["index", "middle", "pinky", "thumb"]

# 每指 4 关节 → LEAP 电机偏移顺序 (相对手指起始 ID)
# 标准指: [abd, mcp, pip, dip]; 拇指物理接线不同: [mcp, abd, pip, dip]
_STANDARD_JOINT_ORDER = [0, 1, 2, 3]   # [abd, mcp, pip, dip]
_THUMB_JOINT_ORDER    = [1, 0, 2, 3]   # [mcp, abd, pip, dip]  → ID 12=mcp, 13=side

# map_keypoints_to_leap_dict() 的输出键顺序, 与电机顺序对应
_JOINT_KEYS_STANDARD = ["abduction", "mcp", "pip", "dip"]
_JOINT_KEYS_THUMB    = ["mcp", "abduction", "pip", "dip"]

# 扇角符号: 输出 "向食指/拇指方向张开" 为正 (衔接 JOINT_DIR[side]=-1 → 电机向拇指方向)
# 默认 +1 (lateral 轴已指向食指/拇指侧); 真机发现某指方向反时, 把对应指改为 -1
_FAN_SIGN = {
    "index":  1.0,
    "middle": 1.0,
    "ring":   1.0,
    "pinky":  1.0,
    "thumb":  1.0,
}


class JointMapper:
    """Convert MediaPipe HandResult → LEAP Hand 16-DOF relative angles.

    Usage:
        mapper = JointMapper()
        hand_result = tracker.detect(frame)[0]
        angles = mapper.map_keypoints_to_leap(hand_result, image_shape)
        leap.set_leap(OPEN_POSE + angles)
    """

    def __init__(self, gain: Optional[np.ndarray] = None):
        """Initialize joint-angle mapper.

        Args:
            gain: Optional 16-value per-joint gain array. Defaults to the
                  module-level _JOINT_GAIN (tuned defaults).
        """
        self.joint_gain = (
            _JOINT_GAIN.copy() if gain is None
            else np.array(gain, dtype=np.float64)
        )

    # ─── Main API ─────────────────────────────────────────────────

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

    def map_points_to_leap(
        self,
        pts: np.ndarray,
        frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> np.ndarray:
        """Map a (21,3) point cloud to LEAP 16-DOF relative joint angles.

        Accepts any metric/normalized coordinate frame — angle computation is
        scale-invariant. This is the hamer 3D entry point (real MANO kp3d,
        MediaPipe-index order) and the shared core for the MediaPipe path.

        frame: optional pre-computed (wrist, normal, mid_dir, lateral) palm
               reference — lets callers apply temporal smoothing to the frame
               so fan angles don't jitter with the hand's motion.
        """
        pts = np.asarray(pts, dtype=np.float64)
        if pts.shape != (_NUM_LANDMARKS, 3):
            raise ValueError(
                f"expected a ({_NUM_LANDMARKS}, 3) point cloud, got {pts.shape}"
            )

        if frame is None:
            wrist_pt, palm_normal, mid_dir, lateral = self._palm_frame(pts)
        else:
            wrist_pt, palm_normal, mid_dir, lateral = frame
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
        frame: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Like map_points_to_leap() but returns a labeled dict."""
        angles = self.map_points_to_leap(pts, frame=frame)
        result = {}
        for i, name in enumerate(_OUTPUT_FINGER_KEYS):
            start = i * 4
            keys = _JOINT_KEYS_THUMB if name == "thumb" else _JOINT_KEYS_STANDARD
            result[name] = {k: float(angles[start + j]) for j, k in enumerate(keys)}
        return result

    def map_keypoints_to_leap_dict(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Like map_keypoints_to_leap() but returns a labeled dict."""
        pts = self._build_point_cloud(hand_result, image_shape)
        return self.map_points_to_leap_dict(pts)

    # ─── Gain control (interactive tuning) ────────────────────────

    def set_gain(self, joint_id: int, value: float):
        """Set the gain multiplier for one joint (for live tuning)."""
        if 0 <= joint_id < _NUM_LEAP_DOF:
            self.joint_gain[joint_id] = value
        else:
            raise ValueError(f"joint_id must be 0-{_NUM_LEAP_DOF - 1}, got {joint_id}")

    def get_gain(self, joint_id: int) -> float:
        """Return the current gain for one joint."""
        return float(self.joint_gain[joint_id])

    def save_gain(self, path):
        """Persist current per-joint gains to a JSON file.

        Format: {"joint_gain": [g0, g1, ..., g15]}
        """
        import json
        import os
        payload = {"joint_gain": [float(g) for g in self.joint_gain]}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[SAVE] Gains written to {path}")

    def load_gain_from(self, path):
        """Load per-joint gains from a JSON file. No-op if file missing.

        Format: {"joint_gain": [g0, g1, ..., g15]}
        """
        import json
        from pathlib import Path
        p = Path(path)
        if not p.exists():
            return False
        with open(p) as f:
            payload = json.load(f)
        if "joint_gain" in payload and len(payload["joint_gain"]) == _NUM_LEAP_DOF:
            self.joint_gain = np.array(payload["joint_gain"], dtype=np.float64)
            print(f"[LOAD] Gains loaded from {path}")
            return True
        return False

    # ─── Point cloud ───────────────────────────────────────────────

    @staticmethod
    def _build_point_cloud(
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]],
    ) -> np.ndarray:
        """Convert 21 normalized landmarks to (21, 3) coords.

        x, y scaled to pixel space; z scaled proportionally to width.
        """
        if image_shape:
            h, w = image_shape
        else:
            h = w = 1

        pts = np.zeros((_NUM_LANDMARKS, 3), dtype=np.float64)
        for i, lm in enumerate(hand_result.landmarks):
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
            pts[i, 2] = lm.z * w   # z scale ≈ x scale for consistency
        return pts

    # ─── Palm reference frame ──────────────────────────────────────

    @staticmethod
    def _project_to_plane(v: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """把向量投影到手掌平面 (去掉法线方向分量)."""
        return v - np.dot(v, normal) * normal

    @staticmethod
    def _plane_dir(a: np.ndarray, b: np.ndarray,
                   normal: np.ndarray) -> np.ndarray:
        """a→b 方向向量投影到手掌平面并归一化; 退化时返回零向量."""
        v = JointMapper._project_to_plane(b - a, normal)
        nv = np.linalg.norm(v)
        return v / nv if nv > 1e-9 else np.zeros(3)

    @staticmethod
    def _palm_frame(
        pts: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算手掌参考系.

        Returns (wrist, normal, mid_dir, lateral):
          wrist    — 腕部原点
          normal   — 手掌平面法线 (腕→食指MCP 与 腕→小指MCP 叉积)
          mid_dir  — 中指近端方向 (投影到平面, 归一化), 扇角 0° 固定基准
          lateral  — 平面内横向轴, 指向食指/拇指侧, 扇角取正方向
        """
        wrist = pts[Landmark["WRIST"]]
        idx = pts[Landmark["INDEX_MCP"]]
        pky = pts[Landmark["PINKY_MCP"]]

        v1 = idx - wrist
        v2 = pky - wrist
        normal = np.cross(v1, v2)
        n = np.linalg.norm(normal)
        if n < 1e-9:
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal = normal / n

        # 中指近端方向作为固定基准
        mid_dir = JointMapper._plane_dir(pts[Landmark["MIDDLE_MCP"]],
                                         pts[Landmark["MIDDLE_PIP"]], normal)
        if np.linalg.norm(mid_dir) < 1e-9:
            mid_dir = np.array([0.0, 1.0, 0.0])
            mid_dir = JointMapper._project_to_plane(mid_dir, normal)
            if np.linalg.norm(mid_dir) < 1e-9:
                mid_dir = np.array([0.0, 1.0, 0.0])
        mid_dir = mid_dir / np.linalg.norm(mid_dir)

        # 侧向轴指向食指/拇指侧: 取食指方向相对中指方向的垂直分量
        idx_dir = JointMapper._plane_dir(pts[Landmark["INDEX_MCP"]],
                                         pts[Landmark["INDEX_PIP"]], normal)
        lateral = idx_dir - np.dot(idx_dir, mid_dir) * mid_dir
        ln = np.linalg.norm(lateral)
        if ln < 1e-9:
            # 手指并拢时退化 → 用 n×mid_dir 兜底
            lateral = np.cross(normal, mid_dir)
            ln = np.linalg.norm(lateral)
        if ln < 1e-9:
            lateral = np.array([1.0, 0.0, 0.0])
        else:
            lateral = lateral / ln

        return wrist.copy(), normal, mid_dir, lateral

    # ─── Joint angle computations ──────────────────────────────────

    @staticmethod
    def _compute_flexion(
        p_prox: np.ndarray,
        p_joint: np.ndarray,
        p_dist: np.ndarray,
    ) -> float:
        """Flexion angle at p_joint.

        Angle between (p_prox → p_joint) and (p_joint → p_dist).
        0 = straight (vectors collinear), positive = bent.

        Args:
            p_prox: proximal landmark.
            p_joint: the joint center.
            p_dist: distal landmark.
        """
        v_prox = p_joint - p_prox
        v_dist = p_dist - p_joint

        n_prox = np.linalg.norm(v_prox)
        n_dist = np.linalg.norm(v_dist)
        if n_prox < 1e-9 or n_dist < 1e-9:
            return 0.0

        cos_a = np.dot(v_prox, v_dist) / (n_prox * n_dist)
        cos_a = np.clip(cos_a, -1.0, 1.0)
        # Straight → collinear vectors → cos≈1 → acos≈0 → angle≈0
        return float(np.arccos(cos_a))

    @staticmethod
    def _compute_fan_angle(
        seg_start: np.ndarray,
        seg_end: np.ndarray,
        ref_dir: np.ndarray,
        lateral: np.ndarray,
        normal: np.ndarray,
    ) -> float:
        """平面内扇角: 近端指骨相对 ref_dir 的带符号夹角 (in-plane fan).

        seg = seg_end − seg_start 投影到手掌平面并归一化
        l_ref = lateral 相对 ref_dir 的垂直分量 (保证 ref_dir ⊥ l_ref)
        angle = atan2( seg·l_ref , seg·ref_dir )
          0  = 与参考方向平行 (并拢)
          +  = 偏向 lateral 侧 (食指/拇指侧 = 向拇指方向张开)

        Args:
            seg_start, seg_end: 近端指骨两端关键点 (base, j1)
            ref_dir: 参考方向 (食指/中指/无名指用中指方向; 拇指用食指方向)
            lateral: 平面内横向轴, 指向食指/拇指侧
            normal:  手掌平面法线
        """
        seg = seg_end - seg_start
        seg = JointMapper._project_to_plane(seg, normal)
        ns = np.linalg.norm(seg)
        if ns < 1e-9:
            return 0.0
        seg = seg / ns

        # 取 lateral 相对 ref_dir 的垂直分量, 使 ref_dir ⊥ l_ref (夹角定义正确)
        l_ref = lateral - np.dot(lateral, ref_dir) * ref_dir
        nlr = np.linalg.norm(l_ref)
        if nlr < 1e-9:
            return 0.0
        l_ref = l_ref / nlr

        return float(np.arctan2(np.dot(seg, l_ref), np.dot(seg, ref_dir)))

    # ─── Debug ─────────────────────────────────────────────────────

    def print_angles(self, hand_result: HandResult, image_shape=None):
        """Pretty-print joint angles for debugging."""
        d = self.map_keypoints_to_leap_dict(hand_result, image_shape)
        print(f"\nHand: {hand_result.handedness}")
        print(f"{'Finger':>8s}  {'Abduction':>10s}  {'MCP':>10s}  {'PIP':>10s}  {'DIP':>10s}")
        print("-" * 56)
        for finger, j in d.items():
            print(
                f"{finger:>8s}  {j['abduction']:>+10.3f}  "
                f"{j['mcp']:>+10.3f}  {j['pip']:>+10.3f}  {j['dip']:>+10.3f}"
            )
