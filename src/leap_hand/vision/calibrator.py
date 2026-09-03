"""Hand-to-robot calibration and finger identification for LEAP Hand.

Calibrator:      capture open-hand baseline → zero-offset correction.
FingerIdentifier: detect which human finger is currently bent.

Workstream: W1 手势映射 — .claude/workstreams/01-gesture-mapping.md
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from .hand_tracker import HandResult
from .joint_mapper import (JointMapper, _FINGER_MAP, _FINGER_CHAIN,
                           _JOINT_KEYS_STANDARD, _JOINT_KEYS_THUMB)

# Human finger display labels
HUMAN_FINGER_LABELS = {
    "thumb": "Thumb (拇指)",
    "index": "Index (食指)",
    "middle": "Middle (中指)",
    "pinky": "Pinky (小指→LEAPring)",
}

# LEAP motor-group labels — shows which human finger drives each group
LEAP_GROUP_LABELS = [
    "Idx→LEAP[0-3]",    # human index  → LEAP motors 0-3
    "Mid→LEAP[4-7]",    # human middle → LEAP motors 4-7
    "Pky→LEAP[8-11]",   # human pinky  → LEAP motors 8-11
    "Thb→LEAP[12-15]",  # human thumb  → LEAP motors 12-15
]


class Calibrator:
    """Zero-point calibration for hand-to-LEAP joint mapping.

    Usage:
        cal = Calibrator(mapper)
        cal.calibrate(open_hand, image_shape)    # record open-hand baseline
        angles = cal.map(hand, image_shape)       # zero-corrected angles
    """

    def __init__(self, mapper: JointMapper):
        self.mapper = mapper
        self._baseline: Optional[np.ndarray] = None
        self._calibrated = False
        # 3D point-cloud (hamer) baseline kept in its OWN slots so it can never
        # be mixed with the HandResult baseline (different 3D sources/scales).
        self._baseline_points: Optional[np.ndarray] = None
        self._calibrated_points = False

    def calibrate(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Record current hand pose as the zero reference.

        Call this with your hand fully open (all fingers straight).
        """
        self._baseline = self.mapper.map_keypoints_to_leap(hand_result, image_shape)
        self._calibrated = True
        return self._baseline

    def calibrate_points(self, pts: np.ndarray, frame=None) -> np.ndarray:
        """Record current 3D point cloud as the zero reference (call with hand fully open)."""
        self._baseline_points = self.mapper.map_points_to_leap(pts, frame=frame)
        self._calibrated_points = True
        return self._baseline_points

    def map_points(self, pts: np.ndarray, frame=None) -> np.ndarray:
        """Zero-corrected joint angles from a (21,3) point cloud (hamer path)."""
        raw = self.mapper.map_points_to_leap(pts, frame=frame)
        if self._calibrated_points and self._baseline_points is not None:
            return np.clip(raw - self._baseline_points, -0.3, 2.8)
        return raw

    def save_points_baseline(self, path) -> None:
        """Persist the points baseline so a later session doesn't need re-zeroing."""
        if self._baseline_points is not None:
            with open(path, "w") as f:
                json.dump(
                    {"baseline_points": [float(x) for x in self._baseline_points]},
                    f, indent=2,
                )

    def load_points_baseline(self, path) -> bool:
        """Load a persisted points baseline; returns True if loaded."""
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
            self._baseline_points = np.array(data["baseline_points"], dtype=np.float64)
            self._calibrated_points = True
            return True
        return False

    def map(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Compute zero-corrected joint angles (baseline subtracted)."""
        raw = self.mapper.map_keypoints_to_leap(hand_result, image_shape)
        if self._calibrated and self._baseline is not None:
            return np.clip(raw - self._baseline, -0.3, 2.8)
        return raw

    def map_dict(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict:
        """Zero-corrected angles as a dict keyed by LEAP group."""
        angles = self.map(hand_result, image_shape)
        keys = ["index", "middle", "pinky", "thumb"]
        result = {}
        for i, name in enumerate(keys):
            s = i * 4
            jkeys = _JOINT_KEYS_THUMB if name == "thumb" else _JOINT_KEYS_STANDARD
            result[name] = {k: float(angles[s + j]) for j, k in enumerate(jkeys)}
        return result

    @property
    def baseline(self) -> Optional[np.ndarray]:
        return self._baseline

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated or self._calibrated_points


class FingerIdentifier:
    """Detect which human finger is currently bent.

    Usage:
        fi = FingerIdentifier()
        bent, scores = fi.identify(hand_result, image_shape)
        if bent:
            print(f"Bent finger: {HUMAN_FINGER_LABELS[bent]}")

    Two methods:
        1. Joint-angle method (default): uses mapped LEAP angles
        2. Geometric method (by_flexion): uses landmark geometry directly
    """

    def __init__(
        self,
        mapper: Optional[JointMapper] = None,
        bend_threshold: float = 0.3,
    ):
        self.mapper = mapper or JointMapper()
        self.threshold = bend_threshold

    def identify(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Identify most bent finger using joint-angle based scoring.

        Returns:
            (human_finger_name | None, {finger: score, ...})
        """
        d = self.mapper.map_keypoints_to_leap_dict(hand_result, image_shape)

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

    def identify_geometry(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[Optional[str], Dict[str, float]]:
        """Identify bent finger by geometry (tip-to-base straightness ratio).

        Straight finger → ratio ≈ 1.0 (tip is far from base)
        Bent finger → ratio < 1.0 (tip curves back toward base)
        """
        h, w = image_shape if image_shape else (1, 1)

        pts = np.zeros((21, 3), dtype=np.float64)
        for i, lm in enumerate(hand_result.landmarks):
            pts[i, 0] = lm.x * w
            pts[i, 1] = lm.y * h
            pts[i, 2] = lm.z * w

        scores = {}
        for fname, _ in _FINGER_MAP:
            chain = _FINGER_CHAIN[fname]
            base = pts[chain[0]]
            tip = pts[chain[3]]

            straight = np.linalg.norm(tip - base)
            curved = sum(
                np.linalg.norm(pts[chain[i + 1]] - pts[chain[i]])
                for i in range(len(chain) - 1)
            )
            if curved > 1e-9:
                scores[fname] = max(0.0, 1.0 - straight / curved)
            else:
                scores[fname] = 0.0

        max_finger = max(scores, key=scores.get)
        return (
            (max_finger, scores) if scores[max_finger] >= self.threshold * 0.3
            else (None, scores)
        )

    def bent_fingers(
        self,
        hand_result: HandResult,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, float]:
        """Return ALL fingers with their bend scores (for display)."""
        _, scores = self.identify(hand_result, image_shape)
        return scores
