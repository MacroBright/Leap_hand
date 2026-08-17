"""Gesture mapping package for LEAP Hand — MediaPipe Hands → LEAP 16DOF.

Modules:
    hand_tracker:  MediaPipe Hands 21-keypoint real-time detection.
    joint_mapper:  Keypoint → LEAP Hand 16-DOF joint angle mapping.
    calibrator:    Zero-point calibration + finger identification.
    wrist_tracker: Wrist spatial pose → arm end-effector 6DOF (TODO).
"""

from .hand_tracker import HandTracker
from .joint_mapper import JointMapper
from .calibrator import Calibrator, FingerIdentifier
from .filter import EMAFilter, OneEuroFilter

__all__ = ["HandTracker", "JointMapper", "Calibrator", "FingerIdentifier", "EMAFilter", "OneEuroFilter"]
