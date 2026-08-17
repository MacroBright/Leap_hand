"""Gesture mapping package for LEAP Hand — MediaPipe Hands → LEAP 16DOF.

Modules:
    hand_tracker:   MediaPipe Hands 21-keypoint real-time detection.
    joint_mapper:   Keypoint → LEAP Hand 16-DOF joint angle mapping.
    calibrator:     Zero-point calibration + finger identification.
    wrist_tracker:  Wrist 3D + palm 6-DOF → 机械臂末端位姿跟随 (位置环).
                   apply_rotation() 2026-08 随 handeye_calib.py 迁 Arm 时
                   内联到本文件 (私有 _apply_rotation), 不再跨仓依赖.

注: handeye_calib.py / arm_client.py / demo_arm_teleop.py 2026-08 已迁
     Arm-robot_VLA/scripts/. 本包仅保留灵巧手 + 视觉共用模块.
"""

from .hand_tracker import HandTracker
from .joint_mapper import JointMapper
from .calibrator import Calibrator, FingerIdentifier
from .filter import EMAFilter, OneEuroFilter

__all__ = ["HandTracker", "JointMapper", "Calibrator", "FingerIdentifier", "EMAFilter", "OneEuroFilter"]
