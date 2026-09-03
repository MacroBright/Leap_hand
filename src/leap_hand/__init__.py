"""LEAP Hand: 16-DOF Dexterous Hand perception, kinematics, control, and teleoperation subsystem."""
__version__ = "0.2.0"

from .controller.hand_controller import LeapNode, POSES, OPEN_POSE
from .driver.dynamixel_client import DynamixelClient
from .kinematics.leap_fk import LEAPHandFK
from .kinematics.limits import angle_safety_clip
from .vision.hand_tracker import HandTracker
from .vision.joint_mapper import JointMapper

# High-level alias
LeapHand = LeapNode

__all__ = [
    "__version__",
    "LeapNode",
    "LeapHand",
    "POSES",
    "OPEN_POSE",
    "DynamixelClient",
    "LEAPHandFK",
    "angle_safety_clip",
    "HandTracker",
    "JointMapper",
]
