"""Verify standard package imports and modular API design of leap_hand."""
import numpy as np
import pytest


def test_top_level_exports():
    import leap_hand
    assert hasattr(leap_hand, "__version__")
    assert leap_hand.__version__ == "0.2.0"
    assert hasattr(leap_hand, "LeapNode")
    assert hasattr(leap_hand, "LeapHand")
    assert hasattr(leap_hand, "POSES")
    assert hasattr(leap_hand, "OPEN_POSE")
    assert hasattr(leap_hand, "DynamixelClient")
    assert hasattr(leap_hand, "LEAPHandFK")
    assert hasattr(leap_hand, "HandTracker")
    assert hasattr(leap_hand, "JointMapper")
    assert len(leap_hand.OPEN_POSE) == 16


def test_driver_exports():
    from leap_hand.driver import DynamixelClient
    assert DynamixelClient is not None


def test_kinematics_exports():
    from leap_hand.kinematics import (
        LEAPHandFK,
        LEAPsim_to_LEAPhand,
        angle_safety_clip,
        OneEuroFilter,
        EMAFilter,
    )
    # Test angle_safety_clip function
    test_angles = np.full(16, 3.14)
    clipped = angle_safety_clip(test_angles)
    assert clipped.shape == (16,)

    # Test FK instantiation
    fk = LEAPHandFK()
    assert fk is not None


def test_controller_and_pose_manager():
    from leap_hand.controller import (
        DEFAULT_POSES,
        is_valid_pose,
        load_poses,
        save_poses,
        unwrap_to_limits,
    )
    open_p = DEFAULT_POSES["全开/平伸"]
    assert len(open_p) == 16
    assert is_valid_pose(open_p) is True

    # Zero pose should be invalid
    assert is_valid_pose(np.zeros(16)) is False

    # Test unwrap
    unwrapped = unwrap_to_limits(open_p)
    assert unwrapped.shape == (16,)


def test_vision_exports():
    from leap_hand.vision import (
        Calibrator,
        FingerIdentifier,
        HandTracker,
        JointMapper,
        WristTracker,
    )
    mapper = JointMapper()
    assert mapper is not None
    assert mapper.joint_gain.shape == (16,)


def test_cli_entry_points():
    from leap_hand.cli import (
        control_main,
        calibrate_main,
        teleop_main,
        teleop_3d_main,
        diagnostics_main,
        latency_main,
    )
    assert callable(control_main)
    assert callable(calibrate_main)
    assert callable(teleop_main)
    assert callable(teleop_3d_main)
    assert callable(diagnostics_main)
    assert callable(latency_main)
