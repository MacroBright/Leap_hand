import pytest

from leap_hand_utils.safety_config import SAFE_PROFILE, SafetyProfile


def test_safe_profile_has_validated_project_defaults():
    assert SAFE_PROFILE.kp == 300
    assert SAFE_PROFILE.ki == 0
    assert SAFE_PROFILE.kd == 100
    assert SAFE_PROFILE.goal_current == 150
    assert SAFE_PROFILE.max_speed_rad_s == 1.0
    assert SAFE_PROFILE.startup_seconds == 2.0
    assert SAFE_PROFILE.shutdown_seconds == 2.0
    assert SAFE_PROFILE.tracking_loss_seconds == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kp", -1),
        ("ki", -1),
        ("kd", -1),
        ("goal_current", 0),
        ("max_speed_rad_s", 0),
        ("startup_seconds", 0),
        ("shutdown_seconds", 0),
        ("tracking_loss_seconds", -0.1),
    ],
)
def test_safety_profile_rejects_unsafe_values(field, value):
    values = SAFE_PROFILE.__dict__.copy()
    values[field] = value
    with pytest.raises(ValueError):
        SafetyProfile(**values)
