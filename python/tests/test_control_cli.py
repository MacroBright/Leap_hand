from interactive_control import build_parser as interactive_parser
from main import build_parser as main_parser
from safe_control import build_parser as half_parser
from safe_middle_finger import build_parser as middle_parser, interpolate

import numpy as np


def test_optional_controllers_accept_safe_drive():
    assert interactive_parser().parse_args(["--safe-drive"]).safe_drive
    assert main_parser().parse_args(["--safe-drive"]).safe_drive


def test_optional_controllers_keep_normal_mode_by_default():
    assert interactive_parser().parse_args([]).safe_drive is False
    assert main_parser().parse_args([]).safe_drive is False


def test_always_safe_scripts_accept_consistency_alias():
    assert half_parser().parse_args(["--safe-drive"]).safe_drive
    assert middle_parser().parse_args(["--safe-drive"]).safe_drive


def test_always_safe_scripts_remain_safe_without_alias():
    assert half_parser().parse_args([]).safe_drive is True
    assert middle_parser().parse_args([]).safe_drive is True


def test_middle_finger_interpolation_uses_requested_duration():
    class Client:
        def write_desired_pos(self, ids, pose):
            pass

    sleeps = []
    interpolate(
        Client(),
        np.zeros(16),
        np.ones(16),
        duration_s=3.0,
        steps=3,
        sleep=sleeps.append,
    )
    assert sum(sleeps) == 3.0
