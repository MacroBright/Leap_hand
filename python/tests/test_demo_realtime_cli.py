import inspect

import pytest

from gesture_mapping.demo_realtime import build_parser, main


def test_drive_modes_are_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--drive", "--safe-drive"])


def test_safe_drive_flag_is_opt_in():
    args = build_parser().parse_args(["--camera", "0", "--safe-drive"])
    assert args.safe_drive is True
    assert args.drive is False
    assert args.camera == 0


def test_vision_only_remains_the_default():
    args = build_parser().parse_args([])
    assert args.safe_drive is False
    assert args.drive is False


def test_safe_hardware_starts_only_after_camera_warmup():
    source = inspect.getsource(main)
    assert source.index('print("[INFO] Camera warm.') < source.index(
        "safe_leap.start()"
    )
