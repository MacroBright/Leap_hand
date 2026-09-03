"""camera.py API 契约测试（深度扩展后不破坏 read() 兼容）。"""
import pytest


def test_real_sense_source_has_depth_api():
    from gesture_mapping.camera import RealSenseSource
    assert hasattr(RealSenseSource, "read")
    assert hasattr(RealSenseSource, "read_with_depth")
    assert hasattr(RealSenseSource, "intrinsics")


def test_open_realsense_returns_source_or_none():
    from gesture_mapping.camera import open_realsense
    cam = open_realsense()
    if cam is not None:  # 有 D455 时
        try:
            ok, bgr = cam.read()
            assert ok and bgr is not None
            assert bgr.shape[2] == 3
        finally:
            cam.release()
    else:  # 无硬件时跳过
        pytest.skip("无 RealSense 设备")
