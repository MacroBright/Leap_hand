"""Unified camera source for the LEAP Hand gesture pipeline.

Prefers the official Intel RealSense SDK (pyrealsense2) so the D455's
color stream is selected explicitly — scanning raw V4L2 ports is
unreliable because the D455 re-enumerates its color/depth/IR nodes on
every run. Falls back to an OpenCV port scan if the SDK is unavailable.

Both sources expose the same minimal interface:
    ok, bgr_frame = cam.read()
    cam.release()

Workstream: W1 手势映射 — .claude/workstreams/01-gesture-mapping.md
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraSource:
    """Abstract camera source: read() → (ok, BGR frame), release()."""

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


class RealSenseSource(CameraSource):
    """D455 color stream via the official pyrealsense2 SDK.

    Requests the RGB stream explicitly (640x480 @30fps), so there is no
    ambiguity about which V4L2 node is color. BGR output matches OpenCV.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._profile = self._pipeline.start(config)
        self._width, self._height = width, height

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frames = self._pipeline.wait_for_frames()
            color = frames.get_color_frame()
            if color is None:
                return False, None
            img = np.asanyarray(color.get_data())
            return True, img.copy()
        except Exception:
            return False, None

    def release(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass


def open_realsense(width: int = 640, height: int = 480, fps: int = 30) -> Optional[CameraSource]:
    """Open the D455 color stream via the official SDK.

    The D455 does not expose 640x480 BGR8 @60fps — requesting it makes
    pipeline.start() throw "Couldn't resolve requests". Default to 30fps,
    which is reliable; a faster BGR combo is not supported on this device.

    Returns None if pyrealsense2 is not installed or no RealSense device
    is present (caller should fall back to OpenCV).
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        return None

    try:
        ctx = rs.context()
        if not any(ctx.query_devices()):
            return None
        cam = RealSenseSource(width, height, fps)
        # one warm frame to confirm the stream works
        ok, frame = cam.read()
        if not ok or frame is None:
            cam.release()
            return None
        return cam
    except Exception:
        return None
