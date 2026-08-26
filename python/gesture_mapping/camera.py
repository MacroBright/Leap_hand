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

    When enable_depth=True the depth stream is requested at the same
    resolution and aligned to the color frame (rs.align), so read_with_depth()
    returns a depth map registered to the BGR image (u16 mm). Depth and color
    intrinsics are exposed via intrinsics().
    """

    def __init__(self, width: int = 848, height: int = 480, fps: int = 60,
                 enable_depth: bool = True):
        import pyrealsense2 as rs
        self._rs = rs
        self._pipeline = rs.pipeline()
        
        # 优先尝试 60fps 高帧率模式 (D455/D435 官方原生 848x480 @60fps / 640x480 @60fps)
        started = False
        candidates = [
            (width, height, fps),
            (848, 480, 60),
            (640, 480, 60),
            (640, 480, 30),
        ]
        for w_c, h_c, fps_c in candidates:
            try:
                config = rs.config()
                config.enable_stream(rs.stream.color, w_c, h_c, rs.format.bgr8, fps_c)
                if enable_depth:
                    config.enable_stream(rs.stream.depth, w_c, h_c, rs.format.z16, fps_c)
                self._profile = self._pipeline.start(config)
                self._width, self._height, self._fps = w_c, h_c, fps_c
                started = True
                break
            except Exception:
                continue

        if not started:
            raise RuntimeError("Failed to start RealSense pipeline with any candidate stream profile")

        self._align = rs.align(rs.stream.color)
        self._intrinsics = None
        try:
            intr = (self._profile.get_stream(rs.stream.color)
                    .as_video_stream_profile().get_intrinsics())
            self._intrinsics = (intr.fx, intr.fy, intr.ppx, intr.ppy)
        except Exception:
            self._intrinsics = None

    def _next_aligned(self):
        frames = self._pipeline.wait_for_frames()
        return self._align.process(frames)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            frames = self._next_aligned()
            color = frames.get_color_frame()
            if color is None:
                return False, None
            return True, np.asanyarray(color.get_data()).copy()
        except Exception:
            return False, None

    def read_with_depth(self) -> Tuple[bool, Optional[np.ndarray],
                                       Optional[np.ndarray], Optional[tuple]]:
        """返回 (ok, bgr, depth_mm, intrinsics)。depth 已与 color 对齐 (u16 mm)。"""
        try:
            frames = self._next_aligned()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if color is None:
                return False, None, None, None
            bgr = np.asanyarray(color.get_data()).copy()
            depth_mm = (np.asanyarray(depth.get_data()).copy()
                        if depth is not None else None)
            return True, bgr, depth_mm, self._intrinsics
        except Exception:
            return False, None, None, None

    def intrinsics(self):
        return self._intrinsics

    def release(self):
        try:
            self._pipeline.stop()
        except Exception:
            pass


def open_realsense(width: int = 848, height: int = 480, fps: int = 60) -> Optional[CameraSource]:
    """Open the RealSense color + depth stream (defaults to high-speed 60fps with automatic fallback)."""
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
