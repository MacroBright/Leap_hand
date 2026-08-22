"""Low-force, rate-limited controller for LEAP Hand visual teleoperation."""

import time
from typing import Callable, Optional, Sequence

import numpy as np

from leap_hand_utils.safety_config import SAFE_PROFILE, SafetyProfile


class SafeLeapController:
    """Own the safe motor lifecycle for visual teleoperation.

    The Dynamixel client is injected so unit tests never touch a serial device.
    """

    MOTOR_IDS = list(range(16))
    def __init__(
        self,
        client,
        open_pose: Sequence[float],
        profile: SafetyProfile = SAFE_PROFILE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_speed_rad_s: Optional[float] = None,
        loss_timeout_s: Optional[float] = None,
    ):
        self.client = client
        self.open_pose = self._validate_pose(open_pose, "open_pose")
        self.profile = profile
        self.clock = clock
        self.sleep = sleep
        self.max_speed_rad_s = float(
            profile.max_speed_rad_s
            if max_speed_rad_s is None
            else max_speed_rad_s
        )
        self.loss_timeout_s = float(
            profile.tracking_loss_seconds
            if loss_timeout_s is None
            else loss_timeout_s
        )
        if self.max_speed_rad_s <= 0:
            raise ValueError("max_speed_rad_s must be positive")
        if self.loss_timeout_s < 0:
            raise ValueError("loss_timeout_s cannot be negative")

        self.commanded_pose: Optional[np.ndarray] = None
        self.last_seen_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.torque_enabled = False
        self.closed = False

    @staticmethod
    def _validate_pose(pose, name):
        value = np.asarray(pose, dtype=float)
        if value.shape != (16,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain 16 finite values")
        return value.copy()

    def _limited_target(self, target, now):
        target = self._validate_pose(target, "target")
        if self.commanded_pose is None or self.last_update_time is None:
            return target
        dt = max(0.0, now - self.last_update_time)
        max_delta = self.max_speed_rad_s * dt
        delta = np.clip(
            target - self.commanded_pose,
            -max_delta,
            max_delta,
        )
        return self.commanded_pose + delta

    def _write_limited(self, target, now):
        pose = self._limited_target(target, now)
        try:
            self.client.write_desired_pos(self.MOTOR_IDS, pose)
        except Exception:
            self.shutdown(return_open=False)
            raise
        self.commanded_pose = pose
        self.last_update_time = now
        return pose.copy()

    def _interpolate(self, start, target, duration_s, frequency_hz=50.0):
        start = self._validate_pose(start, "start")
        target = self._validate_pose(target, "target")
        steps = max(1, int(duration_s * frequency_hz))
        delay = duration_s / steps if duration_s else 0.0
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            pose = start + alpha * (target - start)
            self.client.write_desired_pos(self.MOTOR_IDS, pose)
            self.commanded_pose = pose.copy()
            self.sleep(delay)
        self.last_update_time = self.clock()

    def start(self, interpolation_s=None):
        if interpolation_s is None:
            interpolation_s = self.profile.startup_seconds
        try:
            self.client.connect()
            current = self._validate_pose(self.client.read_pos(), "current pose")

            # EEPROM-like mode changes and gains are configured with torque off.
            self.client.set_torque_enabled(self.MOTOR_IDS, False)
            self.client.sync_write(self.MOTOR_IDS, np.zeros(16), 9, 1)
            self.client.sync_write(self.MOTOR_IDS, np.full(16, 5), 11, 1)
            self.client.sync_write(
                self.MOTOR_IDS, np.full(16, self.profile.kp), 84, 2
            )
            self.client.sync_write(
                self.MOTOR_IDS, np.full(16, self.profile.ki), 82, 2
            )
            self.client.sync_write(
                self.MOTOR_IDS, np.full(16, self.profile.kd), 80, 2
            )
            self.client.sync_write(
                self.MOTOR_IDS,
                np.full(16, self.profile.goal_current),
                102,
                2,
            )

            # Prevent an enable-time jump by making the measured pose the goal.
            self.client.write_desired_pos(self.MOTOR_IDS, current)
            self.commanded_pose = current.copy()
            self.client.set_torque_enabled(self.MOTOR_IDS, True)
            self.torque_enabled = True

            self._interpolate(current, self.open_pose, interpolation_s)
            self.last_seen_time = self.clock()
        except Exception:
            self.shutdown(return_open=False)
            raise

    def track(self, target):
        now = self.clock()
        self.last_seen_time = now
        return self._write_limited(target, now)

    def on_tracking_lost(self):
        now = self.clock()
        if self.commanded_pose is None:
            return None
        if (
            self.last_seen_time is None
            or now - self.last_seen_time <= self.loss_timeout_s
        ):
            # Reset the rate-limit interval during the grace period so the
            # first recovery step cannot accumulate a large jump.
            self.last_update_time = now
            return self.commanded_pose.copy()
        return self._write_limited(self.open_pose, now)

    def shutdown(self, return_open=True, interpolation_s=None):
        if self.closed:
            return
        if interpolation_s is None:
            interpolation_s = self.profile.shutdown_seconds
        try:
            if (
                return_open
                and self.torque_enabled
                and self.commanded_pose is not None
            ):
                self._interpolate(
                    self.commanded_pose, self.open_pose, interpolation_s
                )
        finally:
            try:
                self.client.set_torque_enabled(
                    self.MOTOR_IDS, False, retries=1
                )
            finally:
                self.torque_enabled = False
                self.client.port_handler.closePort()
                self.closed = True
