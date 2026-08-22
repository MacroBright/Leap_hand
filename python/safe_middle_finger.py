#!/usr/bin/env python3
"""Low-force LEAP Hand middle-finger gesture test.

Sequence: current pose -> open -> gesture -> open -> torque off.
The middle finger (IDs 4-7) stays at the recorded open pose while the other
three fingers use the recorded fist pose.
"""

import argparse
import time

import numpy as np

from leap_hand_utils.dynamixel_client import DynamixelClient
from leap_hand_utils.safety_config import SAFE_PROFILE
from main import POSES


PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HNYU-if00-port0"
MOTOR_IDS = list(range(16))
BAUDRATE = 4_000_000
STEPS = 100
HOLD_SECONDS = 2.0


def interpolate(client, start, target, duration_s, steps=STEPS, sleep=time.sleep):
    for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
        client.write_desired_pos(
            MOTOR_IDS, start + alpha * (target - start)
        )
        sleep(duration_s / steps)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--safe-drive",
        action="store_true",
        default=True,
        help="Accepted for consistency; this script is always in safe mode",
    )
    return parser


def main(argv=None):
    build_parser().parse_args(argv)
    client = DynamixelClient(MOTOR_IDS, PORT, BAUDRATE)
    torque_enabled = False
    try:
        client.connect()
        current = client.read_pos()
        if current.shape != (16,) or not np.all(np.isfinite(current)):
            raise RuntimeError("Invalid current-position reading; refusing to move")

        client.set_torque_enabled(MOTOR_IDS, False)
        client.sync_write(MOTOR_IDS, np.zeros(16), 9, 1)       # Return delay
        client.sync_write(MOTOR_IDS, np.full(16, 5), 11, 1)   # Current-position mode
        client.sync_write(MOTOR_IDS, np.full(16, SAFE_PROFILE.kp), 84, 2)
        client.sync_write(MOTOR_IDS, np.full(16, SAFE_PROFILE.ki), 82, 2)
        client.sync_write(MOTOR_IDS, np.full(16, SAFE_PROFILE.kd), 80, 2)
        client.sync_write(
            MOTOR_IDS, np.full(16, SAFE_PROFILE.goal_current), 102, 2
        )

        open_pose = np.asarray(POSES["全开/平伸"], dtype=float)
        gesture = np.asarray(POSES["全握拳"], dtype=float).copy()
        gesture[4:8] = open_pose[4:8]  # Middle finger stays extended

        # Set the measured pose as the initial goal to prevent a startup jump.
        client.write_desired_pos(MOTOR_IDS, current)
        client.set_torque_enabled(MOTOR_IDS, True)
        torque_enabled = True

        print("[SAFE] Returning smoothly to open pose")
        interpolate(client, current, open_pose, SAFE_PROFILE.startup_seconds)
        time.sleep(0.5)

        print("[SAFE] Moving to middle-finger gesture")
        interpolate(client, open_pose, gesture, SAFE_PROFILE.startup_seconds)
        time.sleep(HOLD_SECONDS)

        print("[SAFE] Returning smoothly to open pose")
        interpolate(client, gesture, open_pose, SAFE_PROFILE.shutdown_seconds)
        time.sleep(0.5)
        print("[SAFE] Motion completed")
    finally:
        if torque_enabled:
            client.set_torque_enabled(MOTOR_IDS, False, retries=1)
        client.port_handler.closePort()
        print("[SAFE] Torque disabled; port closed")


if __name__ == "__main__":
    main()
