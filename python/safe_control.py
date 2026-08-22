#!/usr/bin/env python3
"""One-shot low-force LEAP Hand half-grasp test.

This script intentionally does not construct LeapNode because LeapNode enables
the normal high-gain configuration during initialization.  It reads the
current pose first, enables torque with conservative gains, interpolates to
the recorded half-grasp, then always disables torque before exiting.
"""

import time

import numpy as np

from leap_hand_utils.dynamixel_client import DynamixelClient
from main import POSES


PORT = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HNYU-if00-port0"
MOTOR_IDS = list(range(16))
BAUDRATE = 4_000_000
KP = 300
KI = 0
KD = 100
GOAL_CURRENT = 150
DURATION_S = 2.0
STEPS = 100


def main():
    client = DynamixelClient(MOTOR_IDS, PORT, BAUDRATE)
    client.connect()
    torque_enabled = False
    try:
        start = client.read_pos()
        if start.shape != (16,) or not np.all(np.isfinite(start)):
            raise RuntimeError("Invalid current-position reading; refusing to move")

        # Configure while torque is off. Address 11 is current-based position mode.
        client.set_torque_enabled(MOTOR_IDS, False)
        client.sync_write(MOTOR_IDS, np.zeros(16), 9, 1)       # Return delay
        client.sync_write(MOTOR_IDS, np.full(16, 5), 11, 1)   # Operating mode
        client.sync_write(MOTOR_IDS, np.full(16, KP), 84, 2)
        client.sync_write(MOTOR_IDS, np.full(16, KI), 82, 2)
        client.sync_write(MOTOR_IDS, np.full(16, KD), 80, 2)
        client.sync_write(MOTOR_IDS, np.full(16, GOAL_CURRENT), 102, 2)

        # Hold the measured pose before torque is enabled to avoid a startup jump.
        client.write_desired_pos(MOTOR_IDS, start)
        client.set_torque_enabled(MOTOR_IDS, True)
        torque_enabled = True

        open_pose = np.asarray(POSES["全开/平伸"], dtype=float)
        target = np.asarray(POSES["半握"], dtype=float)

        print("Returning smoothly to open pose")
        for alpha in np.linspace(0.0, 1.0, STEPS + 1)[1:]:
            client.write_desired_pos(MOTOR_IDS, start + alpha * (open_pose - start))
            time.sleep(DURATION_S / STEPS)

        time.sleep(0.5)
        print("Moving smoothly from open pose to half-grasp")
        for alpha in np.linspace(0.0, 1.0, STEPS + 1)[1:]:
            client.write_desired_pos(MOTOR_IDS, open_pose + alpha * (target - open_pose))
            time.sleep(DURATION_S / STEPS)

        time.sleep(1.0)
        print("Low-force half-grasp completed; returning to open pose")
        for alpha in np.linspace(0.0, 1.0, STEPS + 1)[1:]:
            client.write_desired_pos(MOTOR_IDS, target + alpha * (open_pose - target))
            time.sleep(DURATION_S / STEPS)

        time.sleep(0.5)
        print("Measured position:", client.read_pos())
    finally:
        if torque_enabled:
            client.set_torque_enabled(MOTOR_IDS, False, retries=1)
        client.port_handler.closePort()
        print("Torque disabled; port closed")


if __name__ == "__main__":
    main()
