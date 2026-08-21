# Safe Visual Drive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--safe-drive` mode to the MediaPipe demo that uses validated low-force motor settings, smooth startup/shutdown, target-rate limiting, and a 0.5-second tracking-loss grace period.

**Architecture:** A new `SafeLeapController` owns all safe motor state and accepts an injected Dynamixel client and clock so its behavior can be tested without hardware. `demo_realtime.py` remains responsible for vision and mapping, forwarding valid target poses or missing-hand events to the controller. The existing `LeapNode` and `--drive` path remain unchanged.

**Tech Stack:** Python 3.10, NumPy, Dynamixel SDK wrapper, argparse, pytest, MediaPipe/OpenCV.

---

## File Map

- Create `python/gesture_mapping/safe_leap_controller.py`: low-force configuration, interpolation, rate limiting, tracking-loss behavior, and idempotent shutdown.
- Create `python/tests/test_safe_leap_controller.py`: fake-client unit tests that never open a serial device.
- Modify `python/gesture_mapping/demo_realtime.py`: add mutually exclusive `--safe-drive`, construct the safe controller, and route detection/loss/cleanup events.
- Create `python/tests/test_demo_realtime_cli.py`: parser and mode-selection regression tests.
- Modify `README.md`: document the safe visual-drive command, startup sequence, and emergency-stop behavior.

### Task 1: Pure target limiting and tracking-loss state

**Files:**
- Create: `python/gesture_mapping/safe_leap_controller.py`
- Create: `python/tests/test_safe_leap_controller.py`

- [ ] **Step 1: Write failing tests for rate limiting and tracking loss**

```python
import numpy as np

from gesture_mapping.safe_leap_controller import SafeLeapController


class FakeClient:
    def __init__(self):
        self.writes = []

    def write_desired_pos(self, motor_ids, pose):
        self.writes.append(np.asarray(pose, dtype=float).copy())


def make_controller(clock):
    controller = SafeLeapController(
        client=FakeClient(),
        open_pose=np.zeros(16),
        clock=clock,
        sleep=lambda _: None,
        max_speed_rad_s=1.0,
        loss_timeout_s=0.5,
    )
    controller.commanded_pose = np.zeros(16)
    controller.last_update_time = clock[0]
    return controller


def test_target_step_is_limited_by_elapsed_time():
    clock = [10.0]
    controller = make_controller(lambda: clock[0])
    clock[0] = 10.1
    result = controller.track(np.ones(16))
    np.testing.assert_allclose(result, np.full(16, 0.1), atol=1e-6)


def test_short_tracking_loss_holds_last_target():
    clock = [20.0]
    controller = make_controller(lambda: clock[0])
    controller.last_seen_time = 20.0
    controller.commanded_pose[:] = 0.4
    clock[0] = 20.49
    result = controller.on_tracking_lost()
    np.testing.assert_allclose(result, np.full(16, 0.4))


def test_long_tracking_loss_moves_toward_open():
    clock = [30.0]
    controller = make_controller(lambda: clock[0])
    controller.last_seen_time = 30.0
    controller.last_update_time = 30.5
    controller.commanded_pose[:] = 0.4
    clock[0] = 30.6
    result = controller.on_tracking_lost()
    np.testing.assert_allclose(result, np.full(16, 0.3), atol=1e-6)
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
cd ~/projects/TuinaDex/Leap_Hand/python
pytest -q tests/test_safe_leap_controller.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'gesture_mapping.safe_leap_controller'`.

- [ ] **Step 3: Implement the minimal pure state behavior**

Create `python/gesture_mapping/safe_leap_controller.py` with constants and these methods:

```python
import time
from typing import Callable, Optional, Sequence

import numpy as np


class SafeLeapController:
    MOTOR_IDS = list(range(16))

    def __init__(
        self,
        client,
        open_pose: Sequence[float],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        max_speed_rad_s: float = 1.0,
        loss_timeout_s: float = 0.5,
    ):
        self.client = client
        self.open_pose = np.asarray(open_pose, dtype=float).copy()
        if self.open_pose.shape != (16,) or not np.all(np.isfinite(self.open_pose)):
            raise ValueError("open_pose must contain 16 finite values")
        self.clock = clock
        self.sleep = sleep
        self.max_speed_rad_s = float(max_speed_rad_s)
        self.loss_timeout_s = float(loss_timeout_s)
        self.commanded_pose: Optional[np.ndarray] = None
        self.last_seen_time: Optional[float] = None
        self.last_update_time: Optional[float] = None
        self.torque_enabled = False
        self.closed = False

    def _limited_target(self, target, now):
        target = np.asarray(target, dtype=float)
        if target.shape != (16,) or not np.all(np.isfinite(target)):
            raise ValueError("target must contain 16 finite values")
        if self.commanded_pose is None or self.last_update_time is None:
            return target.copy()
        dt = max(0.0, now - self.last_update_time)
        delta = np.clip(
            target - self.commanded_pose,
            -self.max_speed_rad_s * dt,
            self.max_speed_rad_s * dt,
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

    def track(self, target):
        now = self.clock()
        self.last_seen_time = now
        return self._write_limited(target, now)

    def on_tracking_lost(self):
        now = self.clock()
        if self.commanded_pose is None:
            return None
        if self.last_seen_time is None or now - self.last_seen_time <= self.loss_timeout_s:
            return self.commanded_pose.copy()
        return self._write_limited(self.open_pose, now)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run `pytest -q tests/test_safe_leap_controller.py`.

Expected: `3 passed`.

- [ ] **Step 5: Commit the pure controller behavior**

```bash
git add python/gesture_mapping/safe_leap_controller.py \
        python/tests/test_safe_leap_controller.py
git commit -m "feat: add safe LEAP target state controller"
```

### Task 2: Safe hardware startup, interpolation, and shutdown

**Files:**
- Modify: `python/gesture_mapping/safe_leap_controller.py`
- Modify: `python/tests/test_safe_leap_controller.py`

- [ ] **Step 1: Extend the fake client and write failing lifecycle tests**

Add a fake that records ordered operations:

```python
class LifecycleClient(FakeClient):
    def __init__(self, current=None, fail_write=False):
        super().__init__()
        self.current = np.asarray(current if current is not None else np.ones(16), dtype=float)
        self.events = []
        self.fail_write = fail_write
        self.port_handler = type("Port", (), {"closePort": lambda port: self.events.append("close")})()

    def connect(self):
        self.events.append("connect")

    def read_pos(self):
        self.events.append("read")
        return self.current.copy()

    def sync_write(self, ids, values, address, size):
        self.events.append(("sync", address, tuple(np.asarray(values).astype(int))))

    def set_torque_enabled(self, ids, enabled, retries=-1):
        self.events.append(("torque", enabled))

    def write_desired_pos(self, ids, pose):
        if self.fail_write:
            raise RuntimeError("write failed")
        self.events.append(("position", np.asarray(pose).copy()))
        super().write_desired_pos(ids, pose)


def test_startup_writes_current_pose_before_enabling_torque():
    client = LifecycleClient(current=np.full(16, 0.25))
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    position_index = next(i for i, e in enumerate(client.events) if e[0] == "position")
    torque_on_index = client.events.index(("torque", True))
    assert position_index < torque_on_index
    np.testing.assert_allclose(client.events[position_index][1], np.full(16, 0.25))


def test_startup_applies_validated_low_force_configuration():
    client = LifecycleClient()
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    assert any(e[0:2] == ("sync", 84) and set(e[2]) == {300} for e in client.events)
    assert any(e[0:2] == ("sync", 80) and set(e[2]) == {100} for e in client.events)
    assert any(e[0:2] == ("sync", 102) and set(e[2]) == {150} for e in client.events)


def test_shutdown_disables_torque_and_closes_port():
    client = LifecycleClient()
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    controller.shutdown(return_open=False)
    assert client.events[-2:] == [("torque", False), "close"]


def test_startup_failure_attempts_torque_off_and_close():
    client = LifecycleClient(fail_write=True)
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    try:
        controller.start(interpolation_s=0.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected startup failure")
    assert ("torque", False) in client.events
    assert "close" in client.events


def test_runtime_write_failure_stops_immediately():
    client = LifecycleClient()
    controller = SafeLeapController(client, np.zeros(16), sleep=lambda _: None)
    controller.start(interpolation_s=0.0)
    client.fail_write = True
    try:
        controller.track(np.ones(16))
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected runtime write failure")
    assert client.events[-2:] == [("torque", False), "close"]
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run `pytest -q tests/test_safe_leap_controller.py`.

Expected: failures report missing `start` and `shutdown` methods.

- [ ] **Step 3: Implement startup, interpolation, and idempotent shutdown**

Add safe configuration constants and lifecycle methods:

```python
    KP = 300
    KI = 0
    KD = 100
    GOAL_CURRENT = 150

    def _interpolate(self, start, target, duration_s, frequency_hz=50.0):
        steps = max(1, int(duration_s * frequency_hz))
        for alpha in np.linspace(0.0, 1.0, steps + 1)[1:]:
            pose = start + alpha * (target - start)
            self.client.write_desired_pos(self.MOTOR_IDS, pose)
            self.commanded_pose = pose.copy()
            self.sleep(duration_s / steps if duration_s else 0.0)
        self.last_update_time = self.clock()

    def start(self, interpolation_s=2.0):
        try:
            self.client.connect()
            current = np.asarray(self.client.read_pos(), dtype=float)
            if current.shape != (16,) or not np.all(np.isfinite(current)):
                raise RuntimeError("invalid current-position reading")
            self.client.set_torque_enabled(self.MOTOR_IDS, False)
            self.client.sync_write(self.MOTOR_IDS, np.zeros(16), 9, 1)
            self.client.sync_write(self.MOTOR_IDS, np.full(16, 5), 11, 1)
            self.client.sync_write(self.MOTOR_IDS, np.full(16, self.KP), 84, 2)
            self.client.sync_write(self.MOTOR_IDS, np.full(16, self.KI), 82, 2)
            self.client.sync_write(self.MOTOR_IDS, np.full(16, self.KD), 80, 2)
            self.client.sync_write(self.MOTOR_IDS, np.full(16, self.GOAL_CURRENT), 102, 2)
            self.client.write_desired_pos(self.MOTOR_IDS, current)
            self.commanded_pose = current.copy()
            self.client.set_torque_enabled(self.MOTOR_IDS, True)
            self.torque_enabled = True
            self._interpolate(current, self.open_pose, interpolation_s)
            self.last_seen_time = self.clock()
        except Exception:
            self.shutdown(return_open=False)
            raise

    def shutdown(self, return_open=True, interpolation_s=2.0):
        if self.closed:
            return
        try:
            if return_open and self.torque_enabled and self.commanded_pose is not None:
                self._interpolate(self.commanded_pose, self.open_pose, interpolation_s)
        finally:
            try:
                self.client.set_torque_enabled(self.MOTOR_IDS, False, retries=1)
            finally:
                self.torque_enabled = False
                self.client.port_handler.closePort()
                self.closed = True
```

- [ ] **Step 4: Run controller tests and verify GREEN**

Run `pytest -q tests/test_safe_leap_controller.py`.

Expected: all controller tests pass without accessing `/dev/ttyUSB0`.

- [ ] **Step 5: Commit lifecycle support**

```bash
git add python/gesture_mapping/safe_leap_controller.py \
        python/tests/test_safe_leap_controller.py
git commit -m "feat: add low-force LEAP lifecycle"
```

### Task 3: Add the `--safe-drive` CLI path

**Files:**
- Modify: `python/gesture_mapping/demo_realtime.py:248-411`
- Create: `python/tests/test_demo_realtime_cli.py`

- [ ] **Step 1: Extract argument parsing and write a failing mutual-exclusion test**

The test defines the desired parser API:

```python
import pytest

from gesture_mapping.demo_realtime import build_parser


def test_drive_modes_are_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--drive", "--safe-drive"])


def test_safe_drive_flag_is_opt_in():
    args = build_parser().parse_args(["--camera", "0", "--safe-drive"])
    assert args.safe_drive is True
    assert args.drive is False
```

- [ ] **Step 2: Run CLI tests and verify RED**

Run `pytest -q tests/test_demo_realtime_cli.py`.

Expected: import fails because `build_parser` does not exist.

- [ ] **Step 3: Implement the parser and safe controller factory**

In `demo_realtime.py`, replace inline parser construction with:

```python
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=-1,
                        help="Camera index (default: auto-detect)")
    drive_group = parser.add_mutually_exclusive_group()
    drive_group.add_argument("--drive", action="store_true",
                             help="Drive LEAP Hand with legacy settings")
    drive_group.add_argument("--safe-drive", action="store_true",
                             help="Drive LEAP Hand with low-force safety mode")
    parser.add_argument("--no-display", action="store_true")
    return parser
```

At the start of `main`, use `args = build_parser().parse_args()`.

Construct safe mode without changing legacy mode:

```python
    leap = None
    safe_leap = None
    if args.safe_drive:
        from leap_hand_utils.dynamixel_client import DynamixelClient
        from main import OPEN_POSE
        from gesture_mapping.safe_leap_controller import SafeLeapController

        port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTB8HNYU-if00-port0"
        client = DynamixelClient(list(range(16)), port, 4_000_000)
        safe_leap = SafeLeapController(client, OPEN_POSE)
        safe_leap.start()
        print("[INFO] LEAP Hand connected in low-force safety mode.")
    elif args.drive:
        from main import LeapNode
        try:
            leap = LeapNode()
            print("[INFO] LEAP Hand connected.")
        except OSError as exc:
            print(f"[WARN] Cannot connect: {exc}")
```

- [ ] **Step 4: Route tracking, loss, and cleanup to safe mode**

For valid hand results:

```python
                from leap_hand_utils import leap_hand_utils as lhu
                target = lhu.angle_safety_clip(OPEN_POSE + JOINT_DIR * angles)
                if safe_leap is not None:
                    safe_leap.track(target)
                elif leap is not None:
                    leap.set_leap(target)
```

For no-hand frames:

```python
                if safe_leap is not None:
                    safe_leap.on_tracking_lost()
                elif leap is not None:
                    leap.set_open()
```

In `finally`:

```python
        if safe_leap is not None:
            safe_leap.shutdown(return_open=True)
        elif leap is not None:
            leap.set_open()
            leap.disconnect()
```

- [ ] **Step 5: Run CLI and controller tests**

Run:

```bash
pytest -q tests/test_demo_realtime_cli.py tests/test_safe_leap_controller.py
```

Expected: all tests pass and no serial device is opened.

- [ ] **Step 6: Commit CLI integration**

```bash
git add python/gesture_mapping/demo_realtime.py \
        python/tests/test_demo_realtime_cli.py
git commit -m "feat: add safe MediaPipe drive mode"
```

### Task 4: Regression, documentation, and hardware gate

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document vision-only and safe-drive commands**

Add a concise section containing:

````markdown
### Safe MediaPipe hardware drive

Test vision first, without motor control:

```bash
python gesture_mapping/demo_realtime.py --camera 0
```

After confirming tracking and clearing the hand workspace, enable low-force
control:

```bash
python gesture_mapping/demo_realtime.py --camera 0 --safe-drive
```

Safe mode starts by returning smoothly to the open pose, holds the last target
for 0.5 seconds after tracking loss, then relaxes toward open. Press `Q` or
`Ctrl+C` to return open and disable torque. Keep the hardware power cutoff
within reach during all physical tests.
````

- [ ] **Step 2: Run the full offline suite**

Run:

```bash
cd ~/projects/TuinaDex/Leap_Hand/python
pytest -q
python -m compileall gesture_mapping safe_control.py safe_middle_finger.py
```

Expected: existing offline tests and new tests pass; compileall reports no
syntax errors. Hardware-dependent tests may retain their documented skips.

- [ ] **Step 3: Verify repository scope**

Run:

```bash
cd ~/projects/TuinaDex/Leap_Hand
git diff --check
git status --short
git diff --stat 629c4ab..HEAD
```

Expected: only the design, plan, safe controller, demo integration, tests, and
README changes are tracked. The downloaded model remains ignored. The existing
untracked `python/safe_control.py` and `python/safe_middle_finger.py` are not
accidentally included unless deliberately committed in a separate change.

- [ ] **Step 4: Commit documentation**

```bash
git add README.md
git commit -m "docs: explain safe visual teleoperation"
```

- [ ] **Step 5: Run a vision-only local GUI smoke test**

On the Ubuntu desktop terminal, run:

```bash
conda activate tuinadex_hw
cd ~/projects/TuinaDex/Leap_Hand/python
python gesture_mapping/demo_realtime.py --camera 0
```

Expected: camera window opens, landmarks follow the hand, no serial port is
opened, and `Q` exits cleanly.

- [ ] **Step 6: Stop at the hardware approval gate**

Before running `--safe-drive`, report the complete offline results and ask the
user to clear the workspace, keep the power switch reachable, and confirm the
hand is ready. Only after explicit confirmation run:

```bash
python gesture_mapping/demo_realtime.py --camera 0 --safe-drive
```

Expected: two-second low-force return to open, smooth tracking, a 0.5-second
hold on brief loss, gradual return to open on sustained loss, and torque off
after `Q` or `Ctrl+C`.
