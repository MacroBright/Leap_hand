# Central Safe-Drive Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize every LEAP Hand low-force parameter and let current and future controllers select the complete safe lifecycle through `LeapNode(safe_mode=True)` or a CLI `--safe-drive` flag.

**Architecture:** An immutable `SafetyProfile` in `leap_hand_utils` is the only numeric configuration source. `SafeLeapController` consumes that profile, while `LeapNode` delegates safe-mode lifecycle and command limiting to `SafeLeapController`; normal mode keeps the existing path. CLI programs parse flags explicitly and never rely on hidden `sys.argv` inspection in library code.

**Tech Stack:** Python 3.10, NumPy, argparse, pytest, Dynamixel Protocol 2.0 client

---

## File Map

- Create `python/leap_hand_utils/safety_config.py`: immutable profile and global `SAFE_PROFILE`.
- Create `python/tests/test_safety_config.py`: profile defaults and validation.
- Create `python/tests/test_main_safe_mode.py`: fake-client tests for `LeapNode` delegation and legacy compatibility.
- Modify `python/gesture_mapping/safe_leap_controller.py`: consume a profile instead of local constants.
- Modify `python/tests/test_safe_leap_controller.py`: prove the injected/global profile drives register writes and timing.
- Modify `python/main.py`: add `safe_mode`, safe lifecycle delegation, and CLI flag.
- Modify `python/interactive_control.py`: add `--safe-drive` and pass it to `LeapNode`.
- Modify `python/safe_control.py`: remove duplicated safety constants and accept the consistency flag.
- Modify `python/safe_middle_finger.py`: remove duplicated safety constants and accept the consistency flag.
- Modify `python/tests/test_demo_realtime_cli.py`: retain mutual-exclusion coverage and assert the centralized profile is used.
- Modify `readme.md`: document the single edit location and supported invocation forms.

### Task 1: Add the Immutable Safety Profile

**Files:**
- Create: `python/leap_hand_utils/safety_config.py`
- Create: `python/tests/test_safety_config.py`

- [ ] **Step 1: Write failing profile tests**

```python
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
```

- [ ] **Step 2: Run the test and verify the module is missing**

Run: `cd python && pytest -q tests/test_safety_config.py`

Expected: collection fails with `ModuleNotFoundError: leap_hand_utils.safety_config`.

- [ ] **Step 3: Implement the profile**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyProfile:
    kp: int = 300
    ki: int = 0
    kd: int = 100
    goal_current: int = 150
    max_speed_rad_s: float = 1.0
    startup_seconds: float = 2.0
    shutdown_seconds: float = 2.0
    tracking_loss_seconds: float = 0.5

    def __post_init__(self):
        if min(self.kp, self.ki, self.kd) < 0:
            raise ValueError("PID gains cannot be negative")
        if self.goal_current <= 0:
            raise ValueError("goal_current must be positive")
        if self.max_speed_rad_s <= 0:
            raise ValueError("max_speed_rad_s must be positive")
        if min(self.startup_seconds, self.shutdown_seconds) <= 0:
            raise ValueError("startup and shutdown durations must be positive")
        if self.tracking_loss_seconds < 0:
            raise ValueError("tracking_loss_seconds cannot be negative")


SAFE_PROFILE = SafetyProfile()
```

- [ ] **Step 4: Run the profile tests**

Run: `cd python && pytest -q tests/test_safety_config.py`

Expected: `9 passed`.

- [ ] **Step 5: Commit the profile**

```bash
git add python/leap_hand_utils/safety_config.py python/tests/test_safety_config.py
git commit -m "feat: centralize LEAP safe-drive parameters"
```

### Task 2: Make the Safe Controller Consume the Profile

**Files:**
- Modify: `python/gesture_mapping/safe_leap_controller.py`
- Modify: `python/tests/test_safe_leap_controller.py`

- [ ] **Step 1: Add a failing injected-profile test**

```python
from leap_hand_utils.safety_config import SafetyProfile


def test_startup_uses_the_injected_safety_profile():
    profile = SafetyProfile(kp=210, kd=70, goal_current=90)
    client = FakeClient()
    controller = SafeLeapController(
        client,
        np.zeros(16),
        profile=profile,
        sleep=lambda _: None,
    )
    controller.start(interpolation_s=0.0)
    assert ("sync", 84, tuple([210] * 16)) in client.events
    assert ("sync", 80, tuple([70] * 16)) in client.events
    assert ("sync", 102, tuple([90] * 16)) in client.events
```

- [ ] **Step 2: Verify the new test fails**

Run: `cd python && pytest -q tests/test_safe_leap_controller.py::test_startup_uses_the_injected_safety_profile`

Expected: FAIL because `SafeLeapController.__init__` does not accept `profile`.

- [ ] **Step 3: Replace local constants with the profile**

Update construction and defaults:

```python
from leap_hand_utils.safety_config import SAFE_PROFILE, SafetyProfile


def __init__(
    self,
    client,
    open_pose,
    profile: SafetyProfile = SAFE_PROFILE,
    clock=time.monotonic,
    sleep=time.sleep,
    max_speed_rad_s=None,
    loss_timeout_s=None,
):
    self.profile = profile
    self.max_speed_rad_s = (
        profile.max_speed_rad_s if max_speed_rad_s is None else float(max_speed_rad_s)
    )
    self.loss_timeout_s = (
        profile.tracking_loss_seconds if loss_timeout_s is None else float(loss_timeout_s)
    )
```

Use `self.profile.kp`, `ki`, `kd`, and `goal_current` for register writes. Default `start()` and
`shutdown()` interpolation arguments to `None`, resolving them to the profile durations inside the
method so profile changes cannot be shadowed by definition-time constants.

- [ ] **Step 4: Run all safe-controller tests**

Run: `cd python && pytest -q tests/test_safety_config.py tests/test_safe_leap_controller.py`

Expected: all tests pass, including startup ordering, rate limiting, failure cleanup, and idempotent shutdown.

- [ ] **Step 5: Commit the controller migration**

```bash
git add python/gesture_mapping/safe_leap_controller.py python/tests/test_safe_leap_controller.py
git commit -m "refactor: make safe controller use shared profile"
```

### Task 3: Add Safe Mode to LeapNode

**Files:**
- Modify: `python/main.py`
- Create: `python/tests/test_main_safe_mode.py`

- [ ] **Step 1: Write fake-client lifecycle tests**

Create a fake client that records `connect`, `read_pos`, `sync_write`, torque, position, and port-close
events. Monkeypatch `main.DynamixelClient` before constructing `LeapNode`.

```python
def test_safe_leap_node_prevents_startup_jump(monkeypatch):
    client = FakeClient(current=np.full(16, 0.25))
    monkeypatch.setattr(main, "DynamixelClient", lambda *args: client)
    node = main.LeapNode(safe_mode=True, sleep=lambda _: None)
    position_index = next(i for i, event in enumerate(client.events) if event[0] == "position")
    assert position_index < client.events.index(("torque", True))
    node.disconnect()


def test_safe_leap_node_rate_limits_commands(monkeypatch):
    clock = FakeClock(10.0)
    client = FakeClient(current=np.zeros(16))
    monkeypatch.setattr(main, "DynamixelClient", lambda *args: client)
    node = main.LeapNode(safe_mode=True, clock=clock, sleep=lambda _: None)
    clock.value += 0.1
    node.set_leap(np.ones(16))
    np.testing.assert_allclose(node.curr_pos, np.full(16, 0.1), atol=1e-6)
    node.disconnect()


def test_legacy_leap_node_keeps_normal_limits(monkeypatch):
    client = FakeClient(current=np.zeros(16))
    monkeypatch.setattr(main, "DynamixelClient", lambda *args: client)
    node = main.LeapNode()
    assert node.kP == 600
    assert node.kD == 200
    assert node.curr_lim == 350
    node.disconnect()
```

- [ ] **Step 2: Verify safe-mode tests fail**

Run: `cd python && pytest -q tests/test_main_safe_mode.py`

Expected: FAIL because `LeapNode` does not accept `safe_mode`, `clock`, or `sleep`.

- [ ] **Step 3: Implement delegation without changing normal mode**

Extend the constructor:

```python
def __init__(
    self,
    port=None,
    calib_mode=False,
    safe_mode=False,
    profile=SAFE_PROFILE,
    clock=time.monotonic,
    sleep=time.sleep,
):
    self.safe_mode = bool(safe_mode)
    self._safe_controller = None
```

For safe mode, each candidate port gets a fresh client and `SafeLeapController`; the controller's
`start()` call performs the one and only connection attempt. A failed candidate is shut down before
the next candidate is created. Normal mode retains the existing connection loop and register
sequence. In `set_leap`, safe mode calls `track()` and stores the returned rate-limited pose. In
`disconnect`, safe mode calls `shutdown()`, while normal mode keeps the existing disconnect behavior.

- [ ] **Step 4: Run the lifecycle and existing controller tests**

Run: `cd python && pytest -q tests/test_main_safe_mode.py tests/test_safe_leap_controller.py`

Expected: all tests pass without opening a serial port.

- [ ] **Step 5: Commit the safe LeapNode API**

```bash
git add python/main.py python/tests/test_main_safe_mode.py
git commit -m "feat: expose centralized safe mode through LeapNode"
```

### Task 4: Migrate Command-Line Controllers

**Files:**
- Modify: `python/main.py`
- Modify: `python/interactive_control.py`
- Modify: `python/safe_control.py`
- Modify: `python/safe_middle_finger.py`
- Modify: `python/gesture_mapping/demo_realtime.py`
- Modify: `python/tests/test_demo_realtime_cli.py`
- Create: `python/tests/test_control_cli.py`

- [ ] **Step 1: Write failing CLI parser tests**

Refactor each CLI to expose `build_parser()` and test parsing without running hardware:

```python
from interactive_control import build_parser as interactive_parser
from main import build_parser as main_parser
from safe_control import build_parser as half_parser
from safe_middle_finger import build_parser as middle_parser


def test_optional_controllers_accept_safe_drive():
    assert interactive_parser().parse_args(["--safe-drive"]).safe_drive
    assert main_parser().parse_args(["--safe-drive"]).safe_drive


def test_always_safe_scripts_accept_consistency_alias():
    assert half_parser().parse_args(["--safe-drive"]).safe_drive
    assert middle_parser().parse_args(["--safe-drive"]).safe_drive
```

- [ ] **Step 2: Verify parser tests fail**

Run: `cd python && pytest -q tests/test_control_cli.py`

Expected: collection or assertion failure because the parsers do not exist.

- [ ] **Step 3: Add explicit parser functions and remove copied constants**

For optional controllers:

```python
def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-drive", action="store_true")
    return parser


args = build_parser().parse_args()
hand = LeapNode(safe_mode=args.safe_drive)
```

For always-safe scripts, accept the flag for consistent invocation but construct register values
only from `SAFE_PROFILE`. Do not introduce an unsafe branch. Update the visual controller to pass
`SAFE_PROFILE` explicitly so a source inspection and unit test can prove central ownership.

- [ ] **Step 4: Run CLI and safe-mode tests**

Run: `cd python && pytest -q tests/test_control_cli.py tests/test_demo_realtime_cli.py tests/test_main_safe_mode.py tests/test_safe_leap_controller.py tests/test_safety_config.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit CLI migration**

```bash
git add python/main.py python/interactive_control.py python/safe_control.py \
  python/safe_middle_finger.py python/gesture_mapping/demo_realtime.py \
  python/tests/test_control_cli.py python/tests/test_demo_realtime_cli.py
git commit -m "feat: support shared safe-drive mode across controllers"
```

### Task 5: Document, Regress, and Push the Feature Branch

**Files:**
- Modify: `readme.md`

- [ ] **Step 1: Document the one global edit location**

Add commands for `main.py`, `interactive_control.py`, visual following, half-grasp, and middle-finger
scripts. State that global safety values live only in
`python/leap_hand_utils/safety_config.py`, take effect on the next process start, and do not replace
the physical cutoff.

- [ ] **Step 2: Run static validation**

Run:

```bash
python -m compileall -q python
git diff --check
```

Expected: both commands exit 0 with no output.

- [ ] **Step 3: Run the complete regression suite**

Run: `cd python && pytest -q tests`

Expected: all runnable tests pass; hardware/GPU tests may retain their existing explicit skips.

- [ ] **Step 4: Verify repository boundaries**

Run:

```bash
git status --short
git branch --show-current
git log --oneline origin/main..HEAD
```

Expected: branch is `codex/safe-visual-drive`; only intended files are changed or committed; `main`
is untouched.

- [ ] **Step 5: Commit documentation**

```bash
git add readme.md
git commit -m "docs: explain global safe-drive configuration"
```

- [ ] **Step 6: Push without merging**

Run: `git push origin codex/safe-visual-drive`

Expected: remote branch advances successfully. Do not open or merge a pull request unless the user
requests it separately.
