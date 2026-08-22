# Central Safe-Drive Configuration Design

**Date:** 2026-08-22

**Scope:** LEAP Hand Python control code

**Branch:** `codex/safe-visual-drive`

## Goal

Provide one authoritative location for all low-force safety parameters. Existing and future
hardware-control scripts must import this profile instead of copying gain, current, timing, or
rate-limit constants. Normal/legacy drive behavior remains unchanged.

## User Interface

Command-line programs that offer both modes expose `--safe-drive` and pass the resulting boolean
to their controller construction. Library code does not inspect `sys.argv`.

The common programmatic entry point is:

```python
LeapNode(safe_mode=True)
```

Scripts that are intentionally always safe may continue to run without a flag. They still import
the same central profile, and may accept `--safe-drive` as a documented no-op alias for command-line
consistency.

## Configuration Ownership

Add `python/leap_hand_utils/safety_config.py` with an immutable `SafetyProfile` and a single
`SAFE_PROFILE` instance. It owns:

- position-loop gains (`kp=300`, `ki=0`, `kd=100`);
- goal current (`150`);
- maximum commanded joint speed (`1.0 rad/s`);
- startup and shutdown interpolation durations (`2.0 s` each);
- tracking-loss grace period (`0.5 s`).

Normal gains and current limits remain in the normal-drive configuration used by `LeapNode`.
Changing `SAFE_PROFILE` changes every safe-mode consumer on the next process start.

## Driver Behavior

`LeapNode(safe_mode=False)` preserves the current legacy behavior.

`LeapNode(safe_mode=True)` must:

1. connect and validate a finite 16-joint measured pose;
2. disable torque before changing operating mode, gains, and current limit;
3. write the measured pose as the first goal before enabling torque;
4. enable torque and interpolate from the measured pose to the validated open pose;
5. rate-limit subsequent 16-joint position commands using a monotonic clock;
6. on shutdown, interpolate to open when possible, disable torque, and close the port;
7. on a write or initialization error, prioritize disabling torque and closing the port.

The implementation must not duplicate register-write constants across application scripts.

## Visual-Control Behavior

`gesture_mapping/safe_leap_controller.py` continues to own camera-tracking semantics. It imports
all numeric values from `SAFE_PROFILE`. On tracking loss it holds the last command for the configured
grace period, then rate-limits movement toward open.

Tracking-loss behavior is not automatically applied to non-visual scripts because those scripts
have no camera-presence signal.

## Existing Script Migration

- `gesture_mapping/demo_realtime.py`: keeps mutually exclusive `--drive` and `--safe-drive` modes
  and consumes the central profile through the safe controller.
- `interactive_control.py`: adds `--safe-drive` and constructs `LeapNode(safe_mode=...)`.
- `main.py`: accepts `--safe-drive` when executed as a CLI and exposes `LeapNode(safe_mode=True)`
  for future Python scripts.
- `safe_control.py` and `safe_middle_finger.py`: remove duplicated constants and use
  `SAFE_PROFILE`; their safe behavior remains unconditional.

Calibration and measurement utilities are excluded unless they intentionally drive through
`LeapNode`. Direct low-level `DynamixelClient` access does not automatically become safe merely
because a process has a `--safe-drive` argument.

## Error Handling and Safety Boundaries

- Reject non-finite values and arrays that are not exactly 16 joints before sending commands.
- Reject non-positive speed limits or interpolation durations and negative loss timeouts.
- Make shutdown idempotent so repeated cleanup cannot re-enable torque or reopen the port.
- Never claim a software limit is a certified physical safety mechanism. A reachable hardware
  power cutoff and a cleared workspace remain required.
- Do not silently fall back from requested safe mode to legacy mode.

## Tests and Acceptance Criteria

Unit tests use injected fake clients and clocks; they never access serial hardware.

Acceptance requires tests proving:

1. every safe consumer reads the same profile values;
2. safe initialization writes the measured pose before enabling torque;
3. safe commands are rate-limited;
4. invalid joint targets are rejected without a motor write;
5. initialization and write failures disable torque and close the port;
6. normal `LeapNode()` retains existing gains and behavior;
7. CLI parsing accepts `--safe-drive` where documented and rejects conflicting modes;
8. the complete existing Leap Hand test suite still passes.

Hardware smoke testing is separate and manual: workspace clear, cutoff reachable, start from torque
off, run open-only first, and only then test a low-amplitude gesture.

## Version-Control Delivery

Implementation remains on `codex/safe-visual-drive`, is committed with an explicit safety-focused
message, and may be pushed for review. It must not be merged into `main` as part of this work.
