# Safe Visual Drive Design

## Goal

Add an opt-in safe hardware mode to the MediaPipe teleoperation demo without
changing the behavior of the existing `--drive` option. The new mode must use
the conservative gains already validated on the physical LEAP Hand and must
handle startup, temporary tracking loss, shutdown, and exceptions safely.

## User Interface

The new command is:

```bash
python gesture_mapping/demo_realtime.py --camera 0 --safe-drive
```

`--drive` and `--safe-drive` are mutually exclusive. Running without either
flag remains vision-only and never opens the motor serial port.

## Architecture

Create a `SafeLeapController` in a focused module under
`python/gesture_mapping/`. It owns the Dynamixel client, safe configuration,
current commanded pose, interpolation, velocity limiting, tracking-loss state,
and shutdown sequence. The MediaPipe demo supplies absolute 16-motor target
poses and timestamps; it does not write directly to the motors in safe mode.

The existing `LeapNode` and original `--drive` path remain unchanged.

## Safe Hardware Configuration

- Protocol: Dynamixel 2.0 at 4 Mbps, IDs 0-15.
- Operating mode: current-based position mode (`5`).
- Position gain: `kP=300`.
- Integral gain: `kI=0`.
- Derivative gain: `kD=100`.
- Goal current: `150`.
- The measured current position is written as the initial goal before torque is
  enabled, preventing an initialization jump.

## State and Motion Flow

Startup:

1. Connect and read all 16 current positions.
2. Reject invalid, incomplete, or non-finite readings.
3. Configure motors while torque is disabled.
4. Write the measured position as the goal and enable torque.
5. Interpolate from the measured position to the recorded open pose over two
   seconds.

Tracking:

- Valid MediaPipe results are converted through the existing mapper.
- Each outgoing pose is clipped by the existing joint safety limits.
- A per-frame velocity limit constrains how far any motor target can change.
- The controller records the time of the last valid hand observation.

Tracking loss:

- For the first 0.5 seconds, retain the last commanded target so a single
  dropped frame does not cause movement.
- After 0.5 seconds, advance gradually toward the recorded open pose using the
  same velocity limiter.
- A newly detected hand resumes tracking from the current commanded pose,
  avoiding a discontinuous jump.

Shutdown:

- On normal exit or `Ctrl+C`, interpolate to the open pose and disable torque.
- On a communication/configuration exception, attempt to disable torque
  immediately and close the port; do not attempt additional movement.
- Cleanup must be idempotent so nested error paths cannot leave the port open.

## Testing

Automated tests use a fake Dynamixel transport and never access `/dev/ttyUSB0`.
They verify:

- current pose is set before torque enable;
- safe gains and goal current are applied;
- startup interpolation ends at the open pose;
- short tracking loss holds the last target;
- loss beyond 0.5 seconds moves toward open;
- per-step motion is velocity limited;
- shutdown returns open and disables torque;
- exceptions still disable torque and close the connection;
- `--drive` and `--safe-drive` cannot be selected together.

After automated regression passes, hardware validation runs vision-only first,
then `--safe-drive` with the workspace clear and power cutoff immediately
available. No hardware test uses the original high-force mode.

## Non-goals

- Do not change the original `LeapNode` parameters or `--drive` behavior.
- Do not add RealSense or HaMeR support.
- Do not recalibrate recorded poses or joint gains.
- Do not commit the downloaded MediaPipe model.
