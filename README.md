# FlightStack

**A simulation-first attitude-control stack built to make frame mistakes, unstable gains, and hardware assumptions fail in tests before they reach a motor.**

FlightStack is a small flight-control laboratory: quaternion attitude math, a cascaded attitude/rate controller, rotational rigid-body dynamics, IMU simulation, a complementary estimator, binary HIL framing, and a dependency-free C++20 control core for eventual MCU work.

This is not a wrapper around a flight-control framework and it is not a free-flight autopilot. The current milestone is a reproducible control/estimation testbed with explicit coordinate-frame conventions and Python/C++ parity.

## Verified reference scenario

The default simulation starts at **+20° roll, -12° pitch, +8° yaw** and commands a non-identity target of **+30° roll, +25° pitch, -20° yaw**. At a 500 Hz integration/control step, the current reference gains converge to about **0.66° geodesic attitude error after 4 seconds**.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
flightstack simulate --csv telemetry/reference.csv
```

The regression suite also checks much larger multi-axis rotations, including an initial/target pair separated by roughly 160°, so an incorrect quaternion multiplication order cannot pass merely because the target is identity.

## What is implemented

- scalar-first quaternion algebra (`[w, x, y, z]`)
- exact exponential-map body-rate integration
- exact shortest-path quaternion-log attitude error
- body-frame current-to-target error: `q_current^-1 * q_target`
- cascaded quaternion attitude P loop + 3-axis rate PID
- derivative-on-measurement with low-pass filtering (avoids setpoint derivative kick)
- conditional integration anti-windup and actuator/torque saturation
- full 3x3 symmetric positive-definite inertia validation
- Euler rigid-body rotational dynamics with gyroscopic coupling
- closed-loop telemetry and CSV export
- deterministic IMU noise/bias simulation
- gravity-corrected quaternion complementary estimator
- HIL packet framing with CRC-16/CCITT
- dependency-free C++20 quaternion + attitude-loop primitives
- Python and C++ regression tests
- GitHub Actions quality gates

## Repository map

```text
src/flightstack/
  math/         quaternion/frame primitives
  control/      attitude + rate loops
  sim/          rigid-body plant, scenarios, telemetry
  sensors/      IMU simulation
  estimation/   complementary attitude estimator
  hil/          transport-agnostic binary framing
cpp/
  include/      embedded-oriented C++20 primitives
  tests/        golden-vector parity tests
docs/
  frame-conventions.md
  architecture.md
  safety.md
```

## Why the frame convention matters

FlightStack treats the attitude quaternion as **body -> world** and gyroscope rates as **body-frame rates**. That implies right-multiplicative integration and, critically, a body-frame control error of:

```text
q_error = conjugate(q_current) * q_target
```

Using the opposite order can appear correct when the target is identity but fail badly for arbitrary attitudes. The test suite contains non-identity target regressions specifically to prevent that class of bug. See [`docs/frame-conventions.md`](docs/frame-conventions.md).

## Tests

```bash
pytest
ruff check .
mypy src

cmake -S cpp -B cpp/build
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

Coverage includes quaternion sign equivalence, constant-rate exact rotation, arbitrary current/target frame semantics, PID saturation/unwind behavior, derivative kick prevention, malformed inertia rejection, torque-free spherical dynamics, large-angle closed-loop convergence, deterministic IMU noise, gravity-based tilt convergence, CRC known vectors, packet corruption, and C++ golden vectors.

## Next engineering milestones

The next meaningful additions are hardware-facing rather than more simulation decoration:

- timestamped recorded-IMU playback and bias calibration
- estimator comparison under vibration/bias/dropout
- mixer + motor/actuator saturation model
- serial HIL transport with loop timing/jitter statistics
- MCU rate loop and Python-vs-C++ golden telemetry
- watchdog / arming / failsafe state machine
- restrained bench-rig validation with props removed

## Safety

FlightStack is an educational/research control stack. It is not certified flight software. Hardware work should begin with motors disconnected and then with **propellers removed**. See [`docs/safety.md`](docs/safety.md).

## License

MIT.
