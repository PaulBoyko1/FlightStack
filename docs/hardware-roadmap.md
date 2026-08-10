# Hardware roadmap

## Current boundary

FlightStack is currently a desktop reference simulator and control laboratory.
Its 5-inch TOML values are rounded reference parameters, not an identified
airframe; `flightstack serve`, the Rust simulator, and the browser client must
not command real motors.  There is no hardware actuator transport, arming
state machine, failsafe, calibrated sensor stack, or flight-tested controller
in this revision.

The existing HIL module is a transport-agnostic CRC-framed packet format and
the C++20 module is a small control/math portability seam.  Neither one
constitutes an MCU flight application.

## Staged path

| Stage | Goal | Current status | Exit evidence |
| --- | --- | --- | --- |
| 0. Desktop contracts | Keep frames, CTBR, mixer, motor model, and race semantics explicit and regression-tested. | Implemented reference/test scope. | Green Python/C++/Rust/web checks and reviewed frame/config changes. |
| 1. Vehicle identification | Replace approximate mass/inertia/geometry/thrust/lag values with measured values and uncertainty ranges. | Not started. | Versioned measurement procedure, raw data, fit script, updated TOML/provenance. |
| 2. Recorded-data loop | Add timestamped sensor/actuator log playback and compare estimator/controller behavior against recordings. | Not started; deterministic synthetic IMU exists. | Repeatable playback fixtures and error/timing reports. |
| 3. Bench interface | Design a narrow, authenticated hardware adapter with explicit timing, arming, heartbeat, and failsafe behavior. | Not started. | Reviewed interface, fault-injection tests, and independent kill path. |
| 4. Prop-off bench tests | Verify orientation signs, mixer signs, motor mapping, sensor calibration, and watchdog behavior without propellers. | Not started. | Signed checklist and logged results for every motor and axis. |
| 5. Restrained low-risk tests | Validate only after the prior stages and local rules/site conditions are satisfied. | Not started. | Conservative envelope, abort criteria, telemetry/replay, and operator review. |
| 6. Sim-to-real iteration | Re-identify the model, document mismatch, and rerun deterministic/robustness evaluation. | Not started. | Versioned before/after artifacts and explicit residual limitations. |

## Non-negotiable safeguards

- Start with motors electrically disconnected; use propellers removed for all
  early bench work.
- Treat coordinate-frame and motor-order verification as measurements, not as
  assumptions copied from a simulator.
- Require an independent means to remove motor power and a clear abort path.
- Never let a browser tab, development server, or unvalidated policy become the
  only safety mechanism.
- Keep every calibration, firmware, configuration, test condition, and failure
  report versioned with the code that produced it.

The broader safety context is in [safety.md](safety.md).  This roadmap does not
grant operational approval or replace local law, airspace rules, manufacturer
requirements, or qualified safety review.
