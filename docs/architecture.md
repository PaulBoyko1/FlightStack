# Architecture

FlightStack keeps the physical/reference model separate from presentation and
from high-level pilot decisions.  The important design choice is that the
browser is never a second physics engine.

## Authoritative interactive path

```text
keyboard / gamepad
        |
        v
Three.js browser client -- JSON WebSocket --> FlightSession (Python)
                                                |
              +---------------------------------+---------------------------+
              |                                 |                           |
              v                                 v                           v
       HumanPilot                        ClassicalRacePilot   optional LearnedPolicyPilot
              \                                 |                           /
               +---------- CTBR PilotCommand: collective N, body-rate rad/s -+
                                                |
                                                v
              BodyRateController (existing derivative-on-measurement PID)
                                                |
                                                v
                    QuadMixer (geometry/spin-derived 4 x 4 allocation)
                                                |
                                                v
           Multirotor (motor lag -> forces/moments -> deterministic 6DOF step)
                                                |
                     +--------------------------+-------------------------+
                     |                          |                         |
                     v                          v                         v
             RaceState / gates            collision helpers         ReplayRecorder
                     |                          |                         |
                     +--------------------------+-------------------------+
                                                |
                                                v
                          authoritative state + race telemetry -> browser
```

`FlightSession` owns the local interactive run.  Its physics loop only advances
the `FixedStepRuntime` in exact 0.002 s steps; it uses accumulated wall time
with a bounded catch-up budget rather than changing the integration timestep.
It sends state packets every 15 completed physics steps.  The session owns
race progression, collision decisions, pilot selection, and replay capture.

The browser's responsibilities are intentionally limited to input capture,
WebSocket transport, Three.js rendering, camera selection, HUD/minimap, and
technical display.  It maps FlightStack world vectors `[x, y, z]` to Three's
Y-up renderer with one documented conversion `[x, z, -y]`; no frame conversion
or simulation decision is hidden in the scene graph.

## Canonical data contracts

The Python reference definitions live in `flightstack.sim.vehicle`:

| Contract | Meaning |
| --- | --- |
| `VehicleConfig` | Validated TOML model: mass/inertia, geometry/spin directions, motor limits/lag, drag, gravity, and rate-loop limits. |
| `FlightState` | Simulation time, world position/velocity, body-to-world quaternion, body rate, and four motor thrust states. |
| `PilotCommand` | Shared CTBR command: non-negative collective thrust in N plus three target body rates in rad/s. |
| `Disturbance` | Explicit world force, body torque, wind, and per-motor efficiency. It makes test/experiment disturbances reproducible rather than implicit. |

The canonical frames are deliberately fixed:

- world: right-handed local ENU-like, `+Z` up;
- body: FLU, `+X` forward, `+Y` left, `+Z` up;
- quaternion: scalar-first `[w, x, y, z]`, body-to-world;
- rates, torques, and rotor arm geometry: body-frame quantities.

Named `wxyz_to_xyzw` and `xyzw_to_wxyz` adapters handle storage changes at
external boundaries.  They do not silently reinterpret the rotation.  See
[frame conventions](frame-conventions.md).

## Reference vehicle/control pipeline

`Multirotor.step_command` executes this sequence at a fixed step:

1. `BodyRateController` clamps the CTBR body-rate target and produces a body
   torque using the existing derivative-on-measurement, conditional
   anti-windup PID.
2. `QuadMixer` maps collective thrust and torque to four motor targets from
   the configured rotor positions and spin directions.  It reports saturation
   explicitly.
3. Each motor follows its own exact first-order rise/fall response.
4. Rotor thrust and reaction torque, drag, gravity, optional external force,
   optional wind, and optional motor efficiency effects are accumulated.
5. Translation is updated semi-implicitly; body angular rate is integrated
   with Euler rigid-body dynamics and the attitude uses the exact body-rate
   exponential update.

The configuration file is the source of reference parameters for Python and
Rust.  Values are transparent starting values, not a calibration claim.  The
more detailed model description is in [physics.md](physics.md).

## Pilots and races

Every implemented high-level pilot uses the same CTBR boundary:

- `HumanPilot` applies a continuous deadzone/expo curve to normalized sticks,
  maps throttle around physical hover, and respects configured body-rate
  limits.
- `ClassicalRacePilot` is a conservative position/velocity guidance baseline.
  It points its desired body `+Z` along a requested thrust vector and then
  emits a CTBR request.  It is not a trajectory optimizer.
- `LearnedPolicyPilot` loads a Stable-Baselines3 PPO checkpoint only after its
  required metadata sidecar confirms the archive content hash, action and
  observation schemas, vehicle configuration, complete AI-configuration hash,
  and control period.  It then emits CTBR through the same rate controller,
  mixer, and motor path.  Missing or incompatible checkpoints do not fall back
  to Human or Classical behavior.

Tracks are JSON data.  `RaceState` accepts only the next ordered gate event;
swept segment/plane intersection happens in the gate's local basis, so a fast
pass is not missed because neither endpoint is near the gate center.  Ground
and gate-frame helpers are separate from aperture crossing.  A session ends on
the first configured collision and must be reset.

## State-racing environment, training, and experiments

`FlightStackRaceEnv` is a state-based environment over the same Python
`FixedStepRuntime` and `RaceState` contracts.  A normalized four-value policy
action is held across ten 2 ms physics steps, then maps into the shared CTBR
command.  Its 27-value observation uses body-local gate/vehicle quantities and
previous-action history rather than a raw quaternion; reward terms are
versioned and exposed in environment information/telemetry.

`LearnedPolicyPilot` uses that same configured 20 ms control period when it is
called from a 2 ms interactive or headless runtime: it holds the last CTBR
command between inference updates.  That train/deploy timing match prevents a
checkpoint trained at 50 Hz from being queried at 500 Hz merely because the
reference physics is faster.

The NumPy core can be wrapped in a native Gymnasium environment when the
optional `.[train]` dependencies are installed.  `flightstack train` delegates
PPO/MLP optimization to Stable-Baselines3 and writes a model plus metadata;
`flightstack serve --policy ...` validates and exposes a compatible checkpoint
to the browser.  Compatibility is not a quality signal: no recommended learned
checkpoint ships, and the locally exercised smoke and 10,000-step PPO runs did
not complete their seeded technical-eight evaluation.

`ReferenceVectorEnv` batches independent instances of the exact Python
environment for deterministic batch/paired work.  It is intentionally not a
JAX implementation or a high-throughput replacement physics backend.  The
headless experiment runner similarly uses the canonical Python plant and can
write summary, telemetry, and replay JSON for one named scenario.  Paired
evaluation and robustness-grid construction are explicit APIs rather than
implicit benchmark claims.

## Replay and telemetry

`ReplayRecorder` captures authoritative fixed-step `FlightState`, pilot kind,
CTBR command, compact race data, and event mappings as versioned JSON
(`flightstack-replay-v1`).  Interactive recordings are session-owned and in
memory; its Python `write()` API can persist a recording.  The headless
experiment runner emits replay JSON alongside result and sampled-telemetry
artifacts.  The browser has no replay download control yet.

Telemetry packets carry canonical state, motor thrust, current command, race
status, and track geometry.  They are intended as inspection data, not a
realtime actuator protocol.

## Other runtime boundaries

### Python laboratory modules

The original focused rotational plant remains in `flightstack.sim.rigid_body`
for quaternion/control regression tests.  The `sensors`, `estimation`, and
`hil` packages provide deterministic IMU simulation, a complementary attitude
estimator, and transport-agnostic CRC-framed HIL packets.  They are useful
laboratory components, but they are not yet wired into the interactive 6DOF
server as a full sensor/estimator loop.

### C++20 primitives

`cpp/` contains dependency-free quaternion and attitude-loop primitives with
golden-vector tests.  It is a portability/control seam, not a firmware target
or a full 6DOF runtime.

### Rust reference runtime

`rust/flightstack-core` defines the canonical state, CTBR command, quaternion
helpers, and validated vehicle configuration.  `rust/flightstack-sim`
implements the deterministic motor/mixer/6DOF path.  The workspace embeds the
same tracked vehicle TOML for the convenience reference loader, avoiding a
second hand-maintained parameter set.  The interactive server does not yet
dispatch physics to Rust, and a cross-language trajectory-parity suite remains
to be added.

## What this architecture intentionally does not do

- No browser-side authority or hidden browser physics.
- No generic rigid-body engine replacing the multirotor equations.
- No raw-motor learned policy default.
- No JAX backend, curriculum scheduler, or claim that an exported PPO model is
  a quality racing policy.
- No claim that a simulator configuration is an identified physical vehicle.
- No hardware transport or arming/failsafe authority.
