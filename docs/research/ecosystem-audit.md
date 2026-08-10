# Ecosystem audit

This is a compact decision record, not a source dump.  FlightStack reuses
mature permissively licensed infrastructure where it reduces risk while keeping
its vehicle equations, frame contracts, pilots, and experiments original.

| Project | Pinned reference | License | Initial value | Decision |
| --- | --- | --- | --- | --- |
| Elodin AI Grand Prix | `13f9f9e` | Apache-2.0 | Motor/force pipeline and sensor architecture reference | Inspected for reference; no source copied |
| LSY Drone Racing | `9ecb1cb` | MIT | Swept gate-crossing and training-evaluation design | Inspected; independently implemented race contracts |
| Crazyflow | `58e8fb4` | MIT | Vectorized vehicle/training patterns | Deferred until a parity-backed training backend is selected |
| gym-pybullet-drones | `e712698` | MIT | Gymnasium API comparison | Reference only; never authoritative physics |
| RLtools | `b32f998` | MIT | CPU continuous-control research option | Deferred |
| SimpleFlight | `f5ae8fc` | MIT | CTBR learned-policy interface rationale | Interface reference only |
| Three.js | `0.185.0` | MIT | Browser rendering | Declared web dependency; no vendoring |
| Rapier/Parry | current pinned crate release | Apache-2.0 | Collision geometry | Deferred while the narrow FlightStack collision layer is sufficient |

The `flightstack_5in` values are a consciously approximate 5-inch reference
model.  They are not presented as measurements of a specific physical craft.

## Implemented reuse decisions

- The local browser client uses Three.js only for rendering.  `FlightSession`
  remains the fixed-step state/race/collision authority; the client sends
  normalized inputs and displays telemetry.
- The current Python and Rust 6DOF implementations are FlightStack-native.
  Elodin's pinned simulator was inspected as a physics/configuration reference;
  no Elodin source is vendored.
- The current swept gate/race implementation is FlightStack-native.  LSY Drone
  Racing informed the choice of robust plane crossing and an explicit race
  state, but its source is not copied.
- Gymnasium and Stable-Baselines3 are optional `.[train]` dependencies for the
  FlightStack-native state racing environment, PPO plumbing, and metadata
  checked checkpoint adapter.  No quality checkpoint, model weight, or dataset
  ships; the local smoke and 10,000-step PPO runs did not complete their seeded
  technical-eight evaluation.
- `ReferenceVectorEnv` batches the exact Python reference environment.  The
  high-throughput/JAX backend decision remains deferred pending parity work.
