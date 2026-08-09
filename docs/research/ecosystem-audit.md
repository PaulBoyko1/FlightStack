# Ecosystem audit

This is a compact decision record, not a source dump.  FlightStack will reuse
mature permissively licensed infrastructure where it reduces risk while keeping
its vehicle equations, frame contracts, pilots, and experiments original.

| Project | Pinned reference | License | Initial value | Decision |
| --- | --- | --- | --- | --- |
| Elodin AI Grand Prix | `13f9f9e` | Apache-2.0 | Motor/force pipeline and sensor architecture reference | Inspect; no source copied |
| LSY Drone Racing | `9ecb1cb` | MIT | Swept gate-crossing and training-evaluation design | Inspect; independently implement contracts |
| Crazyflow | `58e8fb4` | MIT | Vectorized vehicle/training patterns | Defer until the reference environment is stable |
| gym-pybullet-drones | `e712698` | MIT | Gymnasium API comparison | Reference only; never authoritative physics |
| RLtools | `b32f998` | MIT | CPU continuous-control research option | Deferred |
| SimpleFlight | `f5ae8fc` | MIT | CTBR learned-policy interface rationale | Interface reference only |
| Three.js | current pinned package release | MIT | Browser rendering | Planned dependency; no vendoring |
| Rapier/Parry | current pinned crate release | Apache-2.0 | Later collision geometry | Deferred until race collision phase |

The `flightstack_5in` values are a consciously approximate 5-inch reference
model.  They are not presented as measurements of a specific physical craft.
