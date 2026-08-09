# Reference vehicle physics

FlightStack's first full vehicle model is a deterministic, fixed-step 6DOF
reference implementation in `flightstack.sim.vehicle`.  It extends—rather than
replaces—the focused rotational plant used by the original controller tests.

## Frames

| Quantity | Convention |
| --- | --- |
| World | Right-handed local ENU-like frame: +X east, +Y north, +Z up |
| Body | FLU: +X forward, +Y left, +Z up |
| Attitude | `[w, x, y, z]`, body-to-world |
| Angular velocity / torque | Body frame |
| Rotor thrust | Positive body +Z |

The quaternion update remains right-multiplicative:

```text
q_next = q_body_to_world * Exp(omega_body * dt)
```

## State and integration

The canonical state holds time, world position/velocity, a body-to-world
quaternion, body rate, and four individual motor thrust states.  The reference
integrator is deterministic and uses this ordering each fixed step:

1. Clamp and exponentially update each motor thrust using separate rise/fall
   time constants.
2. Compute thrust, gravity, body-frame linear drag (transformed to world), and
   externally supplied world force; update velocity and position.
3. Compute arm moments, reaction yaw moments, angular drag, gyroscopic coupling
   and external torque; update body rate and integrate orientation exactly.

This is deliberately transparent rather than an implicit general-purpose rigid
body engine.  The individual equations are regression tested for free fall,
hover, motor response, force/moment signs, saturation, degradation, and
determinism.

## Mixer and CTBR boundary

Every pilot emits a collective-thrust/body-rate (CTBR) command.  The existing
FlightStack derivative-on-measurement rate PID generates a body torque, then a
four-by-four allocation matrix maps `[collective, roll, pitch, yaw]` to motor
thrust targets.  Motor saturation is surfaced in the result; it is never
silently hidden.

The reference vehicle is configured solely by
[`config/vehicles/flightstack_5in.toml`](../config/vehicles/flightstack_5in.toml).
Its parameters are estimated/adapted/tuned starter values, not measured claims.
