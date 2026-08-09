# Engineering decisions

## 2026-08 — Reference 6DOF path begins in Python

### Decision

Extend the existing Python reference model with a deterministic 6DOF vehicle
before introducing a renderer, physics engine, or vectorized training backend.

### Evidence

The existing Python/C++ tests already establish exact quaternion and controller
contracts.  A small, inspectable 6DOF extension provides executable parity
fixtures and protects those contracts before adding more runtimes.

### Alternatives considered

Making PyBullet/Rapier authoritative, or beginning with browser-side motion.

### Impact

The initial vertical slice has one transparent source of physical truth.  A
Rust runtime and a faster training backend will be judged against it rather
than inventing incompatible conventions.

## 2026-08 — Reconcile 5-inch configuration with the vetted source pack

### Decision

Use rounded generic 5-inch mass, inertia, motor geometry, thrust, and damping
values adapted from the vetted Elodin AI Grand Prix configuration, while
retaining FlightStack's own motor ordering, exact exponential motor update, and
body-frame linear drag model.

### Evidence

The source pack identifies `sim/config.py` and `sim/physics.py` at Elodin's
Apache-2.0 pinned commit as the relevant physical reference.  Its `0.65 kg`,
`[0.002, 0.002, 0.0035] kg m²`, approximately `0.11 m` arm-radius model is a
better documented starting point than a hand-estimated component spacing.

### Alternatives considered

Keeping the initial approximate geometry, or adopting Elodin's semi-implicit
motor update and world-frame quadratic drag verbatim.

### Impact

The reference configuration has traceable starting values without claiming
hardware measurement.  Exact motor updates remain analytically testable; the
chosen simple body-frame drag stays explicit and compatible with the canonical
FLU vehicle contract.

## 2026-08 — Keep NumPy below 2.5 while Python 3.11 is supported

### Decision

Constrain NumPy to `<2.5`.

### Evidence

NumPy 2.5 publishes stubs using Python 3.12-only syntax while FlightStack's
strict type check is intentionally configured for Python 3.11.

### Impact

Fresh Python 3.11 CI installs remain type-checkable; this is not a runtime
physics constraint.
