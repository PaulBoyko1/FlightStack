# Engineering decisions

## 2026-08: Reference 6DOF path begins in Python

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

## 2026-08: Reconcile 5-inch configuration with the vetted source pack

### Decision

Use rounded generic 5-inch mass, inertia, motor geometry, thrust, and damping
values adapted from the vetted Elodin AI Grand Prix configuration, while
retaining FlightStack's own motor ordering, exact exponential motor update, and
body-frame linear drag model.

### Evidence

The source pack identifies `sim/config.py` and `sim/physics.py` at Elodin's
Apache-2.0 pinned commit as the relevant physical reference.  Its `0.65 kg`,
`[0.002, 0.002, 0.0035] kg m^2`, approximately `0.11 m` arm-radius model is a
better documented starting point than a hand-estimated component spacing.

### Alternatives considered

Keeping the initial approximate geometry, or adopting Elodin's semi-implicit
motor update and world-frame quadratic drag verbatim.

### Impact

The reference configuration has traceable starting values without claiming
hardware measurement.  Exact motor updates remain analytically testable; the
chosen simple body-frame drag stays explicit and compatible with the canonical
FLU vehicle contract.

## 2026-08: Keep NumPy below 2.5 while Python 3.11 is supported

### Decision

Constrain NumPy to `<2.5`.

### Evidence

NumPy 2.5 publishes stubs using Python 3.12-only syntax while FlightStack's
strict type check is intentionally configured for Python 3.11.

### Impact

Fresh Python 3.11 CI installs remain type-checkable; this is not a runtime
physics constraint.

## 2026-08: Make the local Python session authoritative for interaction

### Decision

Run the interactive 6DOF reference simulation, race state, collision checks,
pilot selection, and replay capture in Python `FlightSession`; use the
Three.js/Vite browser only as a WebSocket client and renderer.

### Evidence

The source pack explicitly recommends Three.js as a browser renderer and
requires the browser not to become the authoritative simulation backend.  The
existing Python reference model already has the canonical CTBR, frame, and
fixed-step contracts needed to make one source of truth practical.

### Alternatives considered

Browser-side physics, or switching the interactive authority to a generic
rigid-body engine.

### Impact

The scene maps FlightStack's Z-up coordinates to Three's Y-up basis at one UI
boundary.  A future Rust server or higher-throughput training backend must
demonstrate parity before it replaces this reference path.

## 2026-08: Reserve, rather than fake, the learned-pilot mode

### Decision

Keep `PilotKind.LEARNED` and the common CTBR seam, but reject learned-mode
selection until a metadata-validated trained checkpoint and inference adapter
exist.

### Evidence

The source pack calls for real PPO training/evaluation and the same CTBR
semantics for human, classical, and learned pilots.  A placeholder policy or a
silent fallback would make UI behavior and experimental claims misleading.

### Impact

The classical pilot remains a deterministic baseline only.  The UI can expose
the intended comparison, while the server clearly reports that no learned model
is installed.
