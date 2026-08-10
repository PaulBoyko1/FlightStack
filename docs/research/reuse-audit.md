# Reuse audit

## Current FlightStack implementation

The Python 6DOF reference plant, TOML configuration format, CTBR controller
seam, mixer, track/race event state, collision helpers, human/classical pilots,
replay format, Rust runtime code, and Three.js scene/client code are
FlightStack-native implementation.  No upstream source files or visual assets
have been copied into this repository.

The implementation was informed by the narrow, pinned references in
[`SOURCE_MANIFEST.md`](SOURCE_MANIFEST.md), especially their physical starting
parameters, motor/force pipeline, swept gate-crossing rationale, and CTBR
control boundary.  Reference influence is not code vendoring: FlightStack
retains its own frames, tests, state/configuration types, equations, and
architecture.

## Declared commodity dependencies

- **NumPy** (BSD-3-Clause) provides Python numerical arrays.
- **aiohttp** (Apache-2.0) provides the local HTTP/WebSocket server.
- **Three.js** `0.185.0` (MIT) provides browser rendering.  The drone and gate
  meshes in `web/src/main.ts` are procedural FlightStack code; no third-party
  meshes, textures, or environment assets are shipped.
- **Vite**, **TypeScript**, and `@types/three` are web build/type tooling.
- **Serde** and the Rust `toml` crate are workspace dependencies for the Rust
  canonical/configuration path.

Their license summaries and direct-use scope are listed in
[`THIRD_PARTY.md`](../../THIRD_PARTY.md).  Dependency source is installed by
package managers and is not committed as FlightStack source.

## Deferred reuse

- A maintained PPO package will be selected only after the environment and
  evaluation contract are implemented.
- A single vectorized training backend will be chosen only after it can be
  compared against the reference vehicle with focused parity scenarios.
- Rapier/Parry remains deferred; the current small gate-frame/ground collision
  layer is FlightStack-owned and deliberately narrow.

Any direct source adaptation must add the upstream repository/path, pinned
commit, license, required attribution, and a short explanation of the adapted
scope to this document and `THIRD_PARTY.md` before it is committed.  A model,
dataset, mesh, texture, or checkpoint has its own provenance/license review;
it is not covered merely by the repository license of a related project.
