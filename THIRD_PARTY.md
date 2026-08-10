# Third-party notices

FlightStack itself is MIT licensed.  The repository does **not** vendor
third-party source code, meshes, textures, datasets, or model checkpoints.
Generated dependency directories and the production web bundle are ignored by
Git; package managers obtain their own copies under their respective licenses.

## Direct declared dependencies

| Component | Package | Declared use | License |
| --- | --- | --- |
| Python reference runtime | NumPy | vectors, matrices, deterministic numerical operations | BSD-3-Clause |
| Python interactive server | aiohttp | local HTTP/WebSocket transport | Apache-2.0 |
| Browser renderer | Three.js `0.185.0` | scene graph and WebGL rendering | MIT |
| Browser development | Vite | web build/tooling | MIT |
| Browser development | TypeScript | type checking/build input | Apache-2.0 |
| Browser development | `@types/three` | Three.js TypeScript declarations | MIT |
| Rust reference workspace | Serde | configuration/data serialization support | MIT OR Apache-2.0 |
| Rust reference workspace | `toml` | TOML configuration parsing | MIT OR Apache-2.0 |

Exact resolved package versions and transitive dependencies are captured by
`web/pnpm-lock.yaml`, `rust/Cargo.lock`, and the Python environment used for a
run.  Shipping a binary, browser bundle, checkpoint, or dataset may introduce
additional notice obligations; update this file before adding any such artifact
to the repository or release output.

## Pinned implementation references

The repository includes documentation identifying permissively licensed
upstream *references* (for example Elodin AI Grand Prix, LSY Drone Racing,
Crazyflow, and SimpleFlight).  Their code and assets are not copied into
FlightStack.  The pinned commit, license, intended use, and no-copy boundary
are maintained in [SOURCE_MANIFEST.md](docs/research/SOURCE_MANIFEST.md) and
[reuse-audit.md](docs/research/reuse-audit.md).

GPL-only or unlicensed implementation is not copied into the MIT FlightStack
core.  If future work directly adapts source, it must record the upstream path,
commit, license, attribution/notice requirement, and scope of changes here and
in the reuse audit.
