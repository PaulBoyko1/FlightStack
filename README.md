# FlightStack

[![CI](https://github.com/PaulBoyko1/FlightStack/actions/workflows/ci.yml/badge.svg)](https://github.com/PaulBoyko1/FlightStack/actions/workflows/ci.yml)

**FlightStack is a simulation-first flight-control and drone-racing laboratory.**
It keeps one explicit frame contract and a deterministic multirotor reference
model so that control, race, UI, training, and experiment changes can be
checked before they are treated as hardware work.

> **Safety/status:** this repository is research software, not certified flight
> software and not an authorization to fly a real vehicle.  The current
> reference configuration is an approximate generic 5-inch quad, not a measured
> FlightStack airframe.  See [Safety](docs/safety.md) and the
> [hardware roadmap](docs/hardware-roadmap.md).

## What is implemented now

- A deterministic Python 6DOF reference quad: translation, rotation, gravity,
  body-frame drag, individual motor thrust states, motor rise/fall lag, a
  geometry-derived mixer, and reproducible disturbances.
- A single TOML vehicle definition at
  [`config/vehicles/flightstack_5in.toml`](config/vehicles/flightstack_5in.toml),
  consumed by the Python runtime and embedded by the Rust reference runtime.
- A shared collective-thrust/body-rate (CTBR) command seam:
  `pilot -> rate PID -> mixer -> motors -> plant`.
- Human stick shaping, a conservative deterministic classical gate pilot, a
  data-driven technical-eight track, swept gate-plane crossing, and ground /
  gate-frame collision events.
- An authoritative local Python server with a fixed 2 ms physics step, JSON
  WebSocket telemetry, in-memory JSON replay capture, and a Three.js browser
  client.  The browser renders state; it does not run flight physics.
- A state-based `FlightStackRaceEnv`, optional Gymnasium adapter, versioned
  27-value observation/action/reward contracts, and a checkpoint-backed
  `LearnedPolicyPilot` that still emits the same CTBR command.
- Optional Stable-Baselines3 PPO training/export, checkpoint compatibility
  metadata, headless scenario evaluation, paired-seed statistics, and a
  robustness-grid builder.
- Existing quaternion/control/IMU/HIL laboratory modules, dependency-free C++20
  attitude primitives, and a Rust workspace that implements the same canonical
  vehicle contracts and reference plant.

## Deliberately not claimed yet

- **No quality or recommended learned checkpoint ships.** A 256-step PPO smoke
  run and a 10,000-step full-course PPO run were executed locally, but neither
  completed the seeded technical-eight evaluation.  Their ignored local
  artifacts are not a result or a supported policy release.
- `ReferenceVectorEnv` batches exact Python reference environments; it is **not
  a JAX backend** or a high-throughput replacement physics implementation.
  No JAX backend, curriculum, visual policy, or sim-to-real claim is shipped.
- Rust is a tested reference runtime, but the interactive server and current
  training environment use the Python reference runtime.  Cross-language
  trajectory-parity experiments are still future work.
- No serial transport, MCU firmware integration, actuator interface, hardware
  arming state machine, or flight-test calibration is provided.

## Quick start

FlightStack supports Python 3.11+.  The commands below assume PowerShell;
replace the activation command on other shells.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[dev]'

# Original closed-loop attitude reference scenario.
flightstack simulate --csv telemetry/reference.csv
```

The interactive client is served by the same process that owns the simulation:

```powershell
cd web
pnpm install --frozen-lockfile
pnpm run build
cd ..

flightstack serve
```

Open `http://127.0.0.1:8000`.  The production bundle is intentionally not
committed; rebuild it after changing files under `web/`.

The `serve` command runs the simulation at a canonical **500 Hz / 2 ms** step.
It accumulates wall-clock time so a coarse host event loop does not change the
physics timestep, and broadcasts browser telemetry every 15 physics steps
(nominally about 33 Hz).  A fresh human session remains in preflight until its
collective reaches 95% of hover thrust; this prevents the unattended craft from
falling before a browser has connected.

## Optional PPO training and learned-policy inspection

Training dependencies are opt-in so manual/reference users do not install the
ML stack by default:

```powershell
python -m pip install -e '.[train]'

# A real but deliberately tiny PPO plumbing check.
flightstack train --output artifacts/training-smoke --smoke

# A longer run is still an experiment, not a quality claim.
flightstack train --output artifacts/training-run --timesteps 10000 --seed 17
```

Each run writes `ppo_model.zip` and a required
`ppo_model.metadata.json` sidecar.  The sidecar versions the action and
observation schemas, records the vehicle and complete AI-configuration hashes,
the 20 ms control period, a content hash of the model archive, and the training
configuration.  A supplied policy is loaded only when those contracts match:

```powershell
flightstack serve --policy artifacts/training-run/ppo_model.zip
```

This enables the browser's `LEARNED` selector; it does **not** endorse that
checkpoint's flight quality.  The local smoke and 10,000-step runs did not
complete the seeded technical-eight evaluation, so neither should be presented
as a useful racing policy.  See [AI status](docs/ai.md).

## Reproducible headless evaluation

`evaluate` executes the canonical Python plant/race path without the browser
and writes a result summary, sampled telemetry, and replay when `--output` is
provided:

```powershell
# The default named scenario has seeded wind and motor-efficiency variation.
flightstack evaluate --pilot classical --output artifacts/classical-evaluation

flightstack evaluate `
  --pilot learned `
  --policy artifacts/training-run/ppo_model.zip `
  --scenario technical-eight-wind-degraded `
  --output artifacts/learned-evaluation
```

The `artifacts/` directory is ignored by Git on purpose.  It keeps local
checkpoints and replay/result data out of source history until they are backed
by a reviewed experiment/provenance decision.  See
[experiments.md](docs/experiments.md) for paired evaluation and robustness
tools.

Inspect a recorded replay or export its authoritative source frames for a plot:

```powershell
flightstack replay artifacts/classical-evaluation/replay.json --at 4.0 --interpolate
flightstack replay artifacts/classical-evaluation/replay.json --csv artifacts/replay.csv
```

Replay v1 preserves sampled canonical state (including motor thrust), CTBR
command, pilot, race snapshot, and events.  Playback interpolates only the
continuous state; pilot/race/events remain recorded discrete data.

## Common verification commands

```powershell
# Python reference runtime, AI contracts, and experiment helpers
pytest
ruff check .
mypy src

# C++20 primitives
cmake -S cpp -B cpp/build
cmake --build cpp/build --config Release
ctest --test-dir cpp/build -C Release --output-on-failure

# Rust reference runtime
cd rust
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

On Unix-like systems, omit the Windows-specific `--config Release` / `-C
Release` switches if using a single-config generator.  The CI workflow runs
Python, C++, Rust, and the production web build independently.

## Architecture in one view

```text
Human input / Classical pilot / validated LearnedPolicyPilot
                         |
                         v
               CTBR: collective thrust + body-rate target
                         |
                         v
          rate PID -> 4-rotor mixer -> motor lag -> 6DOF plant
                         |                         |
                         |                         +--> race/collision/replay
                         v
              authoritative FlightSession telemetry
                         |
                         v
                  WebSocket -> Three.js renderer

FlightStackRaceEnv -> same CTBR/mixer/motor/6DOF/race path -> PPO/evaluation
```

The complete boundaries and component responsibilities are in
[Architecture](docs/architecture.md).  The rendered scene converts coordinates
once at its edge; all simulation and race judgement retain FlightStack's
canonical frames.

## Canonical contract

| Item | FlightStack convention |
| --- | --- |
| World axes | Right-handed local ENU-like: `+X` east, `+Y` north, `+Z` up |
| Body axes | FLU: `+X` forward, `+Y` left, `+Z` up |
| Quaternion | Scalar-first `[w, x, y, z]`, body-to-world |
| Body rate / moment | Expressed in body coordinates |
| Rotor thrust | Positive body `+Z` |
| Pilot command | CTBR: collective thrust in N + target body rates in rad/s |

See [frame conventions](docs/frame-conventions.md) and
[reference physics](docs/physics.md) for the equations and adapter rules.

## Repository map

```text
config/vehicles/       canonical reference vehicle TOML
config/ai/             versioned state-racing environment/observation/reward config
config/scenarios/      reproducible headless experiment scenarios
src/flightstack/
  sim/                 original rotational lab + 6DOF vehicle/runtime
  control/, math/      quaternion and cascaded control primitives
  race/                tracks, swept gates, collision helpers, event state
  runtime/             human/classical/learned pilot contracts and replay
  ai/                  race environment, PPO plumbing, policy compatibility
  experiments/         scenarios, headless runs, paired evaluation, robustness grids
  web/                 authoritative local server and WebSocket transport
  sensors/, estimation/, hil/
                        deterministic IMU, estimator, and HIL framing lab
cpp/                   dependency-free C++20 control primitives
rust/                  Rust canonical contracts and 6DOF reference runtime
tracks/                data-driven course definitions
web/                   Three.js/Vite browser client
docs/                  architecture, controls, AI, experiments, and safety notes
```

## Documentation

- [Architecture](docs/architecture.md) — authoritative/runtime boundaries.
- [Controls](docs/controls.md) — browser controls, CTBR mapping, and race
  behavior.
- [Physics](docs/physics.md) and [frame conventions](docs/frame-conventions.md)
  — model and coordinate contract.
- [AI status and contract](docs/ai.md) — environment, PPO/checkpoint boundary,
  and explicit local-run limitations.
- [Experiments](docs/experiments.md) — headless artifacts, paired statistics,
  and robustness tools.
- [Hardware roadmap](docs/hardware-roadmap.md) — staged, safety-first path that
  does not overstate current readiness.
- [Research/source pack](docs/research/SOURCE_MANIFEST.md) — pinned reference
  sources, reuse boundaries, and licensing decisions.

## License and third-party material

FlightStack is MIT licensed.  It does not vendor third-party source, meshes,
textures, datasets, or model checkpoints.  Direct declared dependencies and
reference-source status are recorded in [THIRD_PARTY.md](THIRD_PARTY.md) and
the [research reuse audit](docs/research/reuse-audit.md).
