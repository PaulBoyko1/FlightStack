# FlightStack vetted source manifest

This file is a pre-researched implementation map for the FlightStack digital-twin/racing/autonomy expansion. It is intentionally narrower than an ecosystem survey: each source below has a concrete reason to exist in the build.

The goal is to save implementation/research time while preserving FlightStack's own identity and its tested frame/control semantics.

## Rules

1. Prefer dependencies and small adapters over vendoring entire projects.
2. Preserve FlightStack's canonical quaternion semantics internally: scalar-first `[w,x,y,z]`, body-to-world.
3. Treat external frame/quaternion conventions as explicit tested boundaries.
4. If directly adapting source, retain the license/notice required by that source.
5. Do not copy GPL or unlicensed implementation into the MIT FlightStack core.
6. Physical parameters from external simulators are starting/reference values, not measured FlightStack hardware truth.
7. Re-open an upstream repository only when implementing the subsystem mapped here or when a specific problem requires more research.

---

## S1 — Elodin AI Grand Prix practice simulator

Repository: https://github.com/elodin-sys/ai-grand-prix

Pinned commit: `13f9f9e3d5a3130f0ce0b65500d9f309cc1e11b2`

License: Apache-2.0

Priority: **S — inspect first for reference simulator implementation**

Exact files to inspect:

- `sim/physics.py`
  - motor first-order response
  - individual rotor thrust
  - arm moments
  - yaw reaction torque
  - body/world force application
  - drag
  - gravity
  - 6DOF integration structure
- `sim/config.py`
  - generic racing-quad starting parameters
  - motor layout and dynamics parameters
- `sim/sensors.py`
  - deterministic multi-rate sensor model
  - gyro/accelerometer/barometer/magnetometer noise/bias patterns
- `sim/camera.py`
  - forward FPV sensor camera structure
  - camera intrinsics/rate decoupling
- `sim/course.py`
  - ordered race progress and timing structure
- `sim/visualization.py`
  - motor/propeller visualization
  - thrust-vector visualization
- `sim/main.py`
  - deterministic multi-rate orchestration reference
- `sim/betaflight_bridge.py`
  - future external SITL/lockstep reference only
- `solver/api.py`
  - sensor-update/autopilot/command boundary
  - particularly useful as inspiration for FlightStack's interchangeable pilot abstraction
- `solver/baseline.py`
  - tiny deterministic takeoff/gate-center autopilot useful as a smoke-test baseline

FlightStack application:

- Adapt the transparent physics model into the authoritative FlightStack reference simulator.
- Preserve FlightStack's existing quaternion integration/controller semantics.
- Use the source as a reference rather than wrapping the entire AGP simulator.
- Use its solver boundary to inform `HumanPilot`, `ClassicalPilot`, and `LearnedPilot` contracts.

Critical hazard:

Elodin-related code may use scalar-last `[x,y,z,w]` while FlightStack uses `[w,x,y,z]`. Conversion must occur through one explicit adapter layer with golden tests.

Do not:

- replace FlightStack's tested quaternion/control implementation wholesale;
- blindly inherit external frame conventions;
- present borrowed 5-inch parameters as measured hardware values.

---

## S2 — LSY Drone Racing

Repository: https://github.com/learnsyslab/lsy_drone_racing

Pinned commit: `9ecb1cb0a78a232b7cad764344797bfa1c45f462`

License: MIT

Priority: **S — racing environment, gate geometry, training/inference patterns**

Exact files:

- `lsy_drone_racing/envs/utils.py`
  - geometry utilities; inspect specifically for gate-local / crossing helpers
- `lsy_drone_racing/envs/race_core.py`
  - race-state/environment mechanics
  - gates/obstacles/crash/bounds
- `lsy_drone_racing/envs/drone_race.py`
  - Gymnasium-style single-race environment surface
- `lsy_drone_racing/envs/randomize.py`
  - start-state/track/physical randomization patterns
- `lsy_drone_racing/control/train_rl.py`
  - complete PPO-oriented training plumbing
  - vectorized environments, history, checkpointing, evaluation structure
- `lsy_drone_racing/control/attitude_rl.py`
  - learned-policy deployment/inference path
  - reconstructing observation/history and scaling normalized actions to physical commands
- `lsy_drone_racing/control/state_controller.py`
  - conventional controller/reference pattern
- `tests/unit/envs/test_race_core.py`
  - useful expected-behavior coverage for racing logic
- `tests/integration/test_envs.py`
  - environment integration test patterns

FlightStack application:

- Adapt robust swept gate-plane crossing rather than proximity-only gate detection.
- Reuse the race-state ideas but keep the runtime in FlightStack's architecture.
- Adapt training/checkpoint/inference plumbing rather than reimplementing PPO infrastructure.
- Reuse randomization concepts for Monte Carlo/robustness experiments.

Target FlightStack modules:

- Rust race/gate subsystem
- Python/JAX training environment
- experiment randomization/configuration

Do not:

- clone its whole architecture into FlightStack;
- inherit Crazyflie-specific assumptions without translating them into canonical `VehicleConfig`;
- let training state conventions leak into the reference runtime.

---

## S3 — Crazyflow

Repository: https://github.com/learnsyslab/crazyflow

Pinned commit: `58e8fb4743d35942bd2aa6dd2c42b1190fa5e980`

License: MIT

Priority: **S/A — high-throughput/vectorized training physics reference**

Verified files:

- `crazyflow/drones/params.toml`
  - mass/inertia/arm length
  - thrust and torque curves
  - rotor dynamics coefficients
  - mixing matrices
  - drag terms
- `crazyflow/drones/__init__.py`
  - supported hardware registry
- `crazyflow/dynamics/first_principles/dynamics.py`
  - first-principles rotor/rigid-body dynamics reference
- `.devcontainer/devcontainer.wsl2.json`
  - useful Windows/WSL2 development path; GPU runtime hooks are present

Known limitation:

The public hardware registry at this commit is centered on Crazyflie-family vehicles. Do not silently train a Crazyflie and call it the FlightStack 5-inch vehicle.

FlightStack application:

Perform one bounded spike:

A. cleanly add/adapt a canonical FlightStack 5-inch parameter set to a Crazyflow-derived high-throughput backend, **or**

B. implement a thin vectorized JAX version of the same authoritative FlightStack equations.

Select one based on:

- parity with Rust reference backend
- maintainability
- implementation cost
- vectorized throughput
- clean sharing of `VehicleConfig`

Keep the winner; remove/avoid redundant alternative infrastructure.

---

## A1 — gym-pybullet-drones

Repository: https://github.com/learnsyslab/gym-pybullet-drones

Pinned commit: `e712698a05a80728b06572819dcf044596707754`

License: MIT

Files:

- `gym_pybullet_drones/envs/BaseRLAviary.py`
  - action-space patterns
  - kinematic/RGB/depth observation patterns
  - action history
  - translation of abstract actions into low-level control
- `gym_pybullet_drones/examples/learn.py`
  - compact Stable-Baselines3 train/evaluate/checkpoint example

FlightStack application:

Reference/API inspiration only unless it solves a specific integration problem more cleanly than the selected JAX/LSY path.

Do not make PyBullet the authoritative reference simulator merely for convenience.

---

## A2 — RLtools / Learning to Fly

Repository: https://github.com/rl-tools/rl-tools

Pinned commit: `b32d9985c65a5e098a6bbf190fd994962d288b99`

License: MIT

Relevant subtree/files:

- `src/rl/zoo/l2f/environment.h`
- `src/rl/zoo/l2f/environment_tiny.h`
- `src/rl/zoo/l2f/environment_big.h`
- `src/rl/zoo/l2f/ppo.h`
- `src/rl/zoo/l2f/sac.h`
- `src/rl/zoo/l2f/td3.h`
- broader `src/rl/environments/l2f/` implementation as needed

Useful concepts:

- very fast CPU-oriented continuous-control RL
- multirotor environment architecture
- action history
- domain randomization
- lightweight policy inference/deployment

FlightStack application:

Secondary benchmark/deployment path. Do not integrate multiple full RL frameworks simultaneously without evidence that the comparison is worth the maintenance cost.

---

## A3 — SimpleFlight

Repository: https://github.com/thu-uav/SimpleFlight

Pinned commit: `f5ae8fc3689edc7c982e07cbfd587eeeab72f279`

License: MIT

Exact files of interest:

- `examples/visual_sim_ctbr.py`
- `examples/visual_deployment_ctbr.py`
- `omni_drones/controllers/cf2x_pid.py`
- `omni_drones/controllers/dsl_pid_controller.py`
- `omni_drones/envs/single/track.py`
- `scripts/train.py`
- `scripts/eval.py`

Primary use:

Architectural support for a **collective-thrust + body-rate (CTBR)** learned/control interface.

Recommended FlightStack contract:

`Pilot -> collective thrust + roll/pitch/yaw body-rate targets -> same low-level rate controller -> mixer -> motors -> plant`

This gives human, classical, and learned pilots the same actuator semantics and is better aligned with future real-hardware transfer than making raw motor policy output the default.

---

## A4 — Isaac Drone Racer

Repository: https://github.com/kousheekc/isaac_drone_racer

Pinned commit: `d530f67768d53701454f406e4da967e0e9c30842`

License: BSD-3-Clause at project level; verify individual asset provenance.

Exact files:

- `assets/5_in_drone/urdf/5_in_drone.urdf`
  - 5-inch-style mass/inertia/motor geometry starting reference
- drone body/prop meshes under `assets/5_in_drone/`
  - potential visual source if asset provenance permits
- racing task/environment configuration under the repository's drone-racer task implementation
  - observation/reward/randomization reference

Known practical issue:

Large meshes may use Git LFS. Verify actual binaries, not LFS pointer text.

Use:

- geometry/visual reference
- approximate physical starting configuration
- racing reward/observation comparison

Do not:

- claim these values are measured FlightStack hardware
- force Isaac as a runtime dependency

---

## A5 — Rust/C++ FFI

Repository: https://github.com/dtolnay/cxx

License: MIT/Apache-2.0 ecosystem; verify current crate metadata when integrating.

Use:

- typed Rust/C++ boundary
- narrow transfer of controller state/commands

Target boundary should look like a plausible future embedded interface, not an arbitrary cross-language split.

Do not hand-roll a broad unsafe FFI layer without a measured reason.

---

## A6 — Three.js

Repository: https://github.com/mrdoob/three.js

Researched release: `r185`

License: MIT

Use as the likely browser renderer for:

- GLTF/GLB model loading
- chase/FPV/spectator cameras
- lighting/shadows
- scene graph
- body axes/vectors
- gates/environment

The browser is a client. It must never be the authoritative simulation backend.

---

## A7 — uPlot

Repository: https://github.com/leeoniya/uPlot

License: MIT

Use for compact, high-rate engineering plots. Do not implement a charting engine.

---

## A8 — Rapier / Parry

Repository family: https://github.com/dimforge/rapier

License: Apache-2.0

Use for:

- collision queries/events
- continuous collision detection
- spatial queries
- gate/wall/ground/obstacle collision geometry

Preferred division:

`FlightStack equations own flight dynamics; Rapier/Parry own collision geometry support.`

Do not let a generic rigid-body physics engine silently replace the reference multirotor dynamics unless benchmarks and parity justify it.

---

## A9 — Inertial estimator references

Repository: `Tellicious/InertialEstimators`

License researched as MIT; verify at integration time.

Use only if the core simulator is healthy and an estimator comparison remains valuable.

Potential comparison targets include compact EKF/Madgwick/PX4-style attitude estimators and altitude filtering.

Keep FlightStack's current complementary filter as a transparent baseline.

Do not integrate many estimators just because they are available.

---

## A10 — Rust minimum-snap trajectory reference

Repository: `rsasaki0109/rust_robotics`

License researched as MIT; verify current source/license before adapting.

Useful component:

- seventh-order minimum-snap trajectory implementation with position/velocity/acceleration/jerk boundary constraints and tests

Use only after the main vertical slice is working, likely for the classical/reference trajectory system.

---

## Reference-only / external interoperability

### Betaflight / IndiFlight

Useful for:

- real racing-controller behavior
- SITL interoperability
- eventual comparison/HIL

Do not absorb copyleft implementation into FlightStack's MIT core.

### PX4 / ArduPilot

Useful as architecture/hardware-interface references when a concrete question arises. Avoid pulling their complexity into the v1 simulator.

### MonoRace / A2RL literature

Use as research guidance for:

- aggressive racing policy design
- reward term families
- domain randomization
- camera+IMU future mode

Do not claim code exists where there is no clearly licensed public implementation.

---

# Recommended reuse order

Before writing each subsystem, inspect only these sources:

| FlightStack subsystem | First source | Secondary source |
|---|---|---|
| Reference 6DOF plant | Elodin AGP `sim/physics.py` | Crazyflow first-principles dynamics |
| Vehicle parameters | Elodin config + Isaac 5-inch URDF | Crazyflow params schema |
| Motor lag/mixer | Elodin AGP | Crazyflow |
| Sensors | Elodin AGP sensors | existing FlightStack IMU |
| Gate pass | LSY `envs/utils.py` + `race_core.py` | AGP `course.py` |
| Race state | LSY `race_core.py` | AGP course state |
| Classical smoke bot | AGP baseline | LSY state controller |
| Shared command interface | SimpleFlight CTBR | AGP solver API |
| PPO plumbing | LSY `train_rl.py` | gym-pybullet-drones `learn.py` |
| Policy inference | LSY `attitude_rl.py` | RLtools L2F |
| Training physics | Crazyflow/JAX spike | thin custom JAX parity backend |
| Collision | Rapier/Parry | custom simple primitives only if simpler |
| 3D rendering | Three.js | no second renderer unless necessary |
| Charts | uPlot | no custom charting |
| Rust/C++ boundary | cxx | raw FFI only if justified |

# Core principle

The references remove commodity work. They do not define FlightStack's identity.

The original system should remain centered on:

- canonical state/frame contracts
- tested reference physics
- C++ flight-control core
- Rust runtime
- shared human/classical/learned pilot interface
- manual simulator
- reference/training backend parity
- statistical experiments
- robustness/failure analysis
- eventual sim-to-real measurement
