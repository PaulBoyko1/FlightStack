# FlightStack reference code pack

This file contains **FlightStack-native starting implementations and interface sketches** derived from the already-audited open-source designs listed in `SOURCE_MANIFEST.md`.

These snippets are intentionally small and self-contained. They are not intended to vendor third-party projects. Use them as implementation seeds, then verify against the pinned upstream implementations and FlightStack's own tests.

The purpose is to prevent the implementation agent from spending reasoning time rediscovering basic equations/interfaces.

---

## 1. Canonical quaternion adapter boundary

FlightStack canonical storage is `[w,x,y,z]`. External scalar-last APIs should cross one explicit adapter.

```rust
pub fn wxyz_to_xyzw(q: [f64; 4]) -> [f64; 4] {
    [q[1], q[2], q[3], q[0]]
}

pub fn xyzw_to_wxyz(q: [f64; 4]) -> [f64; 4] {
    [q[3], q[0], q[1], q[2]]
}
```

Required tests:

```text
identity round trip
90 deg X/Y/Z
arbitrary normalized quaternion
q and -q round trip
body->world vector rotation parity with upstream adapter
```

Never scatter index shuffling through simulator/training/frontend code.

---

## 2. Canonical runtime types

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FlightState {
    pub sim_time_s: f64,
    pub position_world_m: [f64; 3],
    pub velocity_world_m_s: [f64; 3],
    pub q_body_to_world_wxyz: [f64; 4],
    pub body_rate_rad_s: [f64; 3],
    pub motor_thrust_n: [f64; 4],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PilotCommand {
    pub collective_thrust_n: f64,
    pub body_rate_rad_s: [f64; 3],
}

pub trait Pilot {
    fn reset(&mut self, initial: &FlightState);
    fn command(&mut self, state: &FlightState, race: &RaceState, dt_s: f64) -> PilotCommand;
}
```

Core architectural invariant:

```text
HumanPilot
ClassicalPilot
LearnedPilot
      ↓
same PilotCommand
      ↓
same low-level rate controller
      ↓
same mixer
      ↓
same motors
      ↓
same plant
```

This is a design seed, not a requirement to preserve exact Rust type names.

---

## 3. Canonical vehicle configuration

Keep one serializable physical source of truth.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VehicleConfig {
    pub name: String,
    pub mass_kg: f64,
    pub inertia_kg_m2: [[f64; 3]; 3],
    pub motor_position_body_m: [[f64; 3]; 4],
    pub motor_spin_direction: [f64; 4],
    pub motor_min_thrust_n: f64,
    pub motor_max_thrust_n: f64,
    pub motor_tau_rise_s: f64,
    pub motor_tau_fall_s: f64,
    pub thrust_to_reaction_torque_m: f64,
    pub linear_drag_coeff: [f64; 3],
    pub angular_drag_coeff: [f64; 3],
    pub gravity_m_s2: f64,
}
```

Suggested source file:

```text
config/vehicles/flightstack_5in.toml
```

Do not duplicate values in Rust, Python, and TypeScript. Generate/load representations from the same config.

Label parameter provenance as measured / estimated / reference-derived / simulation-tuned.

---

## 4. Stable first-order motor response

A simple transparent actuator model covers most of the needed v1 behavior.

```rust
pub fn step_motor_thrust(
    current_n: f64,
    command_n: f64,
    dt_s: f64,
    tau_rise_s: f64,
    tau_fall_s: f64,
    min_n: f64,
    max_n: f64,
) -> f64 {
    let target = command_n.clamp(min_n, max_n);
    let tau = if target >= current_n { tau_rise_s } else { tau_fall_s };

    if tau <= 0.0 {
        return target;
    }

    let alpha = 1.0 - (-dt_s / tau).exp();
    (current_n + alpha * (target - current_n)).clamp(min_n, max_n)
}
```

Useful tests:

```text
zero dt -> unchanged
zero/near-zero tau -> target
monotonic step-up
monotonic step-down
analytic exponential comparison
never exceeds min/max
```

Failure injection becomes straightforward:

```text
effective_thrust_i = nominal_thrust_i * motor_efficiency_i
```

---

## 5. Per-motor force and torque accumulation

Assuming canonical body +Z thrust:

```rust
pub fn accumulate_motor_wrench(
    thrust_n: [f64; 4],
    motor_pos_body_m: [[f64; 3]; 4],
    spin: [f64; 4],
    reaction_arm_m: f64,
) -> ([f64; 3], [f64; 3]) {
    let mut force = [0.0, 0.0, 0.0];
    let mut torque = [0.0, 0.0, 0.0];

    for i in 0..4 {
        let f = [0.0, 0.0, thrust_n[i]];
        force[2] += thrust_n[i];

        let r = motor_pos_body_m[i];
        let arm = [
            r[1] * f[2] - r[2] * f[1],
            r[2] * f[0] - r[0] * f[2],
            r[0] * f[1] - r[1] * f[0],
        ];

        torque[0] += arm[0];
        torque[1] += arm[1];
        torque[2] += arm[2] + spin[i] * reaction_arm_m * thrust_n[i];
    }

    (force, torque)
}
```

Validate motor ordering/signs against an actual selected frame geometry; do not assume this snippet's ordering magically matches the final model.

Required invariants:

```text
all equal thrust -> near-zero roll/pitch/yaw torque
left/right differential -> expected roll sign
front/rear differential -> expected pitch sign
CW/CCW pattern -> expected yaw sign
```

---

## 6. Translational dynamics seed

World convention recommended: right-handed, +Z up.

```text
F_body = [0, 0, total_motor_thrust]
F_world = R(q_body_to_world) * F_body
F_gravity = [0, 0, -m*g]
F_drag = documented velocity-dependent drag
F_total = F_world + F_gravity + F_drag + F_disturbance
accel_world = F_total / m
velocity += accel_world * dt
position += velocity * dt
```

Keep the first implementation transparent. If a more advanced rotor-drag/velocity model from Crazyflow is added, validate it against this baseline rather than silently replacing terms.

---

## 7. Rotational dynamics seed

Preserve the existing FlightStack equation:

```text
omega_dot = J^-1 * (tau - omega x (J*omega))
```

Add optional angular drag/disturbance torque outside the existing verified core:

```text
tau_total = tau_motors + tau_drag + tau_disturbance
```

Continue using FlightStack's exact body-rate quaternion integration. Do not integrate Euler angles.

---

## 8. Mixer formulation

For a generic X quad, implement the mixer from actual motor positions/spin directions rather than memorizing a sign matrix.

A robust approach is to construct a 4x4 allocation matrix from each motor's contribution to:

```text
collective thrust
roll torque
pitch torque
yaw torque
```

Then solve/invert/pseudoinvert once for the configured vehicle.

Concept:

```text
u = [T_collective, tau_x, tau_y, tau_z]

A * motor_thrusts = u

motor_thrusts = allocation(u)
```

After solving:

1. clamp motor thrusts;
2. optionally perform desaturation preserving attitude axes according to a documented priority;
3. expose saturation state in telemetry.

Do not hide mixer signs in unexplained constants.

---

## 9. Swept gate crossing implementation seed

This is the compact FlightStack-native version of the robust gate-local plane-crossing concept audited in LSY Drone Racing.

Define each gate by:

```rust
pub struct Gate {
    pub center_world_m: [f64; 3],
    // Unit basis vectors in world coordinates.
    pub normal_world: [f64; 3],
    pub right_world: [f64; 3],
    pub up_world: [f64; 3],
    pub half_width_m: f64,
    pub half_height_m: f64,
}
```

Helper operations:

```rust
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn lerp(a: [f64; 3], b: [f64; 3], t: f64) -> [f64; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}
```

Crossing logic:

```rust
pub fn swept_gate_crossing(
    previous_world: [f64; 3],
    current_world: [f64; 3],
    gate: &Gate,
) -> Option<[f64; 3]> {
    let p0 = sub(previous_world, gate.center_world_m);
    let p1 = sub(current_world, gate.center_world_m);

    let d0 = dot(p0, gate.normal_world);
    let d1 = dot(p1, gate.normal_world);

    // Require forward crossing. Reverse semantics can be configured separately.
    if !(d0 < 0.0 && d1 >= 0.0) {
        return None;
    }

    let denom = d0 - d1;
    if denom.abs() < 1e-12 {
        return None;
    }

    let t = d0 / denom;
    if !(0.0..=1.0).contains(&t) {
        return None;
    }

    let crossing = lerp(previous_world, current_world, t);
    let rel = sub(crossing, gate.center_world_m);

    let x = dot(rel, gate.right_world);
    let y = dot(rel, gate.up_world);

    if x.abs() <= gate.half_width_m && y.abs() <= gate.half_height_m {
        Some(crossing)
    } else {
        None
    }
}
```

Required tests:

```text
straight valid crossing
45-degree tilted gate
vertical/elevation gate
high-speed crossing where neither endpoint is near gate center
edge inside
edge outside
reverse crossing
segment that never crosses plane
wrong gate order handled by RaceState
same gate not counted repeatedly
```

Gate collision geometry is separate from gate-pass aperture semantics.

---

## 10. Race state seed

```rust
pub struct RaceState {
    pub running: bool,
    pub finished: bool,
    pub lap: u32,
    pub next_gate_index: usize,
    pub lap_started_at_s: f64,
    pub last_gate_at_s: Option<f64>,
    pub best_lap_s: Option<f64>,
    pub collisions: u32,
}
```

Race update should consume events:

```text
GatePassed(index, crossing_position, time)
Collision(object_id, time)
Reset(time)
Start(time)
```

This event-oriented structure makes replay/statistics easier than burying all race logic inside rendering code.

---

## 11. Human control shaping seed

Raw stick input `x in [-1,1]`:

```rust
pub fn deadzone_expo(x: f64, deadzone: f64, expo: f64) -> f64 {
    let a = x.abs();
    if a <= deadzone {
        return 0.0;
    }

    let normalized = (a - deadzone) / (1.0 - deadzone);
    let curved = (1.0 - expo) * normalized + expo * normalized.powi(3);
    curved.copysign(x)
}
```

Suggested mapping:

```text
left vertical   -> thrust
left horizontal -> yaw rate
right vertical  -> pitch rate
right horizontal-> roll rate
```

Expose deadzone/expo/max rate in settings.

---

## 12. CTBR action mapping for learned pilot

Normalized policy action:

```text
a = [thrust, roll_rate, pitch_rate, yaw_rate]
all components in [-1,1]
```

A useful thrust mapping is hover-centered rather than `-1 = zero, 0 = half thrust` by accident.

Example concept:

```python
def map_action(action, mass_kg, gravity, thrust_min, thrust_max, rate_limits):
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

    hover = mass_kg * gravity

    if a[0] >= 0:
        collective = hover + a[0] * (thrust_max - hover)
    else:
        collective = hover + a[0] * (hover - thrust_min)

    body_rates = a[1:4] * np.asarray(rate_limits, dtype=np.float32)
    return collective, body_rates
```

Use actual total vehicle thrust limits, not per-motor limits here.

Version this mapping with checkpoints.

---

## 13. State-based racing observation seed

Do not feed raw world coordinates simply because they are easy.

Preferred invariant/local observation:

```text
body-frame velocity                     3
body angular velocity                   3
body-frame vector to next gate          3
next gate normal expressed in body      3
body-frame vector to gate after next    3
optional second gate normal             3
body gravity/up vector                  3
previous normalized action              4
normalized distance / speed scalars     small
```

This is a starting point. Empirically evaluate ablations.

Important:

- avoid raw quaternion sign ambiguity in the policy input when a body-frame representation is cleaner;
- store observation schema version in checkpoint metadata;
- normalize values using explicit documented scales/statistics.

---

## 14. Racing reward seed

Reward should be delta-based to reduce stationary reward exploits.

Concept:

```python
def race_reward(prev, cur, event, cfg):
    reward = 0.0

    reward += cfg.progress * (cur.progress_to_next_gate - prev.progress_to_next_gate)

    if event.gate_passed:
        reward += cfg.gate_pass

    if event.lap_complete:
        reward += cfg.lap_complete

    if event.collision:
        reward += cfg.collision

    if event.out_of_bounds:
        reward += cfg.out_of_bounds

    reward += cfg.action_delta * np.square(cur.action - prev.action).sum()
    reward += cfg.angular_rate * np.square(cur.body_rate).sum()

    return float(reward)
```

Store weights in versioned YAML/TOML, not hard-coded source.

Start simple and instrument reward terms separately so reward hacking can be diagnosed.

---

## 15. Gymnasium-like environment contract

```python
class FlightStackRaceEnv(gym.Env):
    def __init__(self, config):
        self.config = config
        self.observation_space = ...
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)
        obs = self._observation()
        info = self._info()
        return obs, info

    def step(self, action):
        command = self._map_action(action)

        for _ in range(self.control_substeps):
            self.sim.step(command, self.physics_dt)

        obs = self._observation()
        reward = self._reward()
        terminated = self._terminated()
        truncated = self._truncated()
        info = self._info()
        return obs, reward, terminated, truncated, info
```

The vectorized/JAX training backend may use a more functional API internally, but preserve equivalent semantics.

---

## 16. Functional JAX training-backend shape

Keep JAX state pure/vectorizable.

```python
@jax.tree_util.register_pytree_node_class
@dataclass
class SimState:
    position: jax.Array
    velocity: jax.Array
    quat_wxyz: jax.Array
    body_rate: jax.Array
    motor_thrust: jax.Array


def step_vehicle(params, state, pilot_command, disturbance, dt):
    motor_target = mixer(params, pilot_command, state)
    motor_next = step_motors(params, state.motor_thrust, motor_target, dt)
    force_body, torque_body = motor_wrench(params, motor_next)
    return integrate_6dof(params, state, force_body, torque_body, disturbance, dt)

batched_step = jax.jit(jax.vmap(step_vehicle, in_axes=(None, 0, 0, 0, None)))
```

Exact types may differ. Key properties:

- no hidden mutable global state;
- deterministic PRNG keys;
- vectorizable over environments;
- same canonical vehicle parameters;
- explicit parity tests against Rust reference scenarios.

---

## 17. Backend parity harness shape

Use identical initial states and command sequences.

```python
def compare_backend(reference_runner, training_runner, scenario):
    ref = reference_runner.run(scenario)
    fast = training_runner.run(scenario)

    return {
        "position_rmse_m": rmse(ref.position, fast.position),
        "velocity_rmse_m_s": rmse(ref.velocity, fast.velocity),
        "attitude_geodesic_rmse_rad": quat_geodesic_rmse(ref.quat, fast.quat),
        "body_rate_rmse_rad_s": rmse(ref.body_rate, fast.body_rate),
        "motor_rmse_n": rmse(ref.motor_thrust, fast.motor_thrust),
    }
```

Parity scenarios:

```text
free fall
hover
collective step
single-motor step
roll command
pitch command
yaw command
drag decay
mixed command sequence
fixed disturbance
```

Do not demand bitwise equality across integrators/precision choices. Define bounded physically meaningful tolerances and explain deviations.

---

## 18. Deterministic scenario shape

```yaml
name: nominal-race-01
seed: 42
vehicle: flightstack_5in
track: technical-eight
pilot: classical

initial:
  position_world_m: [0.0, 0.0, 1.0]
  velocity_world_m_s: [0.0, 0.0, 0.0]
  q_body_to_world_wxyz: [1.0, 0.0, 0.0, 0.0]
  body_rate_rad_s: [0.0, 0.0, 0.0]

disturbance:
  wind_world_m_s: [0.0, 0.0, 0.0]
  motor_efficiency: [1.0, 1.0, 1.0, 1.0]
  external_force_world_n: [0.0, 0.0, 0.0]
  external_torque_body_nm: [0.0, 0.0, 0.0]

sensor:
  gyro_bias_rad_s: [0.0, 0.0, 0.0]
  gyro_noise_std: 0.0
  dropout_probability: 0.0
  latency_ms: 0.0
```

Persist the effective resolved config with each experiment run.

---

## 19. Telemetry message seed

Start with JSON. Optimize only after profiling.

```json
{
  "type": "state",
  "sim_time_s": 12.45,
  "pilot": "learned",
  "state": {
    "position_world_m": [1.2, -0.4, 3.6],
    "velocity_world_m_s": [8.1, 1.3, -0.2],
    "q_body_to_world_wxyz": [0.98, 0.03, 0.16, -0.08],
    "body_rate_rad_s": [0.4, -1.1, 0.2]
  },
  "motors": {
    "thrust_n": [2.1, 2.4, 2.0, 2.5]
  },
  "pilot_command": {
    "collective_thrust_n": 9.0,
    "body_rate_rad_s": [0.3, -1.0, 0.1]
  },
  "race": {
    "lap": 1,
    "next_gate": 4,
    "lap_time_s": 12.45,
    "collisions": 0
  }
}
```

Physics tick, policy tick, telemetry tick, and render tick are separate clocks.

---

## 20. Paired-seed experiment runner

For classical-vs-learned comparisons, use paired scenarios.

```python
def paired_eval(pilots, seeds, scenario_factory):
    results = []

    for seed in seeds:
        scenario = scenario_factory(seed)
        for pilot_name, pilot in pilots.items():
            result = run_episode(scenario=scenario, pilot=pilot)
            results.append({
                "seed": seed,
                "pilot": pilot_name,
                **result.metrics,
            })

    return pd.DataFrame(results)
```

This ensures both pilots face the same:

```text
initial state
track
vehicle randomization
wind realization
sensor errors
motor degradation
```

Prefer paired differences/bootstrapped confidence intervals rather than comparing unrelated random batches.

---

## 21. Robustness-grid shape

```python
for wind in wind_levels:
    for efficiency in motor_efficiency_levels:
        for seed in seeds:
            run(
                seed=seed,
                wind=wind,
                motor_efficiency=[1.0, 1.0, efficiency, 1.0],
            )
```

Aggregate:

```text
completion rate
median lap time
collision rate
control effort
```

Render at least one 2D robustness map/heatmap.

---

# What to lift directly from this file

Safe starting points for immediate implementation:

```text
quaternion adapter
canonical state/command contracts
VehicleConfig shape
motor first-order update
per-motor wrench accumulation
swept gate crossing
race events/state
manual deadzone/expo
CTBR action scaling concept
observation schema
reward structure
scenario schema
telemetry schema
backend parity harness
paired-seed experiment structure
```

Before finalizing each subsystem, compare against the exact pinned upstream file in `SOURCE_MANIFEST.md` and write tests.

# What not to recreate from snippets

Use mature libraries for:

```text
PPO optimizer implementation
neural-network framework
Three.js rendering core
uPlot charting
collision engine internals
WebSocket implementation
Rust/C++ bridge generator
```

The objective is to turn known good patterns into a coherent FlightStack implementation, not to maximize hand-written infrastructure.
