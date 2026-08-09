//! Deterministic, fixed-step FlightStack multirotor reference simulation.
//!
//! This crate intentionally implements transparent multirotor equations rather
//! than delegating the flight plant to a generic rigid-body engine.  Its input
//! and output types are the canonical contracts from `flightstack-core`.

use std::fmt;

use flightstack_core::quaternion::{
    integrate_body_rate, rotate_body_to_world, rotate_world_to_body,
};
use flightstack_core::{
    ContractError, ControlConfig, FlightState, Mat3, MotorArray, PilotCommand, Vec3, VehicleConfig,
};

const MATRIX_EPSILON: f64 = 1.0e-12;
const SATURATION_EPSILON: f64 = 1.0e-12;
const DERIVATIVE_CUTOFF_HZ: f64 = 35.0;

type Mat4 = [[f64; 4]; 4];

/// Errors at a simulation boundary.  A failed step never partially commits a
/// new state: all validation occurs before the state assignment.
#[derive(Debug)]
pub enum SimError {
    /// A canonical state/command/quaternion contract was not met.
    Contract(ContractError),
    /// A finite/range requirement specific to a simulation operation failed.
    InvalidInput(String),
    /// Motor geometry and spin directions cannot independently allocate the
    /// collective/roll/pitch/yaw wrench.
    SingularMixer,
    /// The validated inertia matrix could not be solved numerically.
    SingularInertia,
}

impl fmt::Display for SimError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Contract(error) => write!(formatter, "FlightStack contract error: {error}"),
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::SingularMixer => {
                formatter.write_str("motor geometry/spin directions do not form a full-rank mixer")
            }
            Self::SingularInertia => formatter.write_str("inertia matrix could not be solved"),
        }
    }
}

impl std::error::Error for SimError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Contract(error) => Some(error),
            Self::InvalidInput(_) | Self::SingularMixer | Self::SingularInertia => None,
        }
    }
}

impl From<ContractError> for SimError {
    fn from(error: ContractError) -> Self {
        Self::Contract(error)
    }
}

/// Deterministic scenario-owned effects supplied for a single simulation step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Disturbance {
    /// Additional force in world coordinates.
    pub force_world_n: Vec3,
    /// Additional moment in body coordinates.
    pub torque_body_nm: Vec3,
    /// World-frame air velocity used by the body-frame linear drag model.
    pub wind_world_m_s: Vec3,
    /// Per-motor effectiveness from zero (failed) through one (nominal).
    pub motor_efficiency: MotorArray,
}

impl Disturbance {
    /// No external force, wind, torque, or motor degradation.
    pub const fn calm() -> Self {
        Self {
            force_world_n: [0.0; 3],
            torque_body_nm: [0.0; 3],
            wind_world_m_s: [0.0; 3],
            motor_efficiency: [1.0; 4],
        }
    }

    /// Validate a disturbance at its scenario/runtime boundary.
    pub fn validate(&self) -> Result<(), SimError> {
        ensure_finite_vec3(self.force_world_n, "force_world_n")?;
        ensure_finite_vec3(self.torque_body_nm, "torque_body_nm")?;
        ensure_finite_vec3(self.wind_world_m_s, "wind_world_m_s")?;
        for (index, efficiency) in self.motor_efficiency.iter().enumerate() {
            if !efficiency.is_finite() || !(0.0..=1.0).contains(efficiency) {
                return Err(SimError::InvalidInput(format!(
                    "motor_efficiency[{index}] must be finite and in [0, 1]"
                )));
            }
        }
        Ok(())
    }
}

impl Default for Disturbance {
    fn default() -> Self {
        Self::calm()
    }
}

/// Force and torque contributed by the four motors in body coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotorWrench {
    pub force_body_n: Vec3,
    pub moment_body_nm: Vec3,
}

/// Exact first-order motor-thrust response with separate rise/fall constants.
///
/// Inputs are expected to have been validated by [`VehicleConfig`] and the
/// caller.  The output is always clamped to the stated motor range.  A
/// nonpositive time constant acts as an instantaneous actuator, which makes
/// the primitive useful for focused tests even though canonical configs require
/// positive time constants.
pub fn step_motor_thrust(
    current_n: f64,
    command_n: f64,
    dt_s: f64,
    tau_rise_s: f64,
    tau_fall_s: f64,
    min_n: f64,
    max_n: f64,
) -> f64 {
    let target_n = command_n.clamp(min_n, max_n);
    let tau_s = if target_n >= current_n {
        tau_rise_s
    } else {
        tau_fall_s
    };
    if tau_s <= 0.0 {
        return target_n;
    }
    let alpha = 1.0 - (-dt_s / tau_s).exp();
    (current_n + alpha * (target_n - current_n)).clamp(min_n, max_n)
}

/// Sum rotor thrust, arm moment, and yaw reaction moment in body frame.
///
/// A rotor's thrust is `[0, 0, thrust]` because canonical FLU body `+Z` is up.
/// The arm contribution is `r_body x thrust_body` and yaw reaction is
/// `spin_direction * reaction_arm * thrust` around body `+Z`.
pub fn accumulate_motor_wrench(
    thrust_n: MotorArray,
    motor_pos_body_m: [Vec3; 4],
    spin_direction: MotorArray,
    thrust_to_reaction_torque_m: f64,
) -> MotorWrench {
    let mut force_body_n = [0.0; 3];
    let mut moment_body_nm = [0.0; 3];

    for index in 0..4 {
        let thrust = thrust_n[index];
        let position = motor_pos_body_m[index];
        force_body_n[2] += thrust;
        moment_body_nm[0] += position[1] * thrust;
        moment_body_nm[1] -= position[0] * thrust;
        moment_body_nm[2] += spin_direction[index] * thrust_to_reaction_torque_m * thrust;
    }

    MotorWrench {
        force_body_n,
        moment_body_nm,
    }
}

/// Result of mapping a requested CTBR wrench to bounded motor thrust targets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MixerResult {
    pub motor_target_thrust_n: MotorArray,
    pub achieved_collective_thrust_n: f64,
    pub achieved_torque_body_nm: Vec3,
    pub saturated: bool,
}

/// Geometry-derived four-rotor allocation matrix.
///
/// Rows map motor thrusts to `[collective, roll, pitch, yaw]`; the inverse is
/// computed once at construction and bounded results explicitly report
/// saturation rather than hiding it.
#[derive(Debug, Clone)]
pub struct QuadMixer {
    allocation: Mat4,
    allocation_inverse: Mat4,
    motor_min_thrust_n: f64,
    motor_max_thrust_n: f64,
}

impl QuadMixer {
    /// Build an allocation matrix directly from configured geometry and spin.
    pub fn new(config: &VehicleConfig) -> Result<Self, SimError> {
        config.validate()?;
        let motor_positions = config.motor_position_body_m;
        let allocation = [
            [1.0; 4],
            motor_positions.map(|position| position[1]),
            motor_positions.map(|position| -position[0]),
            std::array::from_fn(|index| {
                config.motor_spin_direction[index] * config.thrust_to_reaction_torque_m
            }),
        ];
        let allocation_inverse = invert_4x4(allocation).ok_or(SimError::SingularMixer)?;
        Ok(Self {
            allocation,
            allocation_inverse,
            motor_min_thrust_n: config.motor_min_thrust_n,
            motor_max_thrust_n: config.motor_max_thrust_n,
        })
    }

    /// Return the documented wrench-from-thrust allocation matrix.
    pub const fn allocation_matrix(&self) -> Mat4 {
        self.allocation
    }

    /// Allocate collective thrust and body torque to bounded motor targets.
    pub fn mix(
        &self,
        collective_thrust_n: f64,
        torque_body_nm: Vec3,
    ) -> Result<MixerResult, SimError> {
        ensure_nonnegative(collective_thrust_n, "collective_thrust_n")?;
        ensure_finite_vec3(torque_body_nm, "torque_body_nm")?;

        let requested = [
            collective_thrust_n,
            torque_body_nm[0],
            torque_body_nm[1],
            torque_body_nm[2],
        ];
        let unconstrained = matrix_vector_product_4(self.allocation_inverse, requested);
        let motor_target_thrust_n = unconstrained
            .map(|value| value.clamp(self.motor_min_thrust_n, self.motor_max_thrust_n));
        let achieved = matrix_vector_product_4(self.allocation, motor_target_thrust_n);
        let saturated = unconstrained.iter().zip(motor_target_thrust_n).any(
            |(requested_value, achieved_value)| {
                (requested_value - achieved_value).abs() > SATURATION_EPSILON
            },
        );

        Ok(MixerResult {
            motor_target_thrust_n,
            achieved_collective_thrust_n: achieved[0],
            achieved_torque_body_nm: [achieved[1], achieved[2], achieved[3]],
            saturated,
        })
    }
}

/// Transparent terms from the CTBR body-rate controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PidTerms {
    pub proportional_nm: Vec3,
    pub integral_nm: Vec3,
    pub derivative_nm: Vec3,
    pub output_torque_nm: Vec3,
}

/// Three-axis rate controller with derivative-on-measurement and conditional
/// anti-windup.  It intentionally follows the existing Python FlightStack
/// reference control seam so all pilot types share CTBR actuator semantics.
#[derive(Debug, Clone)]
pub struct BodyRateController {
    config: ControlConfig,
    integral: Vec3,
    previous_measurement: Option<Vec3>,
    filtered_derivative: Vec3,
}

impl BodyRateController {
    /// Initialize an empty rate-loop state from validated vehicle control data.
    pub fn new(config: &VehicleConfig) -> Result<Self, SimError> {
        config.validate()?;
        Ok(Self {
            config: config.control.clone(),
            integral: [0.0; 3],
            previous_measurement: None,
            filtered_derivative: [0.0; 3],
        })
    }

    /// Clear integral/derivative history at an episode or vehicle reset.
    pub fn reset(&mut self) {
        self.integral = [0.0; 3];
        self.previous_measurement = None;
        self.filtered_derivative = [0.0; 3];
    }

    /// Convert a shared CTBR command into an achieved/requested body torque.
    pub fn update(
        &mut self,
        command: PilotCommand,
        measured_body_rate_rad_s: Vec3,
        dt_s: f64,
    ) -> Result<PidTerms, SimError> {
        command.validate()?;
        ensure_finite_vec3(measured_body_rate_rad_s, "measured_body_rate_rad_s")?;
        ensure_positive(dt_s, "dt_s")?;

        let target_body_rate_rad_s = component_clamp(
            command.body_rate_rad_s,
            negate(self.config.max_body_rate_rad_s),
            self.config.max_body_rate_rad_s,
        );
        let error = subtract(target_body_rate_rad_s, measured_body_rate_rad_s);
        let proportional_nm = component_multiply(self.config.rate_kp, error);

        let raw_derivative = self.previous_measurement.map_or([0.0; 3], |previous| {
            scale(subtract(previous, measured_body_rate_rad_s), 1.0 / dt_s)
        });
        self.previous_measurement = Some(measured_body_rate_rad_s);

        let derivative_tau_s = 1.0 / (2.0 * std::f64::consts::PI * DERIVATIVE_CUTOFF_HZ);
        let derivative_alpha = dt_s / (derivative_tau_s + dt_s);
        self.filtered_derivative = add(
            self.filtered_derivative,
            scale(
                subtract(raw_derivative, self.filtered_derivative),
                derivative_alpha,
            ),
        );
        let derivative_nm = component_multiply(self.config.rate_kd, self.filtered_derivative);

        let proposed_integral = component_clamp(
            add(self.integral, scale(error, dt_s)),
            negate(self.config.rate_integral_limit),
            self.config.rate_integral_limit,
        );
        let proposed_output = add(
            add(
                proportional_nm,
                component_multiply(self.config.rate_ki, proposed_integral),
            ),
            derivative_nm,
        );
        for index in 0..3 {
            let pushing_high = proposed_output[index] > self.config.rate_torque_limit_nm[index]
                && error[index] > 0.0;
            let pushing_low = proposed_output[index] < -self.config.rate_torque_limit_nm[index]
                && error[index] < 0.0;
            if !pushing_high && !pushing_low {
                self.integral[index] = proposed_integral[index];
            }
        }

        let integral_nm = component_multiply(self.config.rate_ki, self.integral);
        let output_torque_nm = component_clamp(
            add(add(proportional_nm, integral_nm), derivative_nm),
            negate(self.config.rate_torque_limit_nm),
            self.config.rate_torque_limit_nm,
        );
        Ok(PidTerms {
            proportional_nm,
            integral_nm,
            derivative_nm,
            output_torque_nm,
        })
    }
}

/// Deterministic reference 6DOF multirotor plant.
///
/// Step order is fixed and documented: exact motor update, translational
/// semi-implicit Euler update, rotational body-rate update, then exact
/// quaternion body-rate integration.  It has no wall-clock dependency.
#[derive(Debug, Clone)]
pub struct Multirotor {
    config: VehicleConfig,
    state: FlightState,
    mixer: QuadMixer,
}

impl Multirotor {
    /// Construct a reference vehicle in a stationary level hover state at 1 m.
    pub fn new(config: VehicleConfig) -> Result<Self, SimError> {
        let state = FlightState::hovering(&config, 1.0)?;
        Self::from_state(config, state)
    }

    /// Construct a reference vehicle from a canonical initial state.
    pub fn from_state(config: VehicleConfig, state: FlightState) -> Result<Self, SimError> {
        config.validate()?;
        state.validate()?;
        for (index, thrust) in state.motor_thrust_n.iter().enumerate() {
            if *thrust < config.motor_min_thrust_n || *thrust > config.motor_max_thrust_n {
                return Err(SimError::InvalidInput(format!(
                    "initial motor_thrust_n[{index}] is outside configured motor bounds"
                )));
            }
        }
        let mixer = QuadMixer::new(&config)?;
        Ok(Self {
            config,
            state,
            mixer,
        })
    }

    /// Access the validated physical/control configuration.
    pub fn config(&self) -> &VehicleConfig {
        &self.config
    }

    /// Snapshot the canonical current state.
    pub const fn state(&self) -> FlightState {
        self.state
    }

    /// Access the vehicle's geometry-derived mixer.
    pub const fn mixer(&self) -> &QuadMixer {
        &self.mixer
    }

    /// Reset to a supplied canonical state, or use [`FlightState::hovering`]
    /// before calling this method when a default hover state is desired.
    pub fn reset(&mut self, state: FlightState) -> Result<FlightState, SimError> {
        state.validate()?;
        for (index, thrust) in state.motor_thrust_n.iter().enumerate() {
            if *thrust < self.config.motor_min_thrust_n || *thrust > self.config.motor_max_thrust_n
            {
                return Err(SimError::InvalidInput(format!(
                    "reset motor_thrust_n[{index}] is outside configured motor bounds"
                )));
            }
        }
        self.state = state;
        Ok(self.state)
    }

    /// Compute the current per-motor thrust force and moment in body frame.
    pub fn motor_wrench(&self) -> MotorWrench {
        accumulate_motor_wrench(
            self.state.motor_thrust_n,
            self.config.motor_position_body_m,
            self.config.motor_spin_direction,
            self.config.thrust_to_reaction_torque_m,
        )
    }

    /// Advance a motor-target command using no external disturbance.
    pub fn step_motor_targets_calm(
        &mut self,
        motor_target_thrust_n: MotorArray,
        dt_s: f64,
    ) -> Result<FlightState, SimError> {
        self.step_motor_targets(motor_target_thrust_n, dt_s, Disturbance::calm())
    }

    /// Advance one exact-motor/semi-implicit 6DOF integration step.
    pub fn step_motor_targets(
        &mut self,
        motor_target_thrust_n: MotorArray,
        dt_s: f64,
        disturbance: Disturbance,
    ) -> Result<FlightState, SimError> {
        ensure_positive(dt_s, "dt_s")?;
        ensure_finite_motor_array(motor_target_thrust_n, "motor_target_thrust_n")?;
        disturbance.validate()?;

        let next_motor_thrust_n = std::array::from_fn(|index| {
            let bounded_target = motor_target_thrust_n[index].clamp(
                self.config.motor_min_thrust_n,
                self.config.motor_max_thrust_n,
            );
            let effective_target = bounded_target * disturbance.motor_efficiency[index];
            step_motor_thrust(
                self.state.motor_thrust_n[index],
                effective_target,
                dt_s,
                self.config.motor_tau_rise_s,
                self.config.motor_tau_fall_s,
                self.config.motor_min_thrust_n,
                self.config.motor_max_thrust_n,
            )
        });

        let motor_wrench = accumulate_motor_wrench(
            next_motor_thrust_n,
            self.config.motor_position_body_m,
            self.config.motor_spin_direction,
            self.config.thrust_to_reaction_torque_m,
        );
        let thrust_world_n =
            rotate_body_to_world(self.state.q_body_to_world_wxyz, motor_wrench.force_body_n)?;
        let relative_velocity_body_m_s = rotate_world_to_body(
            self.state.q_body_to_world_wxyz,
            subtract(self.state.velocity_world_m_s, disturbance.wind_world_m_s),
        )?;
        let drag_body_n = negate(component_multiply(
            self.config.linear_drag_n_per_m_s,
            relative_velocity_body_m_s,
        ));
        let drag_world_n = rotate_body_to_world(self.state.q_body_to_world_wxyz, drag_body_n)?;
        let gravity_world_n = [0.0, 0.0, -self.config.mass_kg * self.config.gravity_m_s2];
        let acceleration_world_m_s2 = scale(
            add(
                add(add(thrust_world_n, gravity_world_n), drag_world_n),
                disturbance.force_world_n,
            ),
            1.0 / self.config.mass_kg,
        );
        let next_velocity_world_m_s = add(
            self.state.velocity_world_m_s,
            scale(acceleration_world_m_s2, dt_s),
        );
        let next_position_world_m = add(
            self.state.position_world_m,
            scale(next_velocity_world_m_s, dt_s),
        );

        let omega = self.state.body_rate_rad_s;
        let gyroscopic_nm = cross(
            omega,
            matrix_vector_product_3(self.config.inertia_kg_m2, omega),
        );
        let angular_drag_nm = negate(component_multiply(
            self.config.angular_drag_nm_per_rad_s,
            omega,
        ));
        let angular_acceleration_rad_s2 = solve_3x3(
            self.config.inertia_kg_m2,
            subtract(
                add(
                    add(motor_wrench.moment_body_nm, angular_drag_nm),
                    disturbance.torque_body_nm,
                ),
                gyroscopic_nm,
            ),
        )
        .ok_or(SimError::SingularInertia)?;
        let next_body_rate_rad_s = add(omega, scale(angular_acceleration_rad_s2, dt_s));
        let next_q_body_to_world_wxyz =
            integrate_body_rate(self.state.q_body_to_world_wxyz, next_body_rate_rad_s, dt_s)?;
        let next_state = FlightState::new(
            self.state.sim_time_s + dt_s,
            next_position_world_m,
            next_velocity_world_m_s,
            next_q_body_to_world_wxyz,
            next_body_rate_rad_s,
            next_motor_thrust_n,
        )?;
        self.state = next_state;
        Ok(self.state)
    }

    /// Execute the shared CTBR -> rate PID -> mixer -> motors -> plant chain.
    pub fn step_command(
        &mut self,
        command: PilotCommand,
        controller: &mut BodyRateController,
        dt_s: f64,
        disturbance: Disturbance,
    ) -> Result<(FlightState, MixerResult, PidTerms), SimError> {
        // Stage PID state so a rejected disturbance, allocation, or plant
        // step can be corrected and retried without consuming controller
        // history.
        let mut staged_controller = controller.clone();
        let terms = staged_controller.update(command, self.state.body_rate_rad_s, dt_s)?;
        let mixed = self
            .mixer
            .mix(command.collective_thrust_n, terms.output_torque_nm)?;
        let state = self.step_motor_targets(mixed.motor_target_thrust_n, dt_s, disturbance)?;
        *controller = staged_controller;
        Ok((state, mixed, terms))
    }
}

/// Fixed-rate owner of a vehicle and low-level controller, independent of
/// wall-clock time.  It is convenient for deterministic scenarios and parity
/// tests with a vectorized training backend.
#[derive(Debug, Clone)]
pub struct FixedStepRuntime {
    dt_s: f64,
    vehicle: Multirotor,
    controller: BodyRateController,
}

impl FixedStepRuntime {
    /// Start a level reference hover runtime at a fixed positive timestep.
    pub fn new(config: VehicleConfig, dt_s: f64) -> Result<Self, SimError> {
        let vehicle = Multirotor::new(config)?;
        let controller = BodyRateController::new(vehicle.config())?;
        Self::from_parts(vehicle, controller, dt_s)
    }

    /// Start from an explicit canonical state at a fixed positive timestep.
    pub fn from_state(
        config: VehicleConfig,
        state: FlightState,
        dt_s: f64,
    ) -> Result<Self, SimError> {
        let vehicle = Multirotor::from_state(config, state)?;
        let controller = BodyRateController::new(vehicle.config())?;
        Self::from_parts(vehicle, controller, dt_s)
    }

    fn from_parts(
        vehicle: Multirotor,
        controller: BodyRateController,
        dt_s: f64,
    ) -> Result<Self, SimError> {
        ensure_positive(dt_s, "dt_s")?;
        Ok(Self {
            dt_s,
            vehicle,
            controller,
        })
    }

    /// Fixed timestep in seconds.
    pub const fn dt_s(&self) -> f64 {
        self.dt_s
    }

    /// Current canonical state snapshot.
    pub const fn state(&self) -> FlightState {
        self.vehicle.state()
    }

    /// Reset plant and controller history together.
    pub fn reset(&mut self, state: FlightState) -> Result<FlightState, SimError> {
        let reset_state = self.vehicle.reset(state)?;
        self.controller.reset();
        Ok(reset_state)
    }

    /// Execute one nominal calm CTBR step.
    pub fn step_calm(
        &mut self,
        command: PilotCommand,
    ) -> Result<(FlightState, MixerResult, PidTerms), SimError> {
        self.step(command, Disturbance::calm())
    }

    /// Execute one CTBR step with a deterministic disturbance.
    pub fn step(
        &mut self,
        command: PilotCommand,
        disturbance: Disturbance,
    ) -> Result<(FlightState, MixerResult, PidTerms), SimError> {
        self.vehicle
            .step_command(command, &mut self.controller, self.dt_s, disturbance)
    }
}

fn ensure_positive(value: f64, name: &str) -> Result<(), SimError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(SimError::InvalidInput(format!(
            "{name} must be positive and finite"
        )))
    }
}

fn ensure_nonnegative(value: f64, name: &str) -> Result<(), SimError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(SimError::InvalidInput(format!(
            "{name} must be nonnegative and finite"
        )))
    }
}

fn ensure_finite_vec3(values: Vec3, name: &str) -> Result<(), SimError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SimError::InvalidInput(format!(
            "{name} must contain only finite values"
        )))
    }
}

fn ensure_finite_motor_array(values: MotorArray, name: &str) -> Result<(), SimError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(SimError::InvalidInput(format!(
            "{name} must contain only finite values"
        )))
    }
}

fn add(lhs: Vec3, rhs: Vec3) -> Vec3 {
    [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]]
}

fn subtract(lhs: Vec3, rhs: Vec3) -> Vec3 {
    [lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2]]
}

fn scale(vector: Vec3, scalar: f64) -> Vec3 {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn negate(vector: Vec3) -> Vec3 {
    [-vector[0], -vector[1], -vector[2]]
}

fn component_multiply(lhs: Vec3, rhs: Vec3) -> Vec3 {
    [lhs[0] * rhs[0], lhs[1] * rhs[1], lhs[2] * rhs[2]]
}

fn component_clamp(values: Vec3, lower: Vec3, upper: Vec3) -> Vec3 {
    [
        values[0].clamp(lower[0], upper[0]),
        values[1].clamp(lower[1], upper[1]),
        values[2].clamp(lower[2], upper[2]),
    ]
}

fn cross(lhs: Vec3, rhs: Vec3) -> Vec3 {
    [
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ]
}

fn matrix_vector_product_3(matrix: Mat3, vector: Vec3) -> Vec3 {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn matrix_vector_product_4(matrix: Mat4, vector: [f64; 4]) -> [f64; 4] {
    std::array::from_fn(|row| {
        matrix[row][0] * vector[0]
            + matrix[row][1] * vector[1]
            + matrix[row][2] * vector[2]
            + matrix[row][3] * vector[3]
    })
}

fn invert_4x4(matrix: Mat4) -> Option<Mat4> {
    let mut augmented = [[0.0; 8]; 4];
    for row in 0..4 {
        for column in 0..4 {
            augmented[row][column] = matrix[row][column];
        }
        augmented[row][row + 4] = 1.0;
    }

    for pivot_column in 0..4 {
        let mut pivot_row = pivot_column;
        for candidate_row in (pivot_column + 1)..4 {
            if augmented[candidate_row][pivot_column].abs()
                > augmented[pivot_row][pivot_column].abs()
            {
                pivot_row = candidate_row;
            }
        }
        let pivot = augmented[pivot_row][pivot_column];
        if !pivot.is_finite() || pivot.abs() < MATRIX_EPSILON {
            return None;
        }
        augmented.swap(pivot_column, pivot_row);
        for value in &mut augmented[pivot_column] {
            *value /= pivot;
        }
        let pivot_values = augmented[pivot_column];
        for (row_index, row) in augmented.iter_mut().enumerate() {
            if row_index == pivot_column {
                continue;
            }
            let factor = row[pivot_column];
            for (value, pivot_value) in row.iter_mut().zip(pivot_values) {
                *value -= factor * pivot_value;
            }
        }
    }

    let mut inverse = [[0.0; 4]; 4];
    for row in 0..4 {
        for column in 0..4 {
            inverse[row][column] = augmented[row][column + 4];
        }
    }
    Some(inverse)
}

fn solve_3x3(matrix: Mat3, rhs: Vec3) -> Option<Vec3> {
    let mut augmented = [[0.0; 4]; 3];
    for row in 0..3 {
        augmented[row][0..3].copy_from_slice(&matrix[row]);
        augmented[row][3] = rhs[row];
    }

    for pivot_column in 0..3 {
        let mut pivot_row = pivot_column;
        for candidate_row in (pivot_column + 1)..3 {
            if augmented[candidate_row][pivot_column].abs()
                > augmented[pivot_row][pivot_column].abs()
            {
                pivot_row = candidate_row;
            }
        }
        let pivot = augmented[pivot_row][pivot_column];
        if !pivot.is_finite() || pivot.abs() < MATRIX_EPSILON {
            return None;
        }
        augmented.swap(pivot_column, pivot_row);
        for value in augmented[pivot_column].iter_mut().skip(pivot_column) {
            *value /= pivot;
        }
        let pivot_values = augmented[pivot_column];
        for (row_index, row) in augmented.iter_mut().enumerate() {
            if row_index == pivot_column {
                continue;
            }
            let factor = row[pivot_column];
            for (value, pivot_value) in row
                .iter_mut()
                .skip(pivot_column)
                .zip(pivot_values.iter().skip(pivot_column))
            {
                *value -= factor * pivot_value;
            }
        }
    }
    Some([augmented[0][3], augmented[1][3], augmented[2][3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1.0e-10;

    fn config() -> VehicleConfig {
        VehicleConfig::reference_5in().expect("tracked shared config is valid")
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < EPSILON,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_vec_close(actual: Vec3, expected: Vec3) {
        for index in 0..3 {
            assert_close(actual[index], expected[index]);
        }
    }

    #[test]
    fn motor_response_is_exact_bounded_and_directional() {
        let vehicle_config = config();
        let target = 3.0;
        let dt_s = 0.01;
        let rising = step_motor_thrust(
            0.0,
            target,
            dt_s,
            vehicle_config.motor_tau_rise_s,
            vehicle_config.motor_tau_fall_s,
            vehicle_config.motor_min_thrust_n,
            vehicle_config.motor_max_thrust_n,
        );
        assert_close(
            rising,
            target * (1.0 - (-dt_s / vehicle_config.motor_tau_rise_s).exp()),
        );
        assert!(rising > 0.0 && rising < target);

        let falling = step_motor_thrust(
            rising,
            0.0,
            dt_s,
            vehicle_config.motor_tau_rise_s,
            vehicle_config.motor_tau_fall_s,
            vehicle_config.motor_min_thrust_n,
            vehicle_config.motor_max_thrust_n,
        );
        assert_close(
            falling,
            rising * (-dt_s / vehicle_config.motor_tau_fall_s).exp(),
        );
        assert!(falling < rising && falling > 0.0);

        assert_close(
            step_motor_thrust(
                rising,
                vehicle_config.motor_max_thrust_n * 2.0,
                0.0,
                vehicle_config.motor_tau_rise_s,
                vehicle_config.motor_tau_fall_s,
                vehicle_config.motor_min_thrust_n,
                vehicle_config.motor_max_thrust_n,
            ),
            rising,
        );
        assert_close(
            step_motor_thrust(0.0, 100.0, 0.1, 0.0, 0.0, 0.0, 14.0),
            14.0,
        );
    }

    #[test]
    fn symmetric_and_differential_wrenches_have_documented_signs() {
        let vehicle_config = config();
        let hover_per_motor = vehicle_config.hover_thrust_n() / 4.0;
        let symmetric = accumulate_motor_wrench(
            [hover_per_motor; 4],
            vehicle_config.motor_position_body_m,
            vehicle_config.motor_spin_direction,
            vehicle_config.thrust_to_reaction_torque_m,
        );
        assert_close(symmetric.force_body_n[2], vehicle_config.hover_thrust_n());
        assert_vec_close(symmetric.moment_body_nm, [0.0; 3]);

        let roll = accumulate_motor_wrench(
            [4.0, 2.0, 2.0, 4.0],
            vehicle_config.motor_position_body_m,
            vehicle_config.motor_spin_direction,
            vehicle_config.thrust_to_reaction_torque_m,
        );
        assert!(roll.moment_body_nm[0] > 0.0);

        let pitch = accumulate_motor_wrench(
            [2.0, 4.0, 4.0, 2.0],
            vehicle_config.motor_position_body_m,
            vehicle_config.motor_spin_direction,
            vehicle_config.thrust_to_reaction_torque_m,
        );
        assert!(pitch.moment_body_nm[1] > 0.0);

        let yaw = accumulate_motor_wrench(
            [4.0, 2.0, 4.0, 2.0],
            vehicle_config.motor_position_body_m,
            vehicle_config.motor_spin_direction,
            vehicle_config.thrust_to_reaction_torque_m,
        );
        assert!(yaw.moment_body_nm[2] > 0.0);
    }

    #[test]
    fn mixer_uses_geometry_and_exposes_saturation() {
        let vehicle_config = config();
        let mixer = QuadMixer::new(&vehicle_config).expect("valid configured mixer");
        for requested_wrench in [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ] {
            let motor_thrust = matrix_vector_product_4(mixer.allocation_inverse, requested_wrench);
            let reconstructed = matrix_vector_product_4(mixer.allocation, motor_thrust);
            for (actual, expected) in reconstructed.iter().zip(requested_wrench) {
                assert_close(*actual, expected);
            }
        }

        let equal = mixer
            .mix(vehicle_config.hover_thrust_n(), [0.0; 3])
            .expect("hover allocation");
        for thrust in equal.motor_target_thrust_n {
            assert_close(thrust, vehicle_config.hover_thrust_n() / 4.0);
        }
        assert_vec_close(equal.achieved_torque_body_nm, [0.0; 3]);
        assert!(!equal.saturated);

        let roll = mixer
            .mix(vehicle_config.hover_thrust_n(), [0.03, 0.0, 0.0])
            .expect("roll allocation");
        assert_close(
            roll.achieved_collective_thrust_n,
            vehicle_config.hover_thrust_n(),
        );
        assert_vec_close(roll.achieved_torque_body_nm, [0.03, 0.0, 0.0]);
        assert!(roll.motor_target_thrust_n[0] > roll.motor_target_thrust_n[2]);

        let pitch = mixer
            .mix(vehicle_config.hover_thrust_n(), [0.0, 0.03, 0.0])
            .expect("pitch allocation");
        assert_close(
            pitch.achieved_collective_thrust_n,
            vehicle_config.hover_thrust_n(),
        );
        assert_vec_close(pitch.achieved_torque_body_nm, [0.0, 0.03, 0.0]);
        assert!(pitch.motor_target_thrust_n[1] > pitch.motor_target_thrust_n[0]);

        let yaw = mixer
            .mix(vehicle_config.hover_thrust_n(), [0.0, 0.0, 0.03])
            .expect("yaw allocation");
        assert_close(
            yaw.achieved_collective_thrust_n,
            vehicle_config.hover_thrust_n(),
        );
        assert_vec_close(yaw.achieved_torque_body_nm, [0.0, 0.0, 0.03]);
        assert!(yaw.motor_target_thrust_n[0] > yaw.motor_target_thrust_n[1]);

        let saturated = mixer
            .mix(vehicle_config.motor_max_thrust_n * 4.0, [1.0; 3])
            .expect("bounded allocation");
        assert!(saturated.saturated);
        assert!(saturated
            .motor_target_thrust_n
            .iter()
            .all(|thrust| *thrust <= vehicle_config.motor_max_thrust_n));
    }

    #[test]
    fn free_fall_and_hover_follow_reference_6dof_equations() {
        let vehicle_config = config();
        let free_fall_state = FlightState::new(
            0.0,
            [0.0, 0.0, 10.0],
            [0.0; 3],
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 4],
        )
        .expect("valid falling state");
        let mut free_fall =
            Multirotor::from_state(vehicle_config.clone(), free_fall_state).expect("vehicle");
        let falling = free_fall
            .step_motor_targets_calm([0.0; 4], 0.1)
            .expect("free fall step");
        assert_close(
            falling.velocity_world_m_s[2],
            -vehicle_config.gravity_m_s2 * 0.1,
        );
        assert_close(
            falling.position_world_m[2],
            10.0 - vehicle_config.gravity_m_s2 * 0.1 * 0.1,
        );

        let mut hover = Multirotor::new(vehicle_config.clone()).expect("vehicle");
        let before = hover.state();
        let after = hover
            .step_motor_targets_calm([vehicle_config.hover_thrust_n() / 4.0; 4], 0.002)
            .expect("hover step");
        assert_vec_close(after.velocity_world_m_s, before.velocity_world_m_s);
        assert_vec_close(after.body_rate_rad_s, [0.0; 3]);
    }

    #[test]
    fn non_identity_attitude_rotates_thrust_into_world_frame() {
        let vehicle_config = config();
        let q_body_to_world_wxyz = flightstack_core::quaternion::from_axis_angle(
            [1.0, 0.0, 0.0],
            std::f64::consts::FRAC_PI_2,
        )
        .expect("valid quarter turn");
        let state = FlightState::new(
            0.0,
            [0.0; 3],
            [0.0; 3],
            q_body_to_world_wxyz,
            [0.0; 3],
            [vehicle_config.hover_thrust_n() / 4.0; 4],
        )
        .expect("tilted state");
        let mut vehicle =
            Multirotor::from_state(vehicle_config.clone(), state).expect("tilted vehicle");
        let dt_s = 0.01;
        let after = vehicle
            .step_motor_targets_calm([vehicle_config.hover_thrust_n() / 4.0; 4], dt_s)
            .expect("tilted step");
        assert_vec_close(
            after.velocity_world_m_s,
            [
                0.0,
                -vehicle_config.gravity_m_s2 * dt_s,
                -vehicle_config.gravity_m_s2 * dt_s,
            ],
        );
    }

    #[test]
    fn disturbance_and_motor_failure_hooks_are_deterministic() {
        let vehicle_config = config();
        let zero_motor_state = FlightState::new(
            0.0,
            [0.0; 3],
            [0.0; 3],
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 4],
        )
        .expect("valid state");
        let mut vehicle =
            Multirotor::from_state(vehicle_config.clone(), zero_motor_state).expect("vehicle");
        let degraded = Disturbance {
            motor_efficiency: [0.5, 1.0, 1.0, 1.0],
            ..Disturbance::calm()
        };
        let state = vehicle
            .step_motor_targets([4.0; 4], 1.0, degraded)
            .expect("degraded motor step");
        assert_close(state.motor_thrust_n[0], 2.0);
        assert_close(state.motor_thrust_n[1], 4.0);

        let mut pushed = Multirotor::new(vehicle_config.clone()).expect("vehicle");
        let with_force = pushed
            .step_motor_targets(
                [vehicle_config.hover_thrust_n() / 4.0; 4],
                0.1,
                Disturbance {
                    force_world_n: [1.0, 0.0, 0.0],
                    ..Disturbance::calm()
                },
            )
            .expect("world force step");
        assert_close(
            with_force.velocity_world_m_s[0],
            0.1 / vehicle_config.mass_kg,
        );
    }

    #[test]
    fn fixed_step_ctbr_chain_is_deterministic_and_damps_rate() {
        let vehicle_config = config();
        let command = PilotCommand::new(vehicle_config.hover_thrust_n(), [0.4, -0.25, 0.1])
            .expect("valid command");

        fn run(vehicle_config: &VehicleConfig, command: PilotCommand) -> FlightState {
            let mut runtime =
                FixedStepRuntime::new(vehicle_config.clone(), 0.002).expect("runtime");
            for _ in 0..250 {
                runtime.step_calm(command).expect("deterministic step");
            }
            runtime.state()
        }

        assert_eq!(run(&vehicle_config, command), run(&vehicle_config, command));

        let initial_state = FlightState::new(
            0.0,
            [0.0, 0.0, 1.0],
            [0.0; 3],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, -0.8, 0.5],
            [vehicle_config.hover_thrust_n() / 4.0; 4],
        )
        .expect("valid spinning state");
        let mut runtime =
            FixedStepRuntime::from_state(vehicle_config.clone(), initial_state, 0.002)
                .expect("runtime");
        let initial_norm = vector_norm(runtime.state().body_rate_rad_s);
        let hover = PilotCommand::hover(&vehicle_config).expect("hover command");
        for _ in 0..1_500 {
            runtime.step_calm(hover).expect("rate-control step");
        }
        assert!(vector_norm(runtime.state().body_rate_rad_s) < initial_norm * 0.2);
    }

    #[test]
    fn controller_saturation_prevents_integral_windup() {
        let vehicle_config = config();
        let mut controller = BodyRateController::new(&vehicle_config).expect("controller");
        let command = PilotCommand::new(vehicle_config.hover_thrust_n(), [100.0, 0.0, 0.0])
            .expect("finite command");
        let mut terms = controller
            .update(command, [0.0; 3], 0.002)
            .expect("saturated update");
        for _ in 0..100 {
            terms = controller
                .update(command, [0.0; 3], 0.002)
                .expect("saturated update");
        }
        assert_close(
            terms.output_torque_nm[0],
            vehicle_config.control.rate_torque_limit_nm[0],
        );
        assert_vec_close(terms.integral_nm, [0.0; 3]);
    }

    #[test]
    fn rejected_ctbr_step_preserves_controller_history_for_retry() {
        let vehicle_config = config();
        let initial_state = FlightState::hovering(&vehicle_config, 1.0).expect("hover state");
        let command =
            PilotCommand::new(vehicle_config.hover_thrust_n(), [0.4, -0.25, 0.1]).expect("command");

        let mut expected_vehicle =
            Multirotor::from_state(vehicle_config.clone(), initial_state).expect("vehicle");
        let mut expected_controller = BodyRateController::new(&vehicle_config).expect("controller");
        let expected = expected_vehicle
            .step_command(
                command,
                &mut expected_controller,
                0.002,
                Disturbance::calm(),
            )
            .expect("baseline step");

        let mut retried_vehicle =
            Multirotor::from_state(vehicle_config.clone(), initial_state).expect("vehicle");
        let mut retried_controller = BodyRateController::new(&vehicle_config).expect("controller");
        assert!(retried_vehicle
            .step_command(
                command,
                &mut retried_controller,
                0.002,
                Disturbance {
                    motor_efficiency: [1.0, 1.1, 1.0, 1.0],
                    ..Disturbance::calm()
                },
            )
            .is_err());
        assert_eq!(retried_vehicle.state(), initial_state);
        let retried = retried_vehicle
            .step_command(command, &mut retried_controller, 0.002, Disturbance::calm())
            .expect("retried step");
        assert_eq!(retried, expected);
    }

    #[test]
    fn rejected_runtime_reset_preserves_controller_history() {
        let vehicle_config = config();
        let command =
            PilotCommand::new(vehicle_config.hover_thrust_n(), [0.4, -0.25, 0.1]).expect("command");
        let mut baseline =
            FixedStepRuntime::new(vehicle_config.clone(), 0.002).expect("baseline runtime");
        baseline.step_calm(command).expect("first step");
        let mut retried = baseline.clone();
        let mut invalid_state = retried.state();
        invalid_state.q_body_to_world_wxyz = [0.0; 4];
        assert!(retried.reset(invalid_state).is_err());

        let expected = baseline.step_calm(command).expect("baseline retry");
        let actual = retried.step_calm(command).expect("preserved retry");
        assert_eq!(actual, expected);
    }

    #[test]
    fn malformed_disturbance_is_rejected_without_advancing_state() {
        let vehicle_config = config();
        let mut vehicle = Multirotor::new(vehicle_config.clone()).expect("vehicle");
        let before = vehicle.state();
        let result = vehicle.step_motor_targets(
            [vehicle_config.hover_thrust_n() / 4.0; 4],
            0.002,
            Disturbance {
                motor_efficiency: [1.0, 1.1, 1.0, 1.0],
                ..Disturbance::calm()
            },
        );
        assert!(result.is_err());
        assert_eq!(vehicle.state(), before);
    }

    fn vector_norm(vector: Vec3) -> f64 {
        (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
    }
}
