//! FlightStack's canonical runtime contracts.
//!
//! The orientation convention is intentionally part of the public API:
//! quaternions are scalar-first `[w, x, y, z]` and rotate body-frame vectors
//! into the world frame.  The world frame is right-handed with `+Z` up; the
//! vehicle body frame is FLU (`+X` forward, `+Y` left, `+Z` up).

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

pub mod quaternion;

/// A vector expressed in FlightStack's explicitly documented frame.
pub type Vec3 = [f64; 3];
/// A scalar-first quaternion that maps a body-frame vector into world frame.
pub type QuatWxyz = [f64; 4];
/// A 3 by 3 inertia matrix in body coordinates.
pub type Mat3 = [[f64; 3]; 3];
/// Per-rotor values in FlightStack's front-left, rear-left, rear-right,
/// front-right ordering.
pub type MotorArray = [f64; 4];

/// Contract-validation error shared by state, command, and quaternion APIs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError {
    message: String,
}

impl ContractError {
    /// Construct a validation error with a stable, user-facing explanation.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for ContractError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for ContractError {}

/// Errors emitted while loading or validating a vehicle TOML file.
#[derive(Debug)]
pub enum ConfigError {
    /// The TOML could not be parsed into FlightStack's schema.
    Parse(toml::de::Error),
    /// A caller-supplied TOML file could not be read.
    Io(std::io::Error),
    /// The parsed physical/control values violate an explicit invariant.
    Invalid(ContractError),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(error) => write!(formatter, "invalid vehicle TOML: {error}"),
            Self::Io(error) => write!(formatter, "unable to read vehicle TOML: {error}"),
            Self::Invalid(error) => write!(formatter, "invalid vehicle configuration: {error}"),
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(error) => Some(error),
            Self::Io(error) => Some(error),
            Self::Invalid(error) => Some(error),
        }
    }
}

/// CTBR rate-loop values held beside the vehicle's physical parameters.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ControlConfig {
    pub rate_kp: Vec3,
    pub rate_ki: Vec3,
    pub rate_kd: Vec3,
    pub rate_torque_limit_nm: Vec3,
    pub rate_integral_limit: Vec3,
    pub max_body_rate_rad_s: Vec3,
}

impl ControlConfig {
    /// Check that all gain and limiting values are finite and physically sane.
    pub fn validate(&self) -> Result<(), ContractError> {
        for (name, values) in [
            ("control.rate_kp", self.rate_kp),
            ("control.rate_ki", self.rate_ki),
            ("control.rate_kd", self.rate_kd),
            ("control.rate_torque_limit_nm", self.rate_torque_limit_nm),
            ("control.rate_integral_limit", self.rate_integral_limit),
            ("control.max_body_rate_rad_s", self.max_body_rate_rad_s),
        ] {
            ensure_finite_vec3(values, name)?;
            if values.iter().any(|value| *value < 0.0) {
                return Err(ContractError::new(format!("{name} must be nonnegative")));
            }
        }

        if self.rate_torque_limit_nm.iter().any(|value| *value <= 0.0) {
            return Err(ContractError::new(
                "control.rate_torque_limit_nm must be strictly positive",
            ));
        }
        if self.rate_integral_limit.iter().any(|value| *value <= 0.0) {
            return Err(ContractError::new(
                "control.rate_integral_limit must be strictly positive",
            ));
        }
        if self.max_body_rate_rad_s.iter().any(|value| *value <= 0.0) {
            return Err(ContractError::new(
                "control.max_body_rate_rad_s must be strictly positive",
            ));
        }
        Ok(())
    }
}

/// One serializable source of truth for FlightStack's reference vehicle.
///
/// This schema mirrors `config/vehicles/flightstack_5in.toml`.  The current
/// values are reference-derived / simulation-tuned starter values, not claims
/// of measured hardware performance.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VehicleConfig {
    pub name: String,
    pub version: String,
    pub mass_kg: f64,
    pub inertia_kg_m2: Mat3,
    pub motor_position_body_m: [Vec3; 4],
    pub motor_spin_direction: MotorArray,
    pub motor_min_thrust_n: f64,
    pub motor_max_thrust_n: f64,
    pub motor_tau_rise_s: f64,
    pub motor_tau_fall_s: f64,
    pub thrust_to_reaction_torque_m: f64,
    pub linear_drag_n_per_m_s: Vec3,
    pub angular_drag_nm_per_rad_s: Vec3,
    pub gravity_m_s2: f64,
    pub control: ControlConfig,
}

impl VehicleConfig {
    /// The tracked TOML content used by the Python and Rust reference paths.
    pub const REFERENCE_5IN_TOML: &'static str =
        include_str!("../../../../config/vehicles/flightstack_5in.toml");

    /// Parse and validate a vehicle config from TOML text.
    pub fn from_toml_str(toml_text: &str) -> Result<Self, ConfigError> {
        let config = toml::from_str::<Self>(toml_text).map_err(ConfigError::Parse)?;
        config.validate().map_err(ConfigError::Invalid)?;
        Ok(config)
    }

    /// Load and validate a user-selected vehicle TOML file.
    pub fn from_toml_path(path: impl AsRef<Path>) -> Result<Self, ConfigError> {
        let toml_text = std::fs::read_to_string(path).map_err(ConfigError::Io)?;
        Self::from_toml_str(&toml_text)
    }

    /// Load the repository's canonical, shared 5-inch reference config.
    pub fn reference_5in() -> Result<Self, ConfigError> {
        Self::from_toml_str(Self::REFERENCE_5IN_TOML)
    }

    /// Hover thrust in newtons under the configured standard gravity.
    pub fn hover_thrust_n(&self) -> f64 {
        self.mass_kg * self.gravity_m_s2
    }

    /// Validate physical, geometry, and control invariants before simulation.
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.name.trim().is_empty() || self.version.trim().is_empty() {
            return Err(ContractError::new(
                "vehicle name and version must be nonempty",
            ));
        }
        ensure_positive(self.mass_kg, "mass_kg")?;
        ensure_spd_matrix(self.inertia_kg_m2, "inertia_kg_m2")?;

        for (index, position) in self.motor_position_body_m.iter().enumerate() {
            ensure_finite_vec3(*position, &format!("motor_position_body_m[{index}]"))?;
        }
        for (index, direction) in self.motor_spin_direction.iter().enumerate() {
            if !direction.is_finite() || (*direction != -1.0 && *direction != 1.0) {
                return Err(ContractError::new(format!(
                    "motor_spin_direction[{index}] must be exactly -1 or +1"
                )));
            }
        }

        ensure_nonnegative(self.motor_min_thrust_n, "motor_min_thrust_n")?;
        ensure_positive(self.motor_max_thrust_n, "motor_max_thrust_n")?;
        if self.motor_max_thrust_n <= self.motor_min_thrust_n {
            return Err(ContractError::new(
                "motor_max_thrust_n must exceed motor_min_thrust_n",
            ));
        }
        ensure_positive(self.motor_tau_rise_s, "motor_tau_rise_s")?;
        ensure_positive(self.motor_tau_fall_s, "motor_tau_fall_s")?;
        ensure_nonnegative(
            self.thrust_to_reaction_torque_m,
            "thrust_to_reaction_torque_m",
        )?;
        ensure_nonnegative_vec3(self.linear_drag_n_per_m_s, "linear_drag_n_per_m_s")?;
        ensure_nonnegative_vec3(self.angular_drag_nm_per_rad_s, "angular_drag_nm_per_rad_s")?;
        ensure_positive(self.gravity_m_s2, "gravity_m_s2")?;
        self.control.validate()
    }
}

/// Canonical 6DOF state of a four-motor FlightStack vehicle.
///
/// All position/velocity values are world-frame.  Body rate and motor wrench
/// terms are body-frame.  Use [`FlightState::new`] or [`FlightState::hovering`]
/// to normalize the quaternion at a system boundary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct FlightState {
    pub sim_time_s: f64,
    pub position_world_m: Vec3,
    pub velocity_world_m_s: Vec3,
    pub q_body_to_world_wxyz: QuatWxyz,
    pub body_rate_rad_s: Vec3,
    pub motor_thrust_n: MotorArray,
}

impl FlightState {
    /// Construct and canonicalize a valid state.
    pub fn new(
        sim_time_s: f64,
        position_world_m: Vec3,
        velocity_world_m_s: Vec3,
        q_body_to_world_wxyz: QuatWxyz,
        body_rate_rad_s: Vec3,
        motor_thrust_n: MotorArray,
    ) -> Result<Self, ContractError> {
        let state = Self {
            sim_time_s,
            position_world_m,
            velocity_world_m_s,
            q_body_to_world_wxyz: quaternion::normalize(q_body_to_world_wxyz)?,
            body_rate_rad_s,
            motor_thrust_n,
        };
        state.validate()?;
        Ok(state)
    }

    /// Construct a level, stationary reference hover state at `altitude_m`.
    pub fn hovering(config: &VehicleConfig, altitude_m: f64) -> Result<Self, ContractError> {
        config.validate()?;
        ensure_finite(altitude_m, "altitude_m")?;
        Self::new(
            0.0,
            [0.0, 0.0, altitude_m],
            [0.0; 3],
            [1.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            [config.hover_thrust_n() / 4.0; 4],
        )
    }

    /// Check scalar/vector constraints without changing stored values.
    pub fn validate(&self) -> Result<(), ContractError> {
        ensure_nonnegative(self.sim_time_s, "sim_time_s")?;
        ensure_finite_vec3(self.position_world_m, "position_world_m")?;
        ensure_finite_vec3(self.velocity_world_m_s, "velocity_world_m_s")?;
        ensure_finite_vec3(self.body_rate_rad_s, "body_rate_rad_s")?;
        let norm = quaternion::norm(self.q_body_to_world_wxyz)?;
        if (norm - 1.0).abs() > 1.0e-9 {
            return Err(ContractError::new(
                "q_body_to_world_wxyz must be normalized at a state boundary",
            ));
        }
        for (index, thrust) in self.motor_thrust_n.iter().enumerate() {
            ensure_nonnegative(*thrust, &format!("motor_thrust_n[{index}]"))?;
        }
        Ok(())
    }

    /// Return a copy with a normalized attitude after externally editing state.
    pub fn canonicalized(mut self) -> Result<Self, ContractError> {
        self.q_body_to_world_wxyz = quaternion::normalize(self.q_body_to_world_wxyz)?;
        self.validate()?;
        Ok(self)
    }
}

/// Shared collective-thrust/body-rate (CTBR) command for human, classical, and
/// learned pilot implementations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct PilotCommand {
    pub collective_thrust_n: f64,
    pub body_rate_rad_s: Vec3,
}

impl PilotCommand {
    /// Construct a finite, nonnegative-collective CTBR command.
    pub fn new(collective_thrust_n: f64, body_rate_rad_s: Vec3) -> Result<Self, ContractError> {
        let command = Self {
            collective_thrust_n,
            body_rate_rad_s,
        };
        command.validate()?;
        Ok(command)
    }

    /// Level-hover command for a config's reference gravity.
    pub fn hover(config: &VehicleConfig) -> Result<Self, ContractError> {
        Self::new(config.hover_thrust_n(), [0.0; 3])
    }

    /// Validate command values at a pilot/runtime boundary.
    pub fn validate(&self) -> Result<(), ContractError> {
        ensure_nonnegative(self.collective_thrust_n, "collective_thrust_n")?;
        ensure_finite_vec3(self.body_rate_rad_s, "body_rate_rad_s")
    }
}

pub(crate) fn ensure_finite(value: f64, name: &str) -> Result<(), ContractError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ContractError::new(format!("{name} must be finite")))
    }
}

pub(crate) fn ensure_positive(value: f64, name: &str) -> Result<(), ContractError> {
    ensure_finite(value, name)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(ContractError::new(format!("{name} must be positive")))
    }
}

pub(crate) fn ensure_nonnegative(value: f64, name: &str) -> Result<(), ContractError> {
    ensure_finite(value, name)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(ContractError::new(format!("{name} must be nonnegative")))
    }
}

pub(crate) fn ensure_finite_vec3(values: Vec3, name: &str) -> Result<(), ContractError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ContractError::new(format!(
            "{name} must contain only finite values"
        )))
    }
}

fn ensure_nonnegative_vec3(values: Vec3, name: &str) -> Result<(), ContractError> {
    ensure_finite_vec3(values, name)?;
    if values.iter().all(|value| *value >= 0.0) {
        Ok(())
    } else {
        Err(ContractError::new(format!("{name} must be nonnegative")))
    }
}

fn ensure_spd_matrix(matrix: Mat3, name: &str) -> Result<(), ContractError> {
    for row in matrix {
        ensure_finite_vec3(row, name)?;
    }
    for (row, values) in matrix.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            if (*value - matrix[column][row]).abs() > 1.0e-12 {
                return Err(ContractError::new(format!("{name} must be symmetric")));
            }
        }
    }

    // Sylvester's criterion for a symmetric 3 by 3 inertia tensor.
    let leading_1 = matrix[0][0];
    let leading_2 = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let determinant = determinant_3x3(matrix);
    if leading_1 <= 0.0 || leading_2 <= 0.0 || determinant <= 0.0 {
        return Err(ContractError::new(format!(
            "{name} must be positive definite"
        )));
    }
    Ok(())
}

fn determinant_3x3(matrix: Mat3) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_reference_config_loads_and_has_hover_thrust() {
        let config = VehicleConfig::reference_5in().expect("tracked TOML is valid");
        assert_eq!(config.name, "flightstack_5in");
        assert_eq!(config.version, "2026-08-v1");
        assert!((config.hover_thrust_n() - 6.374_322_5).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_inertia_is_rejected() {
        let mut config = VehicleConfig::reference_5in().expect("tracked TOML is valid");
        config.inertia_kg_m2 = [[0.002, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0035]];
        let error = config.validate().expect_err("non-SPD inertia must fail");
        assert!(error.to_string().contains("positive definite"));
    }

    #[test]
    fn state_constructor_normalizes_quaternion() {
        let state = FlightState::new(
            0.0,
            [0.0; 3],
            [0.0; 3],
            [2.0, 0.0, 0.0, 0.0],
            [0.0; 3],
            [0.0; 4],
        )
        .expect("valid state");
        assert_eq!(state.q_body_to_world_wxyz, [1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn pilot_command_rejects_negative_collective() {
        let error = PilotCommand::new(-0.01, [0.0; 3]).expect_err("negative thrust is invalid");
        assert!(error.to_string().contains("collective_thrust_n"));
    }
}
