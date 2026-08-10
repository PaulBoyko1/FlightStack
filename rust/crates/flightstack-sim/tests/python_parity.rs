//! Shared Python/Rust trajectory parity for the deterministic 6DOF runtime.

use std::path::PathBuf;

use flightstack_core::{FlightState, PilotCommand, VehicleConfig};
use flightstack_sim::{Disturbance, FixedStepRuntime};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    vehicle_config_path: String,
    dt_s: f64,
    tolerances: Tolerances,
    initial_state: StateFixture,
    disturbance: DisturbanceFixture,
    commands: Vec<CommandFixture>,
    expected_steps: Vec<StateFixture>,
}

#[derive(Deserialize)]
struct Tolerances {
    state_abs: f64,
    quaternion_abs: f64,
}

#[derive(Deserialize)]
struct StateFixture {
    sim_time_s: f64,
    position_world_m: [f64; 3],
    velocity_world_m_s: [f64; 3],
    q_body_to_world_wxyz: [f64; 4],
    body_rate_rad_s: [f64; 3],
    motor_thrust_n: [f64; 4],
}

#[derive(Deserialize)]
struct DisturbanceFixture {
    force_world_n: [f64; 3],
    torque_body_nm: [f64; 3],
    wind_world_m_s: [f64; 3],
    motor_efficiency: [f64; 4],
}

#[derive(Deserialize)]
struct CommandFixture {
    collective_thrust_n: f64,
    body_rate_rad_s: [f64; 3],
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .expect("flightstack-sim remains inside the tracked workspace")
}

fn read_fixture() -> Fixture {
    let path = repository_root().join("tests/data/python_rust_6dof_ctbr_v1.toml");
    let content = std::fs::read_to_string(path).expect("shared parity fixture is readable");
    toml::from_str(&content).expect("shared parity fixture remains valid TOML")
}

fn state_from_fixture(source: &StateFixture) -> FlightState {
    FlightState::new(
        source.sim_time_s,
        source.position_world_m,
        source.velocity_world_m_s,
        source.q_body_to_world_wxyz,
        source.body_rate_rad_s,
        source.motor_thrust_n,
    )
    .expect("fixture state meets the canonical FlightState contract")
}

fn disturbance_from_fixture(source: &DisturbanceFixture) -> Disturbance {
    Disturbance {
        force_world_n: source.force_world_n,
        torque_body_nm: source.torque_body_nm,
        wind_world_m_s: source.wind_world_m_s,
        motor_efficiency: source.motor_efficiency,
    }
}

fn command_from_fixture(source: &CommandFixture) -> PilotCommand {
    PilotCommand::new(source.collective_thrust_n, source.body_rate_rad_s)
        .expect("fixture command meets the canonical PilotCommand contract")
}

fn max_abs_difference<const N: usize>(actual: [f64; N], expected: [f64; N]) -> f64 {
    actual
        .into_iter()
        .zip(expected)
        .map(|(actual_value, expected_value)| (actual_value - expected_value).abs())
        .fold(0.0, f64::max)
}

fn record_scalar(
    label: &str,
    step: usize,
    actual: f64,
    expected: f64,
    tolerance: f64,
    max_divergence: &mut f64,
) {
    let divergence = (actual - expected).abs();
    *max_divergence = max_divergence.max(divergence);
    assert!(
        divergence <= tolerance,
        "step {step} {label} divergence {divergence:e} exceeds {tolerance:e}; actual={actual:.17e}, expected={expected:.17e}"
    );
}

fn record_vector<const N: usize>(
    label: &str,
    step: usize,
    actual: [f64; N],
    expected: [f64; N],
    tolerance: f64,
    max_divergence: &mut f64,
) {
    let divergence = max_abs_difference(actual, expected);
    *max_divergence = max_divergence.max(divergence);
    assert!(
        divergence <= tolerance,
        "step {step} {label} max divergence {divergence:e} exceeds {tolerance:e}; actual={actual:?}, expected={expected:?}"
    );
}

fn record_quaternion(
    step: usize,
    actual: [f64; 4],
    expected: [f64; 4],
    tolerance: f64,
    max_divergence: &mut f64,
) {
    let direct = max_abs_difference(actual, expected);
    let opposite_sign = max_abs_difference(actual, expected.map(|value| -value));
    let divergence = direct.min(opposite_sign);
    *max_divergence = max_divergence.max(divergence);
    assert!(
        divergence <= tolerance,
        "step {step} q_body_to_world_wxyz sign-invariant divergence {divergence:e} exceeds {tolerance:e}; actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn rust_fixed_step_runtime_matches_python_6dof_ctbr_fixture() {
    let fixture = read_fixture();
    assert_eq!(fixture.name, "python-rust-6dof-ctbr-disturbance-v1");
    assert_eq!(
        fixture.vehicle_config_path,
        "config/vehicles/flightstack_5in.toml"
    );
    assert!(fixture.tolerances.state_abs.is_finite() && fixture.tolerances.state_abs > 0.0);
    assert!(
        fixture.tolerances.quaternion_abs.is_finite() && fixture.tolerances.quaternion_abs > 0.0
    );
    assert_eq!(fixture.commands.len(), fixture.expected_steps.len());
    assert_eq!(fixture.commands.len(), 6);

    let config =
        VehicleConfig::from_toml_path(repository_root().join(&fixture.vehicle_config_path))
            .expect("shared vehicle config remains valid");
    let mut runtime = FixedStepRuntime::from_state(
        config,
        state_from_fixture(&fixture.initial_state),
        fixture.dt_s,
    )
    .expect("fixture starts a valid deterministic runtime");
    let disturbance = disturbance_from_fixture(&fixture.disturbance);
    let mut max_divergence = 0.0;

    for (step, (command, expected)) in fixture
        .commands
        .iter()
        .zip(&fixture.expected_steps)
        .enumerate()
    {
        let (actual, _, _) = runtime
            .step(command_from_fixture(command), disturbance)
            .expect("fixture step remains valid");
        record_scalar(
            "sim_time_s",
            step,
            actual.sim_time_s,
            expected.sim_time_s,
            fixture.tolerances.state_abs,
            &mut max_divergence,
        );
        record_vector(
            "position_world_m",
            step,
            actual.position_world_m,
            expected.position_world_m,
            fixture.tolerances.state_abs,
            &mut max_divergence,
        );
        record_vector(
            "velocity_world_m_s",
            step,
            actual.velocity_world_m_s,
            expected.velocity_world_m_s,
            fixture.tolerances.state_abs,
            &mut max_divergence,
        );
        record_quaternion(
            step,
            actual.q_body_to_world_wxyz,
            expected.q_body_to_world_wxyz,
            fixture.tolerances.quaternion_abs,
            &mut max_divergence,
        );
        record_vector(
            "body_rate_rad_s",
            step,
            actual.body_rate_rad_s,
            expected.body_rate_rad_s,
            fixture.tolerances.state_abs,
            &mut max_divergence,
        );
        record_vector(
            "motor_thrust_n",
            step,
            actual.motor_thrust_n,
            expected.motor_thrust_n,
            fixture.tolerances.state_abs,
            &mut max_divergence,
        );
    }

    eprintln!("shared Python/Rust 6DOF max divergence: {max_divergence:e}");
}
