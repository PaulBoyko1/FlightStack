import numpy as np
import pytest

from flightstack.math.quaternion import from_euler, rotate, wxyz_to_xyzw, xyzw_to_wxyz
from flightstack.sim.vehicle import (
    Disturbance,
    FixedStepRuntime,
    FlightState,
    Multirotor,
    PilotCommand,
    QuadMixer,
    VehicleConfig,
)


def reference_config() -> VehicleConfig:
    return VehicleConfig.from_toml()


def test_vehicle_config_is_valid_and_hash_is_stable() -> None:
    config = reference_config()
    assert config.name == "flightstack_5in"
    assert config.hover_thrust_n == pytest.approx(config.mass_kg * config.gravity_m_s2)
    assert config.config_hash == VehicleConfig.from_toml().config_hash
    assert len(config.config_hash) == 16


def test_vehicle_config_rejects_non_spd_inertia() -> None:
    config = reference_config()
    with pytest.raises(ValueError, match="positive definite"):
        VehicleConfig(
            **{
                **config.__dict__,
                "inertia_kg_m2": np.diag([0.002, 0.0, 0.004]),
            }
        )


@pytest.mark.parametrize(
    "angles_deg",
    ([0.0, 0.0, 0.0], [90.0, 0.0, 0.0], [0.0, -90.0, 0.0], [20.0, -31.0, 47.0]),
)
def test_wxyz_xyzw_boundary_preserves_rotation(angles_deg: list[float]) -> None:
    q = from_euler(*np.deg2rad(angles_deg))
    round_trip = xyzw_to_wxyz(wxyz_to_xyzw(q))
    np.testing.assert_allclose(round_trip, q)
    np.testing.assert_allclose(rotate(round_trip, [0.3, -0.2, 0.7]), rotate(q, [0.3, -0.2, 0.7]))
    np.testing.assert_allclose(xyzw_to_wxyz(wxyz_to_xyzw(-q)), -q)


def test_motor_response_uses_exact_rise_and_fall_update() -> None:
    config = reference_config()
    state = FlightState.hovering(config)
    state.motor_thrust_n = np.zeros(4)
    vehicle = Multirotor(config, state)
    target = np.full(4, 3.0)
    dt = 0.01
    vehicle.step_motor_targets(target, dt)
    expected_rise = 3.0 * (1.0 - np.exp(-dt / config.motor_tau_rise_s))
    np.testing.assert_allclose(vehicle.state.motor_thrust_n, expected_rise)
    vehicle.step_motor_targets(np.zeros(4), dt)
    expected_fall = expected_rise * np.exp(-dt / config.motor_tau_fall_s)
    np.testing.assert_allclose(vehicle.state.motor_thrust_n, expected_fall)


def test_zero_thrust_free_fall_has_expected_gravity_acceleration() -> None:
    config = reference_config()
    state = FlightState.hovering(config, altitude_m=10.0)
    state.motor_thrust_n = np.zeros(4)
    vehicle = Multirotor(config, state)
    vehicle.step_motor_targets(np.zeros(4), 0.1)
    assert vehicle.state.velocity_world_m_s[2] == pytest.approx(-config.gravity_m_s2 * 0.1)
    assert vehicle.state.position_world_m[2] == pytest.approx(10.0 - config.gravity_m_s2 * 0.01)


def test_level_hover_has_negligible_acceleration_at_equilibrium() -> None:
    config = reference_config()
    vehicle = Multirotor(config, FlightState.hovering(config))
    before = vehicle.state.copy()
    vehicle.step_motor_targets(np.full(4, config.hover_thrust_n / 4.0), 0.002)
    np.testing.assert_allclose(
        vehicle.state.velocity_world_m_s, before.velocity_world_m_s, atol=1e-12
    )
    np.testing.assert_allclose(vehicle.state.body_rate_rad_s, 0.0, atol=1e-12)


def test_symmetric_collective_has_no_body_moment() -> None:
    config = reference_config()
    vehicle = Multirotor(config, FlightState.hovering(config))
    force, moment = vehicle.motor_force_and_moment()
    assert force[2] == pytest.approx(config.hover_thrust_n)
    np.testing.assert_allclose(moment, 0.0, atol=1e-12)


def test_mixer_signs_and_saturation_are_explicit() -> None:
    config = reference_config()
    mixer = QuadMixer(config)
    equal = mixer.mix(config.hover_thrust_n, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(equal.motor_target_thrust_n, config.hover_thrust_n / 4.0)
    np.testing.assert_allclose(equal.achieved_torque_body_nm, 0.0, atol=1e-12)

    roll = mixer.mix(config.hover_thrust_n, [0.03, 0.0, 0.0])
    assert roll.achieved_torque_body_nm[0] > 0.0
    assert roll.motor_target_thrust_n[0] > roll.motor_target_thrust_n[2]

    pitch = mixer.mix(config.hover_thrust_n, [0.0, 0.03, 0.0])
    assert pitch.achieved_torque_body_nm[1] > 0.0
    assert pitch.motor_target_thrust_n[1] > pitch.motor_target_thrust_n[0]

    yaw = mixer.mix(config.hover_thrust_n, [0.0, 0.0, 0.03])
    assert yaw.achieved_torque_body_nm[2] > 0.0
    assert yaw.motor_target_thrust_n[0] > yaw.motor_target_thrust_n[1]

    limited = mixer.mix(config.motor_max_thrust_n * 4.0, [1.0, 1.0, 1.0])
    assert limited.saturated
    assert np.all(limited.motor_target_thrust_n <= config.motor_max_thrust_n)


def test_motor_efficiency_degradation_is_a_reproducible_failure_hook() -> None:
    config = reference_config()
    state = FlightState.hovering(config)
    state.motor_thrust_n = np.zeros(4)
    vehicle = Multirotor(config, state)
    degraded = Disturbance(
        np.zeros(3), np.zeros(3), np.zeros(3), np.array([0.5, 1.0, 1.0, 1.0])
    )
    vehicle.step_motor_targets(np.full(4, 4.0), 1.0, degraded)
    assert vehicle.state.motor_thrust_n[0] == pytest.approx(2.0, rel=1e-6)
    assert vehicle.state.motor_thrust_n[1] == pytest.approx(4.0, rel=1e-6)


def test_fixed_step_runtime_is_deterministic_and_uses_ctbr_chain() -> None:
    config = reference_config()
    command = PilotCommand(config.hover_thrust_n, np.array([0.4, -0.25, 0.1]))

    def run() -> np.ndarray:
        runtime = FixedStepRuntime(config)
        for _ in range(250):
            runtime.step(command)
        state = runtime.state
        return np.concatenate(
            (
                state.position_world_m,
                state.velocity_world_m_s,
                state.q_body_to_world_wxyz,
                state.body_rate_rad_s,
                state.motor_thrust_n,
            )
        )

    np.testing.assert_array_equal(run(), run())


def test_rate_controller_damps_a_body_rate_step_through_mixer_and_motors() -> None:
    config = reference_config()
    state = FlightState.hovering(config)
    state.body_rate_rad_s = np.array([1.0, -0.8, 0.5])
    runtime = FixedStepRuntime(config, state=state)
    command = PilotCommand.hover(config)
    initial_norm = float(np.linalg.norm(runtime.state.body_rate_rad_s))
    for _ in range(1_500):
        runtime.step(command)
    assert float(np.linalg.norm(runtime.state.body_rate_rad_s)) < initial_norm * 0.2


def test_external_force_moves_vehicle_in_world_frame() -> None:
    config = reference_config()
    vehicle = Multirotor(config, FlightState.hovering(config))
    force = Disturbance(np.array([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3), np.ones(4))
    vehicle.step_motor_targets(np.full(4, config.hover_thrust_n / 4.0), 0.1, force)
    assert vehicle.state.velocity_world_m_s[0] == pytest.approx(0.1 / config.mass_kg)
