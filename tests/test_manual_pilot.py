from dataclasses import dataclass

import numpy as np
import pytest

from flightstack.math.quaternion import from_euler
from flightstack.runtime.pilots import (
    HumanPilot,
    ManualControlConfig,
    ManualInput,
    deadzone_expo,
    hover_centered_collective,
)
from flightstack.sim.vehicle import FlightState, VehicleConfig


@dataclass
class EmptyRace:
    next_gate_index: int = 0


def config() -> VehicleConfig:
    return VehicleConfig.from_toml()


def test_deadzone_expo_is_signed_continuous_and_bounded() -> None:
    assert deadzone_expo(0.05, deadzone=0.1, expo=0.3) == 0.0
    assert deadzone_expo(-0.05, deadzone=0.1, expo=0.3) == 0.0
    assert deadzone_expo(1.0, deadzone=0.1, expo=0.3) == pytest.approx(1.0)
    assert deadzone_expo(-1.0, deadzone=0.1, expo=0.3) == pytest.approx(-1.0)
    assert deadzone_expo(0.6, deadzone=0.1, expo=0.6) == pytest.approx(
        -deadzone_expo(-0.6, deadzone=0.1, expo=0.6)
    )


def test_hover_centered_throttle_preserves_low_level_physical_endpoints() -> None:
    vehicle = config()
    assert hover_centered_collective(0.0, vehicle) == 0.0
    assert hover_centered_collective(0.5, vehicle) == pytest.approx(vehicle.hover_thrust_n)
    assert hover_centered_collective(1.0, vehicle) == pytest.approx(
        4.0 * vehicle.motor_max_thrust_n
    )


def test_neutral_game_control_hovers_and_levels() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle, ManualControlConfig(deadzone=0.0, expo=0.0))
    state = FlightState.hovering(vehicle)
    pilot.set_input(ManualInput(throttle=0.5))

    command = pilot.command(state, EmptyRace(), 0.002)

    assert command.collective_thrust_n == pytest.approx(vehicle.hover_thrust_n)
    np.testing.assert_allclose(command.body_rate_rad_s, 0.0, atol=1e-12)


def test_w_input_requests_forward_motion_without_direct_acro_rates() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle, ManualControlConfig(deadzone=0.0, expo=0.0))
    state = FlightState.hovering(vehicle)
    pilot.set_input(ManualInput(throttle=0.5, pitch=0.42))

    command = pilot.command(state, EmptyRace(), 0.002)

    # W means forward velocity intent. The stabilizer first requests pitch while
    # holding only the vertical force required to hover. Extra collective is
    # introduced as the craft actually tilts instead of creating an altitude pop
    # while it is still upright.
    assert command.body_rate_rad_s[1] > 0.0
    assert command.body_rate_rad_s[1] < vehicle.max_body_rate_rad_s[1] * 0.8
    assert command.collective_thrust_n == pytest.approx(vehicle.hover_thrust_n)


def test_tilted_forward_motion_compensates_collective_to_hold_altitude() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle, ManualControlConfig(deadzone=0.0, expo=0.0))
    state = FlightState.hovering(vehicle)
    state.q_body_to_world_wxyz = from_euler(0.0, np.deg2rad(30.0), 0.0)
    pilot.set_input(ManualInput(throttle=0.5, pitch=0.42))

    command = pilot.command(state, EmptyRace(), 0.002)

    assert command.collective_thrust_n > vehicle.hover_thrust_n


def test_qe_yaw_remains_a_turn_command() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle, ManualControlConfig(deadzone=0.0, expo=0.0))
    state = FlightState.hovering(vehicle)
    pilot.set_input(ManualInput(throttle=0.5, yaw=0.5))

    command = pilot.command(state, EmptyRace(), 0.002)

    assert command.body_rate_rad_s[2] > 0.0
    assert command.body_rate_rad_s[2] <= vehicle.max_body_rate_rad_s[2]


def test_space_value_is_bounded_climb_speed_not_raw_racing_thrust() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle)
    state = FlightState.hovering(vehicle)
    pilot.set_input(ManualInput(throttle=0.62))

    command = pilot.command(state, EmptyRace(), 0.002)

    assert command.collective_thrust_n > vehicle.hover_thrust_n
    assert command.collective_thrust_n < vehicle.hover_thrust_n * 1.5


def test_neutral_input_brakes_vertical_and_horizontal_velocity() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle)
    state = FlightState.hovering(vehicle)
    pilot.set_input(ManualInput(throttle=0.5))

    state.velocity_world_m_s[:] = [2.0, -1.0, 2.0]
    command = pilot.command(state, EmptyRace(), 0.002)

    assert command.collective_thrust_n < vehicle.hover_thrust_n * 1.2
    assert np.linalg.norm(command.body_rate_rad_s[:2]) > 0.0


def test_takeoff_command_climbs_then_settles_at_target() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle)
    state = FlightState.hovering(vehicle, altitude_m=0.14)

    low = pilot.takeoff_command(state, 1.2)
    assert low.collective_thrust_n > vehicle.hover_thrust_n

    state.position_world_m[2] = 1.2
    state.velocity_world_m_s[2] = 0.0
    settled = pilot.takeoff_command(state, 1.2)
    assert settled.collective_thrust_n == pytest.approx(vehicle.hover_thrust_n)
    np.testing.assert_allclose(settled.body_rate_rad_s, 0.0, atol=1e-12)


def test_invalid_manual_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="deadzone"):
        ManualControlConfig(deadzone=1.0)
    with pytest.raises(ValueError, match="expo"):
        ManualControlConfig(expo=1.1)
    with pytest.raises(ValueError, match="max_vertical_speed_m_s"):
        ManualControlConfig(max_vertical_speed_m_s=0.0)
    with pytest.raises(ValueError, match="horizontal_speed_kp"):
        ManualControlConfig(horizontal_speed_kp=0.0)
    with pytest.raises(ValueError, match="attitude_kp"):
        ManualControlConfig(attitude_kp=0.0)
