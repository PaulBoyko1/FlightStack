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


def test_hover_centered_throttle_preserves_physical_endpoints() -> None:
    vehicle = config()
    assert hover_centered_collective(0.0, vehicle) == 0.0
    assert hover_centered_collective(0.5, vehicle) == pytest.approx(vehicle.hover_thrust_n)
    assert hover_centered_collective(1.0, vehicle) == pytest.approx(
        4.0 * vehicle.motor_max_thrust_n
    )


def test_human_pilot_uses_shared_ctbr_and_configured_rate_limits() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle, ManualControlConfig(deadzone=0.0, expo=0.0))
    pilot.set_input(ManualInput(throttle=0.5, roll=1.0, pitch=-1.0, yaw=1.0))
    command = pilot.command(FlightState.hovering(vehicle), EmptyRace(), 0.002)
    assert command.collective_thrust_n == pytest.approx(vehicle.hover_thrust_n)
    np.testing.assert_allclose(
        command.body_rate_rad_s,
        [
            vehicle.max_body_rate_rad_s[0],
            -vehicle.max_body_rate_rad_s[1],
            vehicle.max_body_rate_rad_s[2] * 0.82,
        ],
    )


def test_keyboard_climb_command_uses_bounded_vertical_speed_control() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle)
    state = FlightState.hovering(vehicle)

    # The browser uses 0.62 while Space is held.  That value used to map into
    # the quad's enormous raw thrust reserve and produced a near-launch.  It
    # now requests a modest climb speed through the vertical-velocity loop.
    pilot.set_input(ManualInput(throttle=0.62))
    command = pilot.command(state, EmptyRace(), 0.002)

    assert command.collective_thrust_n > vehicle.hover_thrust_n
    assert command.collective_thrust_n < vehicle.hover_thrust_n * 1.5


def test_neutral_throttle_brakes_existing_vertical_velocity() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle)
    state = FlightState.hovering(vehicle)

    state.velocity_world_m_s[2] = 2.0
    pilot.set_input(ManualInput(throttle=0.5))
    climbing = pilot.command(state, EmptyRace(), 0.002)
    assert climbing.collective_thrust_n < vehicle.hover_thrust_n

    state.velocity_world_m_s[2] = -2.0
    descending = pilot.command(state, EmptyRace(), 0.002)
    assert descending.collective_thrust_n > vehicle.hover_thrust_n


def test_neutral_throttle_compensates_for_tilt() -> None:
    vehicle = config()
    pilot = HumanPilot(vehicle)
    state = FlightState.hovering(vehicle)
    state.q_body_to_world_wxyz = from_euler(0.0, np.deg2rad(35.0), 0.0)
    pilot.set_input(ManualInput(throttle=0.5))

    command = pilot.command(state, EmptyRace(), 0.002)

    assert command.collective_thrust_n > vehicle.hover_thrust_n
    assert command.collective_thrust_n < 4.0 * vehicle.motor_max_thrust_n


def test_invalid_manual_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="deadzone"):
        ManualControlConfig(deadzone=1.0)
    with pytest.raises(ValueError, match="expo"):
        ManualControlConfig(expo=1.1)
    with pytest.raises(ValueError, match="max_vertical_speed_m_s"):
        ManualControlConfig(max_vertical_speed_m_s=0.0)
    with pytest.raises(ValueError, match="vertical_speed_kp"):
        ManualControlConfig(vertical_speed_kp=0.0)
    with pytest.raises(ValueError, match="max_vertical_accel_m_s2"):
        ManualControlConfig(max_vertical_accel_m_s2=0.0)
    with pytest.raises(ValueError, match="min_up_alignment"):
        ManualControlConfig(min_up_alignment=0.0)
