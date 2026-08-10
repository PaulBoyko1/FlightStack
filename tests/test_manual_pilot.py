from dataclasses import dataclass

import numpy as np
import pytest

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


def test_invalid_manual_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="deadzone"):
        ManualControlConfig(deadzone=1.0)
    with pytest.raises(ValueError, match="expo"):
        ManualControlConfig(expo=1.1)
