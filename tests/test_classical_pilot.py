import numpy as np
import pytest

from flightstack.math.quaternion import (
    from_euler,
    from_rotation_matrix,
    geodesic_angle,
    rotate,
    to_rotation_matrix,
)
from flightstack.race import Gate, RaceState, Track
from flightstack.runtime.autonomy import ClassicalPilotConfig, ClassicalRacePilot
from flightstack.runtime.pilots import PilotKind
from flightstack.sim.vehicle import FlightState, VehicleConfig
from flightstack.web.server import FlightSession


def vehicle() -> VehicleConfig:
    return VehicleConfig.from_toml()


def race() -> RaceState:
    gate = Gate(
        center_world_m=[0.0, 5.0, 2.0],
        normal_world=[0.0, 1.0, 0.0],
        right_world=[-1.0, 0.0, 0.0],
        up_world=[0.0, 0.0, 1.0],
        half_width_m=1.0,
        half_height_m=1.0,
        gate_id="first",
    )
    result = RaceState(Track(name="one", gates=(gate,), gate_order=(1,)))
    result.start(0.0)
    return result


def test_rotation_matrix_round_trip_preserves_arbitrary_attitude() -> None:
    attitude = from_euler(*np.deg2rad([28.0, -37.0, 121.0]))
    reconstructed = from_rotation_matrix(to_rotation_matrix(attitude))
    assert geodesic_angle(attitude, reconstructed) == pytest.approx(0.0, abs=1e-12)


def test_rotation_matrix_rejects_non_rotation() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        from_rotation_matrix(np.diag([2.0, 1.0, 1.0]))


def test_classical_pilot_targets_next_gate_with_shared_ctbr_limits() -> None:
    config = vehicle()
    pilot = ClassicalRacePilot(config)
    state = FlightState.hovering(config)
    result = pilot.command(state, race(), 0.002)
    assert result.collective_thrust_n > 0.0
    assert result.collective_thrust_n <= 4.0 * config.motor_max_thrust_n
    assert np.all(np.abs(result.body_rate_rad_s) <= config.max_body_rate_rad_s)


def test_classical_thrust_attitude_aligns_body_z_to_requested_force() -> None:
    config = vehicle()
    pilot = ClassicalRacePilot(config)
    state = FlightState.hovering(config)
    state.position_world_m = np.array([-2.0, -1.0, 1.0])
    command = pilot.command(state, race(), 0.002)
    assert command.collective_thrust_n > config.hover_thrust_n
    # The generated rate command is a body-frame corrective command; it must
    # be nonzero for this off-track state rather than bypassing the rate loop.
    assert np.linalg.norm(command.body_rate_rad_s) > 0.0


def test_classical_config_rejects_unphysical_tilt() -> None:
    with pytest.raises(ValueError, match="max_tilt"):
        ClassicalPilotConfig(max_tilt_rad=np.pi / 2.0)


def test_body_z_rotation_contract_is_available_to_guidance() -> None:
    q = from_euler(0.0, 0.0, np.pi / 2.0)
    np.testing.assert_allclose(rotate(q, [1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


def test_classical_baseline_finishes_reference_technical_eight() -> None:
    session = FlightSession.create()
    session.pilot = PilotKind.CLASSICAL
    # The expanded course is 1.5x wider in X/Y than the original reference
    # layout. Keep the deterministic completion requirement, but give the
    # controller enough simulated time to traverse the longer geometry.
    for _ in range(20_000):
        session.step()
        if session.crashed or session.race.finished:
            break
    assert not session.crashed
    assert session.race.finished
    assert session.race.best_lap_s is not None
    assert session.race.best_lap_s < 35.0
