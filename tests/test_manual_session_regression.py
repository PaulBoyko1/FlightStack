import numpy as np

from flightstack.web.server import (
    MANUAL_GROUND_CLEARANCE_M,
    VEHICLE_COLLISION_RADIUS_M,
    FlightSession,
)


def _set_manual(
    session: FlightSession,
    *,
    throttle: float,
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
) -> None:
    session.set_manual_input(
        {
            "throttle": throttle,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }
    )


def _step(session: FlightSession, count: int) -> None:
    for _ in range(count):
        session.step()
        assert not session.crashed


def _finish_takeoff(session: FlightSession) -> None:
    # Browser Space value starts the automatic launch.  Release immediately;
    # takeoff must continue to the configured hover altitude without requiring
    # the user to meter raw motor thrust.
    _set_manual(session, throttle=0.62)
    session.step()
    assert session.armed
    assert session.manual_takeoff_active
    _set_manual(session, throttle=0.5)
    for _ in range(4000):  # up to 8 simulated seconds
        session.step()
        assert not session.crashed
        if not session.manual_takeoff_active:
            break
    assert not session.manual_takeoff_active


def test_manual_session_starts_grounded_and_space_auto_takes_off() -> None:
    session = FlightSession.create()
    ground_z = (
        session.race.track.ground_height_m
        + VEHICLE_COLLISION_RADIUS_M
        + MANUAL_GROUND_CLEARANCE_M
    )

    assert not session.armed
    assert session.state.position_world_m[2] == ground_z
    np.testing.assert_allclose(session.state.velocity_world_m_s, 0.0, atol=1e-12)
    np.testing.assert_allclose(session.state.motor_thrust_n, 0.0, atol=1e-12)

    # WASD before takeoff must not secretly arm or move the craft.
    _set_manual(session, throttle=0.5, pitch=0.42)
    _step(session, 500)
    assert not session.armed
    assert session.state.position_world_m[2] == ground_z
    np.testing.assert_allclose(session.state.velocity_world_m_s, 0.0, atol=1e-12)

    _finish_takeoff(session)
    assert session.race.running
    assert abs(session.state.position_world_m[2] - session.manual_hover_altitude_m) < 0.08
    assert abs(session.state.velocity_world_m_s[2]) < 0.18


def test_after_takeoff_wasd_moves_and_release_brakes_while_holding_altitude() -> None:
    session = FlightSession.create()
    _finish_takeoff(session)
    hover_z = float(session.state.position_world_m[2])

    _set_manual(session, throttle=0.5, pitch=0.42)
    _step(session, 500)
    moving_speed = float(np.linalg.norm(session.state.velocity_world_m_s[:2]))
    assert moving_speed > 0.25
    assert abs(float(session.state.position_world_m[2]) - hover_z) < 0.4

    _set_manual(session, throttle=0.5)
    _step(session, 1000)
    stopped_speed = float(np.linalg.norm(session.state.velocity_world_m_s[:2]))
    assert stopped_speed < moving_speed
    assert abs(float(session.state.position_world_m[2]) - hover_z) < 0.45


def test_space_and_shift_change_altitude_only_after_takeoff_then_neutral_holds() -> None:
    session = FlightSession.create()
    _finish_takeoff(session)
    baseline_z = float(session.state.position_world_m[2])

    _set_manual(session, throttle=0.62)
    _step(session, 300)
    climbed_z = float(session.state.position_world_m[2])
    assert climbed_z > baseline_z + 0.08

    _set_manual(session, throttle=0.5)
    _step(session, 700)
    assert abs(session.state.velocity_world_m_s[2]) < 0.15

    _set_manual(session, throttle=0.38)
    _step(session, 300)
    assert session.state.position_world_m[2] < climbed_z


def test_reset_always_returns_to_idle_grounded_state() -> None:
    session = FlightSession.create()
    _finish_takeoff(session)
    _set_manual(session, throttle=0.62, pitch=0.42)
    _step(session, 200)

    session.reset()

    ground_z = (
        session.race.track.ground_height_m
        + VEHICLE_COLLISION_RADIUS_M
        + MANUAL_GROUND_CLEARANCE_M
    )
    assert not session.armed
    assert not session.manual_takeoff_active
    assert session.state.position_world_m[2] == ground_z
    np.testing.assert_allclose(session.state.velocity_world_m_s, 0.0, atol=1e-12)
    np.testing.assert_allclose(session.state.motor_thrust_n, 0.0, atol=1e-12)

    # Neutral input after reset must leave it sitting on the pad indefinitely.
    _set_manual(session, throttle=0.5)
    _step(session, 1000)
    assert not session.armed
    assert session.state.position_world_m[2] == ground_z
