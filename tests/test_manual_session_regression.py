import numpy as np

from flightstack.web.server import FlightSession


def _set_manual(session: FlightSession, *, throttle: float, pitch: float = 0.0) -> None:
    session.set_manual_input(
        {
            "throttle": throttle,
            "roll": 0.0,
            "pitch": pitch,
            "yaw": 0.0,
        }
    )


def _step(session: FlightSession, count: int) -> None:
    for _ in range(count):
        session.step()
        assert not session.crashed


def test_fresh_manual_flight_and_reset_do_not_latch_vertical_motion() -> None:
    session = FlightSession.create()
    start = session.state.position_world_m.copy()

    # The browser's neutral input must arm and hover on the first run without
    # requiring a reset to make keyboard control become active.
    _set_manual(session, throttle=0.5)
    _step(session, 500)  # 1 second at the authoritative 500 Hz step.
    assert session.armed
    assert abs(session.state.position_world_m[2] - start[2]) < 0.02
    assert abs(session.state.velocity_world_m_s[2]) < 0.02

    # Reproduce the current browser Space command, then release to neutral.
    # The old raw-thrust mapping launched the quad and hover thrust could not
    # remove the accumulated upward momentum.  Neutral must now brake it.
    _set_manual(session, throttle=0.62)
    _step(session, 250)  # hold climb for 0.5 s
    assert session.state.velocity_world_m_s[2] > 0.0

    _set_manual(session, throttle=0.5)
    _step(session, 750)  # release for 1.5 s
    assert abs(session.state.velocity_world_m_s[2]) < 0.08
    assert session.state.position_world_m[2] < start[2] + 1.5

    # Reset must restore position, velocity, controller state, and a subsequent
    # centered browser input must remain a stable hover rather than relaunch.
    session.reset()
    np.testing.assert_allclose(session.state.position_world_m, start, atol=1e-12)
    np.testing.assert_allclose(session.state.velocity_world_m_s, 0.0, atol=1e-12)
    _set_manual(session, throttle=0.5)
    _step(session, 500)
    assert abs(session.state.position_world_m[2] - start[2]) < 0.02
    assert abs(session.state.velocity_world_m_s[2]) < 0.02


def test_wasd_pitch_moves_on_first_run_without_vertical_runaway() -> None:
    session = FlightSession.create()
    start_z = float(session.state.position_world_m[2])

    _set_manual(session, throttle=0.5, pitch=0.42)
    _step(session, 200)

    horizontal_speed = float(np.linalg.norm(session.state.velocity_world_m_s[:2]))
    assert session.armed
    assert horizontal_speed > 0.05
    assert abs(float(session.state.position_world_m[2]) - start_z) < 0.35
