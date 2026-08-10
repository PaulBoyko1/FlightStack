from __future__ import annotations

import asyncio

import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from flightstack.ai.policy import LearnedPolicyPilot
from flightstack.race import Gate, RaceState, Track
from flightstack.runtime.pilots import PilotKind
from flightstack.web.server import FlightSession, create_app


class FixedHoverPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, None]:
        del observation, deterministic
        self.calls += 1
        return np.zeros(4, dtype=np.float32), None


def test_server_health_websocket_and_command_path(tmp_path) -> None:
    async def exercise() -> None:
        async def wait_for(predicate) -> None:
            for _ in range(25):
                if predicate():
                    return
                await asyncio.sleep(0.01)
            assert predicate()

        web_root = tmp_path / "web"
        web_root.mkdir()
        (web_root / "index.html").write_text("<title>FlightStack</title>", encoding="utf-8")
        session = FlightSession.create()
        app = create_app(session=session, web_root=web_root)
        server = TestServer(app)
        client = TestClient(server)
        await client.start_server()
        try:
            health = await client.get("/healthz")
            assert health.status == 200
            assert (await health.json())["status"] == "ok"
            home = await client.get("/")
            assert home.status == 200

            socket = await client.ws_connect("/ws")
            first = await socket.receive_json()
            assert first["type"] == "state"
            np.testing.assert_allclose(
                first["state"]["q_body_to_world_wxyz"],
                [0.7071067811865476, 0.0, 0.0, 0.7071067811865475],
            )
            assert first["race"]["status"] == "preflight"
            await socket.send_json(
                {"type": "manual_input", "throttle": 0.5, "roll": 0.2, "pitch": 0.0, "yaw": 0.0}
            )
            await wait_for(lambda: session.human.input.throttle == 0.5)
            assert session.human.input.throttle == 0.5
            await socket.send_json({"type": "set_pilot", "pilot": "classical"})
            await wait_for(lambda: session.pilot is PilotKind.CLASSICAL)
            assert session.pilot is PilotKind.CLASSICAL
            before = session.state.sim_time_s
            session.step()
            assert session.state.sim_time_s > before
            assert session.race.running
            # The runtime must catch up in fixed 2 ms steps even on systems
            # whose asyncio timer wakes only every ~15 ms.
            paced_before = session.state.sim_time_s
            await asyncio.sleep(0.10)
            assert session.state.sim_time_s - paced_before >= 0.05
            await socket.close()
        finally:
            await client.close()

    asyncio.run(exercise())


def test_learned_mode_requires_a_checkpoint_then_uses_the_ctbr_adapter() -> None:
    session = FlightSession.create()

    notice = session.set_pilot("learned")

    assert notice is not None
    assert session.pilot is PilotKind.HUMAN
    session.learned = LearnedPolicyPilot(session.config, FixedHoverPolicy())
    assert session.set_pilot("learned") is None
    assert session.pilot is PilotKind.LEARNED
    command = session.current_command
    assert command.collective_thrust_n == pytest.approx(session.config.hover_thrust_n)
    np.testing.assert_allclose(command.body_rate_rad_s, 0.0)


def test_telemetry_does_not_advance_the_learned_policy_scheduler() -> None:
    session = FlightSession.create()
    policy = FixedHoverPolicy()
    session.learned = LearnedPolicyPilot(session.config, policy)
    assert session.set_pilot("learned") is None
    assert session.recorder is not None
    assert [event["type"] for event in session.recorder.frames[0].to_mapping()["events"]] == [
        "Reset"
    ]

    # Broadcasting before physics advances must be observational.  A browser
    # connect/disconnect cadence cannot change autonomous behavior.
    session.telemetry()
    session.telemetry()
    assert policy.calls == 0
    np.testing.assert_allclose(
        session.state.motor_thrust_n,
        session.config.hover_thrust_n / 4.0,
    )

    session.step()
    assert session.race.running
    assert policy.calls == 1
    assert [event["type"] for event in session.recorder.frames[-1].to_mapping()["events"]] == [
        "Start"
    ]
    session.telemetry()
    session.telemetry()
    assert policy.calls == 1


def test_session_records_collision_before_a_same_tick_finish(monkeypatch) -> None:
    session = FlightSession.create()
    gate = Gate(
        center_world_m=[0.0, 0.0, 1.0],
        normal_world=[0.0, 1.0, 0.0],
        right_world=[-1.0, 0.0, 0.0],
        up_world=[0.0, 0.0, 1.0],
        half_width_m=2.0,
        half_height_m=2.0,
        gate_id="finish",
        frame_thickness_m=0.0,
        frame_depth_m=0.0,
    )
    session.race = RaceState(Track(name="collision-wins", gates=(gate,), gate_order=(1,)))
    session.race.reset(0.0)
    session.race.start(0.0)
    session.pilot = PilotKind.CLASSICAL
    session.armed = True
    previous = session.state.copy()
    previous.position_world_m[:] = [0.0, -1.0, 1.0]
    session.runtime.reset(previous)
    current = previous.copy()
    current.sim_time_s = previous.sim_time_s + 0.002
    current.position_world_m[:] = [0.0, 1.0, 1.0]
    monkeypatch.setattr(session.runtime, "step", lambda command: (current, None, None))

    def synthetic_collision(_state):
        session.crashed = True
        return session.race.record_collision("synthetic-frame", current.sim_time_s)

    monkeypatch.setattr(session, "_collision_events", synthetic_collision)
    events = session.step()

    assert not session.race.finished
    assert session.race.collisions == 1
    assert [type(event).__name__ for event in events] == ["Collision"]
