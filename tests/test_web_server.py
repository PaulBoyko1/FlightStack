from __future__ import annotations

import asyncio

import numpy as np
from aiohttp.test_utils import TestClient, TestServer

from flightstack.runtime.pilots import PilotKind
from flightstack.web.server import FlightSession, create_app


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
