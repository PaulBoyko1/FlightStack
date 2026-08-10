"""A small authoritative interactive runtime and JSON WebSocket transport.

The browser is intentionally dumb: it sends human inputs and renders state
packets.  This module owns fixed-step physics, race progression, collision
semantics, pilot selection, and replay capture.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import WSMsgType, web

from flightstack.ai.errors import (
    OptionalTrainingDependencyError,
    PolicyNotTrainedError,
    PolicySchemaError,
)
from flightstack.ai.policy import LearnedPolicyPilot
from flightstack.math.quaternion import from_euler
from flightstack.race import (
    RaceEvent,
    RaceState,
    gate_frame_collision,
    ground_collision,
    load_technical_eight,
)
from flightstack.runtime.autonomy import ClassicalRacePilot
from flightstack.runtime.pilots import HumanPilot, ManualInput, PilotKind
from flightstack.runtime.replay import ReplayRecorder
from flightstack.sim.vehicle import FixedStepRuntime, FlightState, PilotCommand, VehicleConfig

PHYSICS_DT_S = 0.002
TELEMETRY_EVERY_STEPS = 15
VEHICLE_COLLISION_RADIUS_M = 0.13
MAX_CATCHUP_STEPS = 100
# A manual pilot must deliberately reach almost-hover thrust before time starts.
# That gives a newly connected browser a stable preflight scene instead of
# consuming a run while no person has had a chance to send an input.
MANUAL_ARM_COLLECTIVE_FRACTION = 0.95

def default_web_dist() -> Path:
    """Locate the Vite production bundle relative to the tracked repository."""
    return Path(__file__).resolve().parents[3] / "web" / "dist"


def _event_mapping(event: RaceEvent) -> dict[str, object]:
    result: dict[str, object] = {"type": type(event).__name__}
    for name, value in vars(event).items():
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[name] = value.item()
        else:
            result[name] = value
    return result


def _initial_state(config: VehicleConfig, start_position_world_m: np.ndarray) -> FlightState:
    """Place a preflight craft in the same actuator state used by train/eval.

    The simulation is not advanced until the session is armed, so hover motor
    state here is an internal controller initial condition rather than a claim
    that a real disarmed quad is spinning.  It prevents an unobserved zero-to-
    hover motor transient from existing only in browser deployment.
    """
    return FlightState(
        sim_time_s=0.0,
        position_world_m=start_position_world_m,
        velocity_world_m_s=np.zeros(3, dtype=np.float64),
        q_body_to_world_wxyz=from_euler(0.0, 0.0, np.pi / 2.0),
        body_rate_rad_s=np.zeros(3, dtype=np.float64),
        motor_thrust_n=np.full(4, config.hover_thrust_n / 4.0, dtype=np.float64),
    )


@dataclass
class FlightSession:
    """One deterministic local interactive session shared by connected clients."""

    config: VehicleConfig
    runtime: FixedStepRuntime
    race: RaceState
    human: HumanPilot
    classical: ClassicalRacePilot
    learned: LearnedPolicyPilot | None = None
    pilot: PilotKind = PilotKind.HUMAN
    armed: bool = False
    crashed: bool = False
    recorder: ReplayRecorder | None = None
    last_command: PilotCommand | None = None

    @classmethod
    def create(cls, *, policy_path: str | Path | None = None) -> FlightSession:
        """Construct a session and optionally validate a learned checkpoint.

        A supplied checkpoint is validated before a browser can select it.  A
        missing model is not an error for the normal manual/classical launch;
        it simply leaves the Learned control unavailable with an explicit UI
        notice.
        """
        config = VehicleConfig.from_toml()
        track = load_technical_eight()
        start = (
            np.array([0.0, -5.4, 1.2], dtype=np.float64)
            if track.start_position_world_m is None
            else np.asarray(track.start_position_world_m, dtype=np.float64)
        )
        runtime = FixedStepRuntime(config, dt=PHYSICS_DT_S, state=_initial_state(config, start))
        race = RaceState(track)
        learned: LearnedPolicyPilot | None = None
        if policy_path is not None:
            try:
                learned = LearnedPolicyPilot.from_checkpoint(policy_path, vehicle=config)
            except (
                OptionalTrainingDependencyError,
                PolicyNotTrainedError,
                PolicySchemaError,
            ) as exc:
                raise ValueError(f"could not load learned FlightStack checkpoint: {exc}") from exc
        session = cls(
            config=config,
            runtime=runtime,
            race=race,
            human=HumanPilot(config),
            classical=ClassicalRacePilot(config),
            learned=learned,
        )
        session.reset()
        return session

    @property
    def state(self) -> FlightState:
        return self.runtime.state

    @property
    def current_command(self) -> PilotCommand:
        if self.pilot is PilotKind.HUMAN:
            return self.human.command(self.state, self.race, PHYSICS_DT_S)
        if self.pilot is PilotKind.CLASSICAL:
            return self.classical.command(self.state, self.race, PHYSICS_DT_S)
        if self.learned is None:
            raise RuntimeError("learned pilot selected without a validated checkpoint")
        return self.learned.command(self.state, self.race, PHYSICS_DT_S)

    def reset(self) -> tuple[RaceEvent, ...]:
        start = self.race.track.start_position_world_m
        if start is None:
            start = np.array([0.0, 0.0, 1.2], dtype=np.float64)
        state = _initial_state(self.config, np.asarray(start, dtype=np.float64))
        self.runtime.reset(state)
        self.human.reset(state)
        self.classical.reset(state)
        if self.learned is not None:
            self.learned.reset(state)
        events = self.race.reset(0.0)
        self.armed = False
        self.crashed = False
        self.last_command = None
        self.recorder = ReplayRecorder(
            {
                "vehicle_config_hash": self.config.config_hash,
                "track": self.race.track.name,
                "physics_dt_s": PHYSICS_DT_S,
                "pilot": self.pilot.value,
            }
        )
        return events

    def set_manual_input(self, payload: dict[str, Any]) -> None:
        try:
            input_state = ManualInput(
                throttle=float(payload.get("throttle", 0.0)),
                roll=float(payload.get("roll", 0.0)),
                pitch=float(payload.get("pitch", 0.0)),
                yaw=float(payload.get("yaw", 0.0)),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "manual input must contain finite numeric throttle/roll/pitch/yaw"
            ) from exc
        self.human.set_input(input_state)

    def set_pilot(self, raw_kind: object) -> str | None:
        try:
            selected = PilotKind(str(raw_kind))
        except ValueError as exc:
            raise ValueError("pilot must be one of human, classical, or learned") from exc
        if selected is PilotKind.LEARNED:
            if self.learned is None:
                return (
                    "No validated learned checkpoint is loaded. Train one with "
                    "`python -m flightstack.ai.training --output models/run --smoke`, "
                    "then launch `flightstack serve --policy models/run/ppo_model.zip`."
                )
        self.pilot = selected
        return None

    def step(self) -> tuple[RaceEvent, ...]:
        if self.crashed or self.race.finished:
            return ()
        previous = self.state
        startup_events: tuple[RaceEvent, ...] = ()
        if not self.armed:
            if self.pilot is PilotKind.HUMAN:
                command = self.current_command
                manual_ready = (
                    command.collective_thrust_n
                    >= self.config.hover_thrust_n * MANUAL_ARM_COLLECTIVE_FRACTION
                )
                if not manual_ready:
                    self.last_command = command
                    return ()
            self.armed = True
            startup_events = self.race.start(previous.sim_time_s)
            # Autonomous pilots must see an active next gate on their first
            # inference.  In particular, do not fill a learned 20 ms hold
            # slot with an idle/preflight observation.
            command = self.current_command
        else:
            command = self.current_command
        self.last_command = command
        current, _mixed, _terms = self.runtime.step(command)
        collision_events = self._collision_events(current)
        if collision_events:
            recorded = startup_events + collision_events
        else:
            recorded = startup_events + self.race.update(
                previous.position_world_m,
                current.position_world_m,
                current.sim_time_s,
                previous_time_s=previous.sim_time_s,
            )
        if self.recorder is not None:
            self.recorder.record(
                current,
                self.pilot,
                command,
                race=self.race.to_mapping(),
                events=event_mappings(recorded),
            )
        return recorded

    def _collision_events(self, state: FlightState) -> tuple[RaceEvent, ...]:
        if ground_collision(
            state.position_world_m,
            vehicle_radius_m=VEHICLE_COLLISION_RADIUS_M,
            ground_height_m=self.race.track.ground_height_m,
        ):
            self.crashed = True
            return self.race.record_collision("ground", state.sim_time_s)
        for gate in self.race.track.gates:
            if gate_frame_collision(
                state.position_world_m,
                gate,
                vehicle_radius_m=VEHICLE_COLLISION_RADIUS_M,
            ):
                self.crashed = True
                return self.race.record_collision(f"gate-frame:{gate.gate_id}", state.sim_time_s)
        return ()

    def telemetry(self) -> dict[str, object]:
        state = self.state
        # Telemetry must never query a mutable pilot: LearnedPolicyPilot's
        # scheduler and previous-action history advance only in ``step``.
        command = (
            self.last_command
            if self.last_command is not None
            else PilotCommand.hover(self.config)
        )
        race = self.race.to_mapping()
        next_gate = race["next_gate_index"]
        lap_time = (
            0.0
            if self.race.lap_started_at_s is None
            else max(0.0, state.sim_time_s - self.race.lap_started_at_s)
        )
        status = (
            "crashed"
            if self.crashed
            else "finished"
            if self.race.finished
            else "running"
            if self.armed
            else "preflight"
        )
        return {
            "type": "state",
            "sim_time_s": state.sim_time_s,
            "pilot": self.pilot.value,
            "available_pilots": [
                PilotKind.HUMAN.value,
                PilotKind.CLASSICAL.value,
                *([PilotKind.LEARNED.value] if self.learned is not None else []),
            ],
            "state": state.to_mapping(),
            "motors": {"thrust_n": state.motor_thrust_n.tolist()},
            "pilot_command": {
                "collective_thrust_n": command.collective_thrust_n,
                "body_rate_rad_s": command.body_rate_rad_s.tolist(),
            },
            "race": {
                "lap": self.race.lap,
                "next_gate": -1 if next_gate is None else next_gate,
                "lap_time_s": lap_time,
                "best_lap_s": self.race.best_lap_s,
                "collisions": self.race.collisions,
                "status": status,
            },
            "track": [gate.to_mapping() for gate in self.race.track.gates],
        }


SESSION_KEY: web.AppKey[FlightSession] = web.AppKey("session", FlightSession)
CLIENTS_KEY: web.AppKey[set[web.WebSocketResponse]] = web.AppKey("clients", set)
PHYSICS_TASK_KEY: web.AppKey[asyncio.Task[None]] = web.AppKey("physics_task", asyncio.Task)
WEB_ROOT_KEY: web.AppKey[Path] = web.AppKey("web_root", Path)


async def _broadcast(app: web.Application) -> None:
    clients = app[CLIENTS_KEY]
    if not clients:
        return
    message = app[SESSION_KEY].telemetry()
    dead: list[web.WebSocketResponse] = []
    for client in clients:
        if client.closed:
            dead.append(client)
            continue
        try:
            await client.send_json(message)
        except ConnectionError:
            dead.append(client)
    for client in dead:
        clients.discard(client)


async def _physics_loop(app: web.Application) -> None:
    """Advance exact 2 ms steps while tracking real elapsed wall time.

    Windows commonly rounds a short ``asyncio.sleep`` to roughly 15 ms.  A
    simple one-step-per-wake loop would consequently run a 500 Hz simulation
    at about 60 Hz.  Accumulating elapsed time preserves both the canonical
    fixed integration step and human-real-time pacing across platforms.
    """
    loop = asyncio.get_running_loop()
    last_tick = loop.time()
    accumulator_s = 0.0
    steps = 0
    while True:
        now = loop.time()
        accumulator_s += min(now - last_tick, PHYSICS_DT_S * MAX_CATCHUP_STEPS)
        last_tick = now
        steps_to_run = min(int(accumulator_s / PHYSICS_DT_S), MAX_CATCHUP_STEPS)
        if steps_to_run:
            session = app[SESSION_KEY]
            for _ in range(steps_to_run):
                session.step()
            accumulator_s -= steps_to_run * PHYSICS_DT_S
            steps += steps_to_run
        if steps and steps % TELEMETRY_EVERY_STEPS < steps_to_run:
            await _broadcast(app)
        await asyncio.sleep(PHYSICS_DT_S / 2.0)


async def _on_startup(app: web.Application) -> None:
    app[PHYSICS_TASK_KEY] = asyncio.create_task(_physics_loop(app))


async def _on_cleanup(app: web.Application) -> None:
    task = app.get(PHYSICS_TASK_KEY)
    if task is not None:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def healthz(_request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", "runtime": "flightstack-python-reference"})


async def root(request: web.Request) -> web.StreamResponse:
    web_root = request.app[WEB_ROOT_KEY]
    entry = web_root / "index.html"
    if entry.is_file():
        return web.FileResponse(entry)
    return web.Response(
        text=(
            "FlightStack web client is not built. Run `cd web && pnpm install && pnpm run build` "
            "then restart `flightstack serve`."
        ),
        content_type="text/plain",
        status=503,
    )


async def websocket(request: web.Request) -> web.WebSocketResponse:
    socket = web.WebSocketResponse(heartbeat=20.0)
    await socket.prepare(request)
    clients = request.app[CLIENTS_KEY]
    session = request.app[SESSION_KEY]
    clients.add(socket)
    await socket.send_json(session.telemetry())
    try:
        async for message in socket:
            if message.type is not WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError:
                await socket.send_json({"type": "notice", "message": "Invalid JSON command."})
                continue
            if not isinstance(payload, dict):
                await socket.send_json(
                    {"type": "notice", "message": "Commands must be JSON objects."}
                )
                continue
            kind = payload.get("type")
            try:
                if kind == "manual_input":
                    session.set_manual_input(payload)
                elif kind == "set_pilot":
                    notice = session.set_pilot(payload.get("pilot"))
                    if notice is not None:
                        await socket.send_json({"type": "notice", "message": notice})
                elif kind == "reset":
                    session.reset()
                else:
                    await socket.send_json({"type": "notice", "message": "Unknown command."})
            except ValueError as exc:
                await socket.send_json({"type": "notice", "message": str(exc)})
    finally:
        clients.discard(socket)
    return socket


def create_app(
    *,
    session: FlightSession | None = None,
    web_root: Path | None = None,
    policy_path: str | Path | None = None,
) -> web.Application:
    """Build a testable aiohttp app with the browser as a pure client."""
    app = web.Application()
    app[SESSION_KEY] = (
        FlightSession.create(policy_path=policy_path) if session is None else session
    )
    app[CLIENTS_KEY] = set()
    app[WEB_ROOT_KEY] = default_web_dist() if web_root is None else web_root
    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/", root)
    asset_root = app[WEB_ROOT_KEY]
    assets = asset_root / "assets"
    if assets.is_dir():
        app.router.add_static("/assets", assets, show_index=False)
    return app


def run(
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    policy_path: str | Path | None = None,
) -> None:
    """Run the local interactive service until interrupted."""
    web.run_app(create_app(policy_path=policy_path), host=host, port=port)


def event_mappings(events: Iterable[RaceEvent]) -> tuple[dict[str, object], ...]:
    """Expose typed race events to replay/experiment callers as JSON-safe data."""
    return tuple(_event_mapping(event) for event in events)
