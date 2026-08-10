import json

import numpy as np
import pytest

from flightstack.math.quaternion import from_euler, rotate
from flightstack.runtime.pilots import PilotKind
from flightstack.runtime.replay import (
    REPLAY_FORMAT_VERSION,
    ReplayFormatError,
    ReplayPlayer,
    ReplayRecorder,
    load_replay,
    read_replay,
)
from flightstack.sim.vehicle import FlightState, PilotCommand, VehicleConfig


def test_replay_records_authoritative_state_and_serializes(tmp_path) -> None:
    config = VehicleConfig.from_toml()
    recorder = ReplayRecorder({"seed": 42, "vehicle_config_hash": config.config_hash})
    recorder.record(
        FlightState.hovering(config),
        PilotKind.HUMAN,
        PilotCommand.hover(config),
        race={"next_gate": 0},
        events=({"type": "reset"},),
    )
    output = recorder.write(tmp_path / "run.json")
    loaded = load_replay(output)
    assert loaded["format"] == REPLAY_FORMAT_VERSION
    assert loaded["metadata"]["seed"] == 42
    assert loaded["frames"][0]["state"]["motor_thrust_n"] == pytest.approx(
        np.full(4, config.hover_thrust_n / 4.0).tolist()
    )
    serialized = json.loads(output.read_text(encoding="utf-8"))
    assert serialized["frames"][0]["events"] == [{"type": "reset"}]


def test_replay_rejects_time_travel() -> None:
    config = VehicleConfig.from_toml()
    recorder = ReplayRecorder({})
    recorder.record(FlightState.hovering(config), PilotKind.HUMAN, PilotCommand.hover(config))
    earlier = FlightState.hovering(config)
    earlier.sim_time_s = -1.0
    with pytest.raises(ValueError):
        recorder.record(earlier, PilotKind.HUMAN, PilotCommand.hover(config))


def _two_frame_replay(tmp_path):
    config = VehicleConfig.from_toml()
    recorder = ReplayRecorder({"seed": 17, "run": "typed-playback"})
    first = FlightState.hovering(config)
    first.sim_time_s = 0.0
    first.position_world_m = np.array([0.0, 0.0, 1.0])
    recorder.record(
        first,
        PilotKind.HUMAN,
        PilotCommand.hover(config),
        race={"lap": 1, "next_gate": 0},
        events=({"type": "Start", "time_s": 0.0},),
    )
    second = FlightState(
        sim_time_s=1.0,
        position_world_m=np.array([2.0, 4.0, 3.0]),
        velocity_world_m_s=np.array([2.0, 4.0, 2.0]),
        q_body_to_world_wxyz=from_euler(0.0, 0.0, np.pi),
        body_rate_rad_s=np.array([0.2, -0.4, 0.6]),
        motor_thrust_n=np.full(4, config.hover_thrust_n / 3.0),
    )
    recorder.record(
        second,
        PilotKind.CLASSICAL,
        PilotCommand(config.hover_thrust_n * 1.1, np.array([0.3, -0.2, 0.1])),
        race={"lap": 1, "next_gate": 1},
        events=({"type": "GatePassed", "gate_index": 0, "time_s": 1.0},),
    )
    return recorder.write(tmp_path / "typed-replay.json")


def test_typed_reader_reconstructs_canonical_v1_frames_and_events(tmp_path) -> None:
    replay_path = _two_frame_replay(tmp_path)

    replay = read_replay(replay_path)

    assert replay.metadata == {"run": "typed-playback", "seed": 17}
    assert replay.start_time_s == pytest.approx(0.0)
    assert replay.end_time_s == pytest.approx(1.0)
    assert replay.duration_s == pytest.approx(1.0)
    assert replay.frames[0].pilot is PilotKind.HUMAN
    assert isinstance(replay.frames[0].state, FlightState)
    assert isinstance(replay.frames[0].command, PilotCommand)
    assert replay.frames[1].events[0].kind == "GatePassed"
    assert replay.frames[1].events[0].data == {"gate_index": 0, "time_s": 1.0}
    assert replay.summary()["events"] == {"GatePassed": 1, "Start": 1}
    assert replay.to_mapping() == load_replay(replay_path)

    # V1 has always carried generic event objects, so a future typed event is
    # additive rather than a requirement for opening older recordings.
    config = VehicleConfig.from_toml()
    legacy = ReplayRecorder({"format_note": "generic-events"})
    legacy.record(
        FlightState.hovering(config),
        PilotKind.HUMAN,
        PilotCommand.hover(config),
        events=({"note": "no explicit event type"},),
    )
    legacy_path = legacy.write(tmp_path / "legacy-generic-event.json")
    legacy_document = read_replay(legacy_path)
    assert legacy_document.frames[0].events[0].kind is None
    assert legacy_document.to_mapping()["frames"][0]["events"] == [
        {"note": "no explicit event type"}
    ]


def test_player_holds_source_frames_and_interpolates_only_continuous_state(tmp_path) -> None:
    player = ReplayPlayer(read_replay(_two_frame_replay(tmp_path)))

    held = player.frame_at(0.5)
    interpolated = player.frame_at(0.5, interpolate=True)

    np.testing.assert_allclose(held.state.position_world_m, [0.0, 0.0, 1.0])
    assert held.events[0].kind == "Start"
    np.testing.assert_allclose(interpolated.state.position_world_m, [1.0, 2.0, 2.0])
    np.testing.assert_allclose(interpolated.state.velocity_world_m_s, [1.0, 2.0, 1.0])
    np.testing.assert_allclose(
        rotate(interpolated.state.q_body_to_world_wxyz, [1.0, 0.0, 0.0]),
        [0.0, 1.0, 0.0],
        atol=1e-10,
    )
    assert interpolated.pilot is PilotKind.HUMAN
    assert interpolated.race == {"lap": 1, "next_gate": 0}
    assert interpolated.events == ()
    assert player.frame_at(-5.0).state.sim_time_s == pytest.approx(0.0)
    assert player.frame_at(5.0).state.sim_time_s == pytest.approx(1.0)
    assert [frame.state.sim_time_s for frame in player.sampled_frames(0.25)] == pytest.approx(
        [0.0, 0.25, 0.5, 0.75, 1.0]
    )


def test_typed_reader_rejects_malformed_v1_frames(tmp_path) -> None:
    replay_path = _two_frame_replay(tmp_path)
    raw = json.loads(replay_path.read_text(encoding="utf-8"))
    raw["frames"][1]["pilot"] = "unrecognized"
    invalid_pilot = tmp_path / "invalid-pilot.json"
    invalid_pilot.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="pilot must be one of"):
        read_replay(invalid_pilot)

    raw = json.loads(replay_path.read_text(encoding="utf-8"))
    raw["frames"][1]["state"]["sim_time_s"] = -1.0
    invalid_state = tmp_path / "invalid-state.json"
    invalid_state.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="state is invalid"):
        read_replay(invalid_state)

    raw = json.loads(replay_path.read_text(encoding="utf-8"))
    raw["frames"][1]["state"]["sim_time_s"] = 0.0
    raw["frames"][0]["state"]["sim_time_s"] = 1.0
    out_of_order = tmp_path / "out-of-order.json"
    out_of_order.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReplayFormatError, match="nondecreasing"):
        read_replay(out_of_order)


def test_player_exports_source_or_interpolated_frames_as_csv(tmp_path) -> None:
    player = ReplayPlayer(read_replay(_two_frame_replay(tmp_path)))

    source_csv = player.export_csv(tmp_path / "source.csv")
    sampled_csv = player.export_csv(tmp_path / "sampled.csv", sample_period_s=0.5)

    source_lines = source_csv.read_text(encoding="utf-8").splitlines()
    sampled_lines = sampled_csv.read_text(encoding="utf-8").splitlines()
    assert source_lines[0].startswith("sim_time_s,pilot,position_x_m")
    assert len(source_lines) == 3
    assert len(sampled_lines) == 4
    assert '"GatePassed"' in source_lines[-1]
