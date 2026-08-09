import json

import numpy as np
import pytest

from flightstack.runtime.pilots import PilotKind
from flightstack.runtime.replay import REPLAY_FORMAT_VERSION, ReplayRecorder, load_replay
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
