from __future__ import annotations

import json

import numpy as np

from flightstack.ai import training
from flightstack.ai.errors import OptionalTrainingDependencyError
from flightstack.cli import build_parser
from flightstack.runtime.pilots import PilotKind
from flightstack.runtime.replay import ReplayRecorder
from flightstack.sim.vehicle import FlightState, PilotCommand, VehicleConfig


def test_evaluate_command_writes_structured_artifacts_for_a_short_track(tmp_path, capsys) -> None:
    track = tmp_path / "straight.json"
    track.write_text(
        json.dumps(
            {
                "name": "cli-straight",
                "start_position_world_m": [0.0, -1.0, 1.5],
                "gates": [
                    {
                        "id": "finish",
                        "center_world_m": [0.0, 0.0, 1.5],
                        "normal_world": [0.0, 1.0, 0.0],
                        "right_world": [-1.0, 0.0, 0.0],
                        "up_world": [0.0, 0.0, 1.0],
                        "width_m": 3.0,
                        "height_m": 3.0,
                        "frame_thickness_m": 0.0,
                        "frame_depth_m": 0.0,
                    }
                ],
                "gate_order": [1],
            }
        ),
        encoding="utf-8",
    )
    scenario = tmp_path / "scenario.toml"
    scenario.write_text(
        "\n".join(
            (
                'name = "cli-straight"',
                "seed = 4",
                f"track = {json.dumps(str(track))}",
                "duration_s = 1.5",
                "physics_dt_s = 0.01",
                "telemetry_period_s = 0.05",
            )
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"
    args = build_parser().parse_args(
        ["evaluate", "--scenario", str(scenario), "--output", str(output)]
    )

    assert args.func(args) == 0

    rendered = json.loads(capsys.readouterr().out)
    assert rendered["metrics"]["completed"]
    assert (output / "result.json").is_file()
    assert (output / "telemetry.json").is_file()
    assert (output / "replay.json").is_file()


def test_train_and_learned_evaluation_arguments_remain_explicit(tmp_path) -> None:
    parser = build_parser()
    train = parser.parse_args(["train", "--output", str(tmp_path / "model"), "--smoke"])
    learned = parser.parse_args(["evaluate", "--pilot", "learned"])

    assert train.smoke
    assert learned.policy is None


def test_train_command_explains_the_optional_extra(monkeypatch, tmp_path, capsys) -> None:
    def unavailable(*_args, **_kwargs):
        raise OptionalTrainingDependencyError(
            "install with `python -m pip install -e \".[train]\"`"
        )

    monkeypatch.setattr(training, "train_ppo", unavailable)
    args = build_parser().parse_args(["train", "--output", str(tmp_path / "model"), "--smoke"])

    assert args.func(args) == 2
    assert ".[train]" in capsys.readouterr().err


def test_replay_command_inspects_and_exports_interpolated_state_frames(tmp_path, capsys) -> None:
    config = VehicleConfig.from_toml()
    recorder = ReplayRecorder({"seed": 4})
    start = FlightState.hovering(config)
    start.position_world_m = np.array([0.0, 0.0, 1.0])
    recorder.record(
        start,
        PilotKind.HUMAN,
        PilotCommand.hover(config),
        events=({"type": "Start", "time_s": 0.0},),
    )
    finish = FlightState.hovering(config)
    finish.sim_time_s = 1.0
    finish.position_world_m = np.array([2.0, 0.0, 1.0])
    recorder.record(
        finish,
        PilotKind.CLASSICAL,
        PilotCommand.hover(config),
        events=({"type": "GatePassed", "time_s": 1.0, "gate_index": 0},),
    )
    replay = recorder.write(tmp_path / "run.json")
    csv_output = tmp_path / "replay.csv"
    args = build_parser().parse_args(
        [
            "replay",
            str(replay),
            "--at",
            "0.5",
            "--interpolate",
            "--csv",
            str(csv_output),
            "--sample-period",
            "0.25",
        ]
    )

    assert args.func(args) == 0

    rendered = json.loads(capsys.readouterr().out)
    assert rendered["frame_count"] == 2
    assert rendered["events"] == {"GatePassed": 1, "Start": 1}
    assert rendered["interpolated"]
    assert rendered["frame"]["state"]["position_world_m"] == [1.0, 0.0, 1.0]
    assert rendered["csv"] == str(csv_output)
    assert len(csv_output.read_text(encoding="utf-8").splitlines()) == 6
