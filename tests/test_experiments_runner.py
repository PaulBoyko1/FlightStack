from __future__ import annotations

import json

from flightstack.experiments import Scenario, run_episode
from flightstack.runtime.autonomy import ClassicalRacePilot
from flightstack.runtime.replay import REPLAY_FORMAT_VERSION, load_replay


def write_straight_track(tmp_path, *, start_height_m: float = 1.5):
    source = tmp_path / "straight-track.json"
    source.write_text(
        json.dumps(
            {
                "name": "straight-smoke",
                "default_laps": 1,
                "ground_height_m": 0.0,
                "start_position_world_m": [0.0, -1.0, start_height_m],
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
    return source


def test_headless_episode_records_canonical_race_telemetry_and_replay(tmp_path) -> None:
    track = write_straight_track(tmp_path)
    scenario = Scenario(
        name="straight-wind-degraded",
        seed=12,
        track=str(track),
        duration_s=1.5,
        physics_dt_s=0.01,
        telemetry_period_s=0.05,
        wind_world_m_s=[0.03, -0.02, 0.0],
        motor_efficiency=[0.99, 0.98, 0.99, 0.97],
    )

    result = run_episode(scenario, ClassicalRacePilot, pilot_name="classical")

    assert result.metrics.completed
    assert result.metrics.termination == "finished"
    assert result.metrics.gates_passed == 1
    assert result.metrics.collisions == 0
    assert result.metrics.distance_travelled_m > 0.9
    assert result.provenance.pilot_name == "classical"
    assert result.provenance.scenario.seed == 12
    assert result.telemetry
    assert result.replay["format"] == REPLAY_FORMAT_VERSION
    assert {event["type"] for event in result.events} >= {
        "Reset",
        "Start",
        "GatePassed",
        "LapCompleted",
        "RaceFinished",
    }

    artifacts = result.write_artifacts(tmp_path / "artifacts")
    assert artifacts.summary_path.is_file()
    assert artifacts.telemetry_path.is_file()
    replay = load_replay(artifacts.replay_path)
    assert replay["metadata"]["provenance"]["pilot_name"] == "classical"
    assert len(replay["frames"]) == len(result.telemetry)


def test_headless_episode_uses_ground_collision_as_a_terminal_event(tmp_path) -> None:
    track = write_straight_track(tmp_path, start_height_m=0.05)
    scenario = Scenario(
        name="ground-contact",
        seed=3,
        track=str(track),
        duration_s=0.02,
        physics_dt_s=0.01,
        telemetry_period_s=0.01,
        vehicle_radius_m=0.13,
    )

    result = run_episode(scenario, ClassicalRacePilot)

    assert result.metrics.termination == "crashed"
    assert not result.metrics.completed
    assert result.metrics.collisions == 1
    assert any(event["type"] == "Collision" for event in result.events)
