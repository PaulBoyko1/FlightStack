from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np

import flightstack.experiments.runner as runner_module
from flightstack.experiments import Scenario, checkpoint_model_identity, run_episode
from flightstack.runtime.autonomy import ClassicalRacePilot
from flightstack.runtime.replay import REPLAY_FORMAT_VERSION, load_replay
from flightstack.sim.vehicle import FlightState


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
    # Replays retain an initial reset/start frame even though telemetry starts
    # after the first physics step, and they never downsample events away.
    assert len(replay["frames"]) == len(result.telemetry) + 1
    replay_event_types = {
        event["type"] for frame in replay["frames"] for event in frame["events"]
    }
    assert replay_event_types >= {
        "Reset",
        "Start",
        "GatePassed",
        "LapCompleted",
        "RaceFinished",
    }


def test_episode_provenance_fingerprints_scenario_track_code_and_optional_model(
    tmp_path, monkeypatch
) -> None:
    track = write_straight_track(tmp_path)
    scenario = Scenario(
        name="fingerprinted-straight",
        seed=12,
        track=str(track),
        duration_s=1.5,
        physics_dt_s=0.01,
        telemetry_period_s=0.05,
        wind_world_m_s=[0.03, -0.02, 0.0],
        motor_efficiency=[0.99, 0.98, 0.99, 0.97],
    )
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"a reproducible learned-policy artifact")
    metadata = checkpoint.with_suffix(".metadata.json")
    metadata.write_text('{"schema":"flightstack-v1"}\n', encoding="utf-8")
    monkeypatch.setattr(
        runner_module,
        "_repository_identity",
        lambda: ("0123456789abcdef0123456789abcdef01234567", True),
    )

    result = run_episode(
        scenario,
        ClassicalRacePilot,
        pilot_model_identity=checkpoint_model_identity(checkpoint),
    )
    provenance = result.provenance
    expected_scenario = scenario.to_mapping()
    expected_scenario_hash = hashlib.sha256(
        json.dumps(expected_scenario, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert provenance.scenario_definition == expected_scenario
    assert provenance.scenario_definition_sha256 == expected_scenario_hash
    assert provenance.track_source_path == str(track.resolve())
    assert provenance.track_content_sha256 == hashlib.sha256(track.read_bytes()).hexdigest()
    assert provenance.vehicle_config_hash
    assert provenance.git_revision == "0123456789abcdef0123456789abcdef01234567"
    assert provenance.git_dirty is True
    assert provenance.pilot_model_identity == {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "metadata_path": str(metadata.resolve()),
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    assert result.to_mapping()["provenance"]["track_content_sha256"] == (
        provenance.track_content_sha256
    )


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
    replay = result.write_artifacts(tmp_path / "ground-artifacts")
    replay_events = [
        event
        for frame in load_replay(replay.replay_path)["frames"]
        for event in frame["events"]
    ]
    assert any(event["type"] == "Collision" for event in replay_events)


def test_collision_precedes_a_same_tick_terminal_gate_crossing(tmp_path, monkeypatch) -> None:
    track = write_straight_track(tmp_path)
    scenario = Scenario(
        name="collision-precedence",
        seed=1,
        track=str(track),
        duration_s=0.01,
        physics_dt_s=0.01,
        telemetry_period_s=0.01,
    )

    class CrossingRuntime:
        def __init__(self, config, dt: float, state: FlightState) -> None:
            del config
            self.dt = dt
            self.state = state

        def step(self, command, disturbance):
            del command, disturbance
            current = FlightState(
                sim_time_s=self.state.sim_time_s + self.dt,
                position_world_m=np.array([0.0, 1.0, 1.5]),
                velocity_world_m_s=np.zeros(3),
                q_body_to_world_wxyz=self.state.q_body_to_world_wxyz,
                body_rate_rad_s=np.zeros(3),
                motor_thrust_n=self.state.motor_thrust_n,
            )
            self.state = current
            return current, SimpleNamespace(saturated=False), None

    monkeypatch.setattr(runner_module, "FixedStepRuntime", CrossingRuntime)
    monkeypatch.setattr(runner_module, "_collision_object", lambda *_args: "ground")

    result = run_episode(scenario, ClassicalRacePilot)

    assert result.metrics.termination == "crashed"
    assert result.metrics.collisions == 1
    assert result.metrics.gates_passed == 0
    assert [event["type"] for event in result.events] == ["Reset", "Start", "Collision"]
