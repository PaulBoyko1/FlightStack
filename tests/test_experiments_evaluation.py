from __future__ import annotations

import json

import numpy as np
import pytest

from flightstack.experiments import (
    Scenario,
    bootstrap_mean_confidence_interval,
    build_robustness_grid,
    paired_evaluate,
)
from flightstack.runtime.autonomy import ClassicalRacePilot


def write_straight_track(tmp_path):
    source = tmp_path / "paired-track.json"
    source.write_text(
        json.dumps(
            {
                "name": "paired-straight",
                "default_laps": 1,
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
    return source


def test_bootstrap_interval_is_seeded_and_handles_one_observation() -> None:
    first = bootstrap_mean_confidence_interval([1.0, 2.0, 4.0], seed=5, bootstrap_samples=80)
    second = bootstrap_mean_confidence_interval([1.0, 2.0, 4.0], seed=5, bootstrap_samples=80)
    singleton = bootstrap_mean_confidence_interval([3.0], seed=8, bootstrap_samples=20)

    assert first == second
    assert first.lower <= np.mean([1.0, 2.0, 4.0]) <= first.upper
    assert singleton.lower == singleton.upper == 3.0
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean_confidence_interval([1.0, 2.0], confidence=1.0)


def test_paired_classical_self_comparison_has_zero_seed_matched_delta(tmp_path) -> None:
    track = write_straight_track(tmp_path)

    def scenario_factory(seed: int) -> Scenario:
        return Scenario(
            name="paired-straight",
            seed=seed,
            track=str(track),
            duration_s=1.5,
            physics_dt_s=0.01,
            telemetry_period_s=0.05,
            wind_world_m_s=[0.02, 0.0, 0.0],
            wind_jitter_std_m_s=0.01,
            motor_efficiency=[0.99, 0.98, 0.99, 0.98],
            motor_efficiency_jitter=0.002,
        )

    evaluation = paired_evaluate(
        {"classical-a": ClassicalRacePilot, "classical-b": ClassicalRacePilot},
        [2, 7],
        scenario_factory,
        bootstrap_samples=40,
    )

    comparison = evaluation.comparisons["classical-b"]
    assert len(evaluation.episodes) == 4
    assert evaluation.aggregates["classical-a"].completion_rate == 1.0
    assert comparison.matched_pairs == 2
    assert comparison.mean_elapsed_time_delta_s == pytest.approx(0.0)
    assert comparison.completion_rate_delta == pytest.approx(0.0)
    assert comparison.elapsed_time_delta_mean_ci.lower == pytest.approx(0.0)


def test_robustness_grid_expands_wind_motor_and_seed_axes() -> None:
    base = Scenario(
        name="grid-base",
        seed=0,
        duration_s=0.02,
        physics_dt_s=0.01,
        motor_efficiency=[1.0, 0.9, 1.0, 0.9],
    )

    cases = build_robustness_grid(
        base,
        wind_speeds_m_s=[0.0, 1.5],
        motor_efficiency_scales=[1.0, 0.8],
        seeds=[10, 11],
        wind_direction_world=[0.0, 2.0, 0.0],
    )

    assert len(cases) == 8
    stressed = next(
        case
        for case in cases
        if case.wind_speed_m_s == 1.5
        and case.motor_efficiency_scale == 0.8
        and case.scenario.seed == 11
    )
    np.testing.assert_allclose(stressed.scenario.wind_world_m_s, [0.0, 1.5, 0.0])
    np.testing.assert_allclose(stressed.scenario.motor_efficiency, [0.8, 0.72, 0.8, 0.72])
    assert stressed.to_mapping()["scenario"]["seed"] == 11
