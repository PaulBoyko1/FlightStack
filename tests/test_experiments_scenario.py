from __future__ import annotations

import numpy as np
import pytest

from flightstack.experiments import Scenario, default_scenarios_dir, load_scenario


def test_tracked_scenario_resolves_seeded_wind_and_motor_degradation() -> None:
    scenario = load_scenario("technical-eight-wind-degraded")
    first = scenario.realize()
    second = scenario.realize()

    assert scenario.name == "technical-eight-wind-degraded"
    assert default_scenarios_dir().is_dir()
    np.testing.assert_allclose(
        first.disturbance.wind_world_m_s,
        second.disturbance.wind_world_m_s,
    )
    np.testing.assert_allclose(
        first.disturbance.motor_efficiency,
        second.disturbance.motor_efficiency,
    )
    assert np.linalg.norm(first.disturbance.wind_world_m_s) > 0.0
    assert np.any(first.disturbance.motor_efficiency < 1.0)
    assert scenario.with_seed(7).realize().seed == 7
    assert scenario.to_mapping()["track"] == "technical-eight"


def test_scenario_rejects_nonintegral_duration_and_invalid_motor_efficiency() -> None:
    with pytest.raises(ValueError, match="integer multiple"):
        Scenario(name="bad-duration", seed=1, duration_s=0.011, physics_dt_s=0.002)
    with pytest.raises(ValueError, match="motor_efficiency"):
        Scenario(name="bad-motor", seed=1, motor_efficiency=[1.0, 1.0, 1.0, 1.1])


def test_scenario_loader_reports_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="scenario file not found"):
        load_scenario("does-not-exist")
