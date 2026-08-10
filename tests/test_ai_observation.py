from dataclasses import replace

import numpy as np
import pytest

from flightstack.ai.config import load_racing_ai_config
from flightstack.ai.observation import (
    OBSERVATION_COMPONENTS,
    OBSERVATION_DIMENSION,
    build_observation,
    distance_to_next_gate,
)
from flightstack.race import Gate, RaceState, Track
from flightstack.sim.vehicle import FlightState, VehicleConfig


def vehicle() -> VehicleConfig:
    return VehicleConfig.from_toml()


def race() -> RaceState:
    gate = Gate(
        center_world_m=[3.0, 0.0, 1.0],
        normal_world=[1.0, 0.0, 0.0],
        right_world=[0.0, 1.0, 0.0],
        up_world=[0.0, 0.0, 1.0],
        half_width_m=1.0,
        half_height_m=1.0,
        gate_id="first",
    )
    result = RaceState(Track(name="one", gates=(gate,), gate_order=(1,)))
    result.start()
    return result


def state(config: VehicleConfig) -> FlightState:
    result = FlightState.hovering(config)
    result.velocity_world_m_s = np.array([3.0, -6.0, 1.0])
    result.body_rate_rad_s = np.array([1.0, -2.0, 0.5])
    return result


def test_observation_is_documented_local_27_float_vector() -> None:
    config = vehicle()
    current = state(config)
    observation = build_observation(
        current,
        race(),
        config,
        load_racing_ai_config().observation,
        np.array([0.2, -0.3, 0.4, -0.5]),
    )
    assert observation.dtype == np.float32
    assert observation.shape == (OBSERVATION_DIMENSION,)
    assert len(OBSERVATION_COMPONENTS) == OBSERVATION_DIMENSION
    np.testing.assert_allclose(observation[:3], [0.25, -0.5, 1.0 / 12.0])
    np.testing.assert_allclose(observation[6:9], [0.25, 0.0, 0.0])
    np.testing.assert_allclose(observation[21:25], [0.2, -0.3, 0.4, -0.5])
    assert np.all(observation >= -1.0)
    assert np.all(observation <= 1.0)


def test_observation_is_invariant_to_quaternion_sign() -> None:
    config = vehicle()
    current = state(config)
    negated = current.copy()
    negated.q_body_to_world_wxyz *= -1.0
    kwargs = (race(), config, load_racing_ai_config().observation, np.zeros(4))
    np.testing.assert_allclose(
        build_observation(current, *kwargs),
        build_observation(negated, *kwargs),
    )


def test_observation_zeros_gate_geometry_when_race_is_idle() -> None:
    config = vehicle()
    idle = race()
    idle.reset()
    observation = build_observation(
        state(config), idle, config, load_racing_ai_config().observation, np.zeros(4)
    )
    np.testing.assert_allclose(observation[6:18], 0.0)
    assert observation[25] == 0.0
    assert distance_to_next_gate(state(config), idle) == 0.0


def test_observation_rejects_unknown_schema_or_bad_previous_action() -> None:
    config = vehicle()
    unknown = replace(load_racing_ai_config().observation, schema_version="unknown")
    with pytest.raises(ValueError, match="unsupported observation schema"):
        build_observation(state(config), race(), config, unknown)
    with pytest.raises(ValueError, match="previous_action"):
        build_observation(
            state(config), race(), config, load_racing_ai_config().observation, np.zeros(3)
        )
