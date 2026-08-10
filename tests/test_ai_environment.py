import importlib.util
from dataclasses import replace

import numpy as np
import pytest

from flightstack.ai.config import RacingAIConfig, load_racing_ai_config
from flightstack.ai.environment import FlightStackRaceEnv, make_gymnasium_env
from flightstack.ai.errors import OptionalTrainingDependencyError
from flightstack.ai.spaces import BoxSpace
from flightstack.race import Gate, Track


def test_reset_seed_and_fixed_step_action_are_deterministic() -> None:
    left = FlightStackRaceEnv()
    right = FlightStackRaceEnv()
    left_obs, left_info = left.reset(seed=173)
    right_obs, right_info = right.reset(seed=173)
    np.testing.assert_array_equal(left_obs, right_obs)
    assert left_info["state"] == right_info["state"]

    observation, reward, terminated, truncated, info = left.step(np.zeros(4))
    assert left.observation_space.contains(observation)
    assert reward == pytest.approx(-0.0004)
    assert not terminated
    assert not truncated
    assert info["sim_time_s"] == pytest.approx(0.02)
    assert info["pilot_command"]["collective_thrust_n"] == pytest.approx(
        left.vehicle.hover_thrust_n
    )


def test_environment_terminates_on_ground_collision_and_requires_reset() -> None:
    env = FlightStackRaceEnv()
    env.reset(seed=2)
    crashed = env.state
    crashed.position_world_m[2] = 0.1
    env.runtime.reset(crashed)
    _observation, _reward, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated
    assert not truncated
    assert info["termination_reason"] == "ground"
    assert any(event["type"] == "Collision" for event in info["events"])
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(np.zeros(4))


def test_environment_records_collision_before_a_same_tick_finish(monkeypatch) -> None:
    gate = Gate(
        center_world_m=[0.0, 0.0, 1.0],
        normal_world=[0.0, 1.0, 0.0],
        right_world=[-1.0, 0.0, 0.0],
        up_world=[0.0, 0.0, 1.0],
        half_width_m=2.0,
        half_height_m=2.0,
        gate_id="finish",
        frame_thickness_m=0.0,
        frame_depth_m=0.0,
    )
    base = load_racing_ai_config()
    ai_config = RacingAIConfig(
        environment=replace(
            base.environment,
            control_substeps=1,
            initial_xy_jitter_m=0.0,
            initial_altitude_jitter_m=0.0,
            initial_yaw_jitter_rad=0.0,
        ),
        observation=base.observation,
        reward=base.reward,
    )
    env = FlightStackRaceEnv(
        track=Track(
            name="collision-wins",
            gates=(gate,),
            gate_order=(1,),
            start_position_world_m=[0.0, -1.0, 1.0],
        ),
        ai_config=ai_config,
    )
    env.reset(seed=0)
    previous = env.state.copy()
    previous.position_world_m[:] = [0.0, -1.0, 1.0]
    env.runtime.reset(previous)
    current = previous.copy()
    current.sim_time_s = ai_config.environment.physics_dt_s
    current.position_world_m[:] = [0.0, 1.0, 1.0]
    monkeypatch.setattr(env.runtime, "step", lambda command: (current, None, None))
    monkeypatch.setattr(env, "_collision_id", lambda state: "synthetic-frame")

    _observation, _reward, terminated, _truncated, info = env.step(np.zeros(4))

    assert terminated
    assert not env.race.finished
    assert env.race.collisions == 1
    assert [event["type"] for event in info["events"]] == ["Collision"]


def test_environment_truncates_at_configured_time_limit() -> None:
    base = load_racing_ai_config()
    ai_config = RacingAIConfig(
        environment=replace(base.environment, max_episode_s=0.001),
        observation=base.observation,
        reward=base.reward,
    )
    env = FlightStackRaceEnv(ai_config=ai_config)
    env.reset(seed=3)
    _observation, _reward, terminated, truncated, info = env.step(np.zeros(4))
    assert not terminated
    assert truncated
    assert info["termination_reason"] == "time_limit"


def test_environment_rejects_bad_actions_and_options() -> None:
    env = FlightStackRaceEnv()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="normalized action"):
        env.step([0.0, 0.0])
    with pytest.raises(ValueError, match="reset options"):
        env.reset(options={"unsupported": True})


def test_dependency_free_box_space_and_native_gym_adapter_contract() -> None:
    space = BoxSpace(-1.0, 1.0, (2,))
    assert space.contains([0.0, 1.0])
    assert not space.contains([2.0, 0.0])
    if importlib.util.find_spec("gymnasium") is None:
        with pytest.raises(OptionalTrainingDependencyError, match=r"\[train\]"):
            make_gymnasium_env()
    else:
        native = make_gymnasium_env()
        observation, _info = native.reset(seed=4)
        assert native.observation_space.contains(observation)
        native.close()
