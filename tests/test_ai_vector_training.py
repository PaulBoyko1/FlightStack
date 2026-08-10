import importlib.util
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from flightstack.ai.errors import OptionalTrainingDependencyError
from flightstack.ai.training import (
    PPOTrainingConfig,
    _make_vector_environment,
    _parser,
    train_ppo,
)
from flightstack.ai.vector import ReferenceVectorEnv


class _FakeDummyVecEnv:
    def __init__(self, factories: list[Callable[[], Any]]) -> None:
        self.factories = factories


class _FakeSubprocVecEnv:
    def __init__(self, factories: list[Callable[[], Any]]) -> None:
        self.factories = factories


class _FakeVecEnvModule:
    DummyVecEnv = _FakeDummyVecEnv
    SubprocVecEnv = _FakeSubprocVecEnv


def test_reference_vector_env_keeps_exact_envs_seeded_and_shaped() -> None:
    env = ReferenceVectorEnv(2)
    left, _left_info = env.reset(seeds=100)
    right, _right_info = env.reset(seeds=[100, 101])
    np.testing.assert_array_equal(left, right)
    observation, rewards, terminated, truncated, infos = env.step(np.zeros((2, 4)))
    assert observation.shape == (2, 27)
    assert rewards.shape == (2,)
    assert not np.any(terminated)
    assert not np.any(truncated)
    assert len(infos) == 2
    with pytest.raises(ValueError, match="shape"):
        env.step(np.zeros((1, 4)))
    with pytest.raises(ValueError, match="exactly"):
        env.reset(seeds=[1])


def test_training_vectorizer_uses_subprocesses_only_for_multi_env_runs() -> None:
    def factory() -> object:
        return object()

    single = _make_vector_environment(_FakeVecEnvModule, [factory])
    parallel = _make_vector_environment(_FakeVecEnvModule, [factory, factory, factory, factory])

    assert isinstance(single, _FakeDummyVecEnv)
    assert isinstance(parallel, _FakeSubprocVecEnv)
    assert len(parallel.factories) == 4
    with pytest.raises(ValueError, match="at least one"):
        _make_vector_environment(_FakeVecEnvModule, [])


def test_training_parser_exposes_parallel_environment_count(tmp_path) -> None:
    args = _parser().parse_args(
        ["--output", str(tmp_path), "--timesteps", "5000", "--seed", "7", "--n-envs", "6"]
    )

    assert args.timesteps == 5000
    assert args.seed == 7
    assert args.n_envs == 6
    assert not args.smoke


def test_training_config_and_optional_dependency_failure(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        PPOTrainingConfig(n_steps=0)
    with pytest.raises(ValueError, match="n_envs"):
        PPOTrainingConfig(n_envs=0)
    if importlib.util.find_spec("stable_baselines3") is None:
        with pytest.raises(OptionalTrainingDependencyError, match=r"\[train\]"):
            train_ppo(tmp_path)
