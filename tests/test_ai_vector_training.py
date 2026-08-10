import importlib.util

import numpy as np
import pytest

from flightstack.ai.errors import OptionalTrainingDependencyError
from flightstack.ai.training import PPOTrainingConfig, train_ppo
from flightstack.ai.vector import ReferenceVectorEnv


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


def test_training_config_and_optional_dependency_failure(tmp_path) -> None:
    with pytest.raises(ValueError, match="n_steps"):
        PPOTrainingConfig(n_steps=0)
    if importlib.util.find_spec("stable_baselines3") is None:
        with pytest.raises(OptionalTrainingDependencyError, match=r"\[train\]"):
            train_ppo(tmp_path)
