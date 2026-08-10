"""Deterministic batched wrapper over the exact reference race environment."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.ai.config import RacingAIConfig
from flightstack.ai.environment import FlightStackRaceEnv
from flightstack.race import Track
from flightstack.sim.vehicle import VehicleConfig


class ReferenceVectorEnv:
    """Batch independent reference environments without a second physics model.

    This is intentionally a deterministic *reference* vector environment, not
    a claim of JAX throughput.  It is useful for paired-seed evaluation and
    keeps every batch member on the same FixedStepRuntime/RaceState equations.
    A JAX backend is deliberately not shipped until it can pass a dedicated
    physics parity harness rather than becoming a superficially vectorized
    alternative model.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        vehicle: VehicleConfig | None = None,
        track: Track | None = None,
        ai_config: RacingAIConfig | None = None,
    ) -> None:
        if isinstance(num_envs, bool) or int(num_envs) <= 0:
            raise ValueError("num_envs must be a positive integer")
        self.envs = tuple(
            FlightStackRaceEnv(vehicle=vehicle, track=track, ai_config=ai_config)
            for _ in range(int(num_envs))
        )
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space

    def reset(
        self, seeds: int | Sequence[int | None] | None = None
    ) -> tuple[NDArray[np.float32], tuple[dict[str, object], ...]]:
        """Reset all environments with explicit, reproducible per-env seeds."""
        resolved = self._seeds(seeds)
        results = tuple(env.reset(seed=seed) for env, seed in zip(self.envs, resolved, strict=True))
        observations, infos = zip(*results, strict=True)
        return np.stack(observations), tuple(infos)

    def step(
        self, actions: ArrayLike
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        tuple[dict[str, object], ...],
    ]:
        """Step a batch of normalized actions through the exact Python plant."""
        batch = np.asarray(actions, dtype=np.float64)
        if batch.shape != (self.num_envs, 4):
            raise ValueError(f"actions must have shape ({self.num_envs}, 4)")
        results = tuple(env.step(action) for env, action in zip(self.envs, batch, strict=True))
        observations, rewards, terminated, truncated, infos = zip(*results, strict=True)
        return (
            np.stack(observations),
            np.asarray(rewards, dtype=np.float64),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
            tuple(infos),
        )

    def close(self) -> None:
        """Close all member environments."""
        for env in self.envs:
            env.close()

    def _seeds(self, seeds: int | Sequence[int | None] | None) -> tuple[int | None, ...]:
        if seeds is None:
            return (None,) * self.num_envs
        if isinstance(seeds, bool):
            raise ValueError("seeds must be integers, a sequence, or None")
        if isinstance(seeds, int):
            return tuple(seeds + index for index in range(self.num_envs))
        result = tuple(seeds)
        if len(result) != self.num_envs:
            raise ValueError(f"seeds must contain exactly {self.num_envs} entries")
        has_invalid_seed = any(
            seed is not None and (isinstance(seed, bool) or not isinstance(seed, int))
            for seed in result
        )
        if has_invalid_seed:
            raise ValueError("seeds must contain integers or None")
        return result
