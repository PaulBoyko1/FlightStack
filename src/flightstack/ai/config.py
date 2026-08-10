"""Versioned, validated training-facing configuration loaded from TOML."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


def default_ai_config_path() -> Path:
    """Return the tracked configuration for the v1 state-based race task."""
    return Path(__file__).resolve().parents[3] / "config" / "ai" / "racing_v1.toml"


def _positive(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _table(data: dict[str, Any], name: str) -> dict[str, Any]:
    result = data.get(name)
    if not isinstance(result, dict):
        raise ValueError(f"AI config must contain a [{name}] table")
    return result


@dataclass(frozen=True)
class EnvironmentConfig:
    """Episode limits and deterministic reset randomization."""

    schema_version: str
    physics_dt_s: float
    control_substeps: int
    max_episode_s: float
    max_distance_from_course_m: float
    max_altitude_m: float
    vehicle_radius_m: float
    initial_xy_jitter_m: float
    initial_altitude_jitter_m: float
    initial_yaw_jitter_rad: float

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("environment.schema_version must be nonempty")
        _positive(self.physics_dt_s, "environment.physics_dt_s")
        _positive_int(self.control_substeps, "environment.control_substeps")
        _positive(self.max_episode_s, "environment.max_episode_s")
        _positive(self.max_distance_from_course_m, "environment.max_distance_from_course_m")
        _positive(self.max_altitude_m, "environment.max_altitude_m")
        _nonnegative(self.vehicle_radius_m, "environment.vehicle_radius_m")
        _nonnegative(self.initial_xy_jitter_m, "environment.initial_xy_jitter_m")
        _nonnegative(self.initial_altitude_jitter_m, "environment.initial_altitude_jitter_m")
        _nonnegative(self.initial_yaw_jitter_rad, "environment.initial_yaw_jitter_rad")

    @property
    def control_dt_s(self) -> float:
        """Elapsed simulation time represented by one policy decision."""
        return self.physics_dt_s * self.control_substeps


@dataclass(frozen=True)
class ObservationConfig:
    """Explicit normalisation scales for the 27-value policy observation."""

    schema_version: str
    body_velocity_scale_m_s: float
    gate_vector_scale_m: float
    distance_scale_m: float
    speed_scale_m_s: float

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("observation.schema_version must be nonempty")
        _positive(self.body_velocity_scale_m_s, "observation.body_velocity_scale_m_s")
        _positive(self.gate_vector_scale_m, "observation.gate_vector_scale_m")
        _positive(self.distance_scale_m, "observation.distance_scale_m")
        _positive(self.speed_scale_m_s, "observation.speed_scale_m_s")


@dataclass(frozen=True)
class RewardConfig:
    """Signed weights for separately instrumented racing reward terms."""

    schema_version: str
    progress_per_m: float
    gate_pass: float
    lap_complete: float
    collision: float
    out_of_bounds: float
    action_delta: float
    angular_rate: float
    time_per_s: float

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("reward.schema_version must be nonempty")
        for name, value in self.__dict__.items():
            if name != "schema_version":
                _finite(value, f"reward.{name}")


@dataclass(frozen=True)
class RacingAIConfig:
    """All versioned knobs needed to construct a reference race environment."""

    environment: EnvironmentConfig
    observation: ObservationConfig
    reward: RewardConfig

    def to_mapping(self) -> dict[str, object]:
        """Return the full numeric policy contract in a canonical mapping.

        Schema labels alone do not protect a trained policy from a changed
        observation scale or control interval.  This mapping deliberately
        includes every checked-in training-facing value so checkpoints can pin
        the exact contract they were optimized against.
        """
        return {
            "environment": asdict(self.environment),
            "observation": asdict(self.observation),
            "reward": asdict(self.reward),
        }

    @property
    def config_hash(self) -> str:
        """SHA-256 identity of the complete validated AI configuration."""
        encoded = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_toml(cls, path: str | Path | None = None) -> RacingAIConfig:
        """Load the single tracked AI contract without duplicating constants."""
        source = default_ai_config_path() if path is None else Path(path)
        with source.open("rb") as handle:
            data: dict[str, Any] = tomllib.load(handle)
        environment = _table(data, "environment")
        observation = _table(data, "observation")
        reward = _table(data, "reward")
        try:
            return cls(
                environment=EnvironmentConfig(
                    schema_version=str(environment["schema_version"]),
                    physics_dt_s=_positive(environment["physics_dt_s"], "environment.physics_dt_s"),
                    control_substeps=_positive_int(
                        environment["control_substeps"], "environment.control_substeps"
                    ),
                    max_episode_s=_positive(
                        environment["max_episode_s"], "environment.max_episode_s"
                    ),
                    max_distance_from_course_m=_positive(
                        environment["max_distance_from_course_m"],
                        "environment.max_distance_from_course_m",
                    ),
                    max_altitude_m=_positive(
                        environment["max_altitude_m"], "environment.max_altitude_m"
                    ),
                    vehicle_radius_m=_nonnegative(
                        environment["vehicle_radius_m"], "environment.vehicle_radius_m"
                    ),
                    initial_xy_jitter_m=_nonnegative(
                        environment["initial_xy_jitter_m"], "environment.initial_xy_jitter_m"
                    ),
                    initial_altitude_jitter_m=_nonnegative(
                        environment["initial_altitude_jitter_m"],
                        "environment.initial_altitude_jitter_m",
                    ),
                    initial_yaw_jitter_rad=_nonnegative(
                        environment["initial_yaw_jitter_rad"],
                        "environment.initial_yaw_jitter_rad",
                    ),
                ),
                observation=ObservationConfig(
                    schema_version=str(observation["schema_version"]),
                    body_velocity_scale_m_s=_positive(
                        observation["body_velocity_scale_m_s"],
                        "observation.body_velocity_scale_m_s",
                    ),
                    gate_vector_scale_m=_positive(
                        observation["gate_vector_scale_m"], "observation.gate_vector_scale_m"
                    ),
                    distance_scale_m=_positive(
                        observation["distance_scale_m"], "observation.distance_scale_m"
                    ),
                    speed_scale_m_s=_positive(
                        observation["speed_scale_m_s"], "observation.speed_scale_m_s"
                    ),
                ),
                reward=RewardConfig(
                    schema_version=str(reward["schema_version"]),
                    progress_per_m=_finite(reward["progress_per_m"], "reward.progress_per_m"),
                    gate_pass=_finite(reward["gate_pass"], "reward.gate_pass"),
                    lap_complete=_finite(reward["lap_complete"], "reward.lap_complete"),
                    collision=_finite(reward["collision"], "reward.collision"),
                    out_of_bounds=_finite(
                        reward["out_of_bounds"], "reward.out_of_bounds"
                    ),
                    action_delta=_finite(reward["action_delta"], "reward.action_delta"),
                    angular_rate=_finite(reward["angular_rate"], "reward.angular_rate"),
                    time_per_s=_finite(reward["time_per_s"], "reward.time_per_s"),
                ),
            )
        except KeyError as exc:
            raise ValueError(f"AI config is missing {exc.args[0]!r}") from exc


def load_racing_ai_config() -> RacingAIConfig:
    """Load FlightStack's checked-in state-racing configuration."""
    return RacingAIConfig.from_toml()
