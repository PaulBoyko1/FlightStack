"""Reproducible, data-driven conditions for FlightStack experiments.

Scenarios own *environment* randomness.  Pilots receive only canonical
``FlightState``/``RaceState`` information and cannot reach into a global random
generator.  Resolving a scenario therefore produces a concrete, inspectable
``Disturbance`` that can be replayed exactly by another pilot.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.sim.vehicle import Disturbance

Vector: TypeAlias = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> Vector:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result.copy()


def _positive(value: object, name: str) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _seed(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("seed must be a nonnegative integer")
    try:
        result = int(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ValueError("seed must be a nonnegative integer") from exc
    if result < 0 or result != value:
        raise ValueError("seed must be a nonnegative integer")
    return int(result)


def default_scenarios_dir() -> Path:
    """Return the tracked directory containing TOML experiment scenarios."""
    return Path(__file__).resolve().parents[3] / "config" / "scenarios"


@dataclass(frozen=True)
class ScenarioRealization:
    """Concrete seeded disturbance parameters used by one headless episode."""

    scenario_name: str
    seed: int
    disturbance: Disturbance

    def to_mapping(self) -> dict[str, object]:
        return {
            "scenario_name": self.scenario_name,
            "seed": self.seed,
            "disturbance": {
                "force_world_n": self.disturbance.force_world_n.tolist(),
                "torque_body_nm": self.disturbance.torque_body_nm.tolist(),
                "wind_world_m_s": self.disturbance.wind_world_m_s.tolist(),
                "motor_efficiency": self.disturbance.motor_efficiency.tolist(),
            },
        }


@dataclass(frozen=True)
class Scenario:
    """Validated episode conditions with explicit, local PRNG ownership.

    ``wind_world_m_s`` and ``motor_efficiency`` are deterministic base values.
    Their optional jitter is sampled exactly once per seed, making paired pilot
    comparisons see the same realized physical conditions.
    """

    name: str
    seed: int
    track: str = "technical-eight"
    laps: int = 1
    duration_s: float = 20.0
    physics_dt_s: float = 0.002
    telemetry_period_s: float = 0.05
    vehicle_radius_m: float = 0.13
    wind_world_m_s: Vector = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    wind_jitter_std_m_s: float = 0.0
    force_world_n: Vector = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    torque_body_nm: Vector = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    motor_efficiency: Vector = field(default_factory=lambda: np.ones(4, dtype=np.float64))
    motor_efficiency_jitter: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("scenario name must be a nonempty string")
        if not isinstance(self.track, str) or not self.track:
            raise ValueError("track must be a nonempty string")
        object.__setattr__(self, "seed", _seed(self.seed))
        if isinstance(self.laps, bool) or int(self.laps) != self.laps or self.laps <= 0:
            raise ValueError("laps must be a positive integer")
        object.__setattr__(self, "laps", int(self.laps))
        duration = _positive(self.duration_s, "duration_s")
        dt = _positive(self.physics_dt_s, "physics_dt_s")
        if duration < dt:
            raise ValueError("duration_s must be at least one physics step")
        steps = round(duration / dt)
        if not np.isclose(steps * dt, duration, rtol=0.0, atol=1e-10):
            raise ValueError("duration_s must be an integer multiple of physics_dt_s")
        object.__setattr__(self, "duration_s", duration)
        object.__setattr__(self, "physics_dt_s", dt)
        object.__setattr__(
            self,
            "telemetry_period_s",
            _positive(self.telemetry_period_s, "telemetry_period_s"),
        )
        object.__setattr__(
            self,
            "vehicle_radius_m",
            _nonnegative(self.vehicle_radius_m, "vehicle_radius_m"),
        )
        object.__setattr__(
            self,
            "wind_world_m_s",
            _vector(self.wind_world_m_s, 3, "wind_world_m_s"),
        )
        object.__setattr__(
            self,
            "wind_jitter_std_m_s",
            _nonnegative(self.wind_jitter_std_m_s, "wind_jitter_std_m_s"),
        )
        object.__setattr__(
            self,
            "force_world_n",
            _vector(self.force_world_n, 3, "force_world_n"),
        )
        object.__setattr__(
            self,
            "torque_body_nm",
            _vector(self.torque_body_nm, 3, "torque_body_nm"),
        )
        efficiency = _vector(self.motor_efficiency, 4, "motor_efficiency")
        if np.any(efficiency < 0.0) or np.any(efficiency > 1.0):
            raise ValueError("motor_efficiency must be in [0, 1]")
        object.__setattr__(self, "motor_efficiency", efficiency)
        object.__setattr__(
            self,
            "motor_efficiency_jitter",
            _nonnegative(self.motor_efficiency_jitter, "motor_efficiency_jitter"),
        )

    @property
    def physics_steps(self) -> int:
        """Number of exact fixed steps in the bounded episode horizon."""
        return int(round(self.duration_s / self.physics_dt_s))

    def realize(self) -> ScenarioRealization:
        """Sample the scenario's one fixed disturbance from its own seed."""
        generator = np.random.default_rng(self.seed)
        wind = self.wind_world_m_s + generator.normal(
            0.0,
            self.wind_jitter_std_m_s,
            size=3,
        )
        efficiency = np.clip(
            self.motor_efficiency
            + generator.normal(0.0, self.motor_efficiency_jitter, size=4),
            0.0,
            1.0,
        )
        return ScenarioRealization(
            scenario_name=self.name,
            seed=self.seed,
            disturbance=Disturbance(
                force_world_n=self.force_world_n,
                torque_body_nm=self.torque_body_nm,
                wind_world_m_s=np.asarray(wind, dtype=np.float64),
                motor_efficiency=np.asarray(efficiency, dtype=np.float64),
            ),
        )

    def with_seed(self, seed: int) -> Scenario:
        """Return the same conditions with a new deterministic PRNG key."""
        return replace(self, seed=_seed(seed))

    def to_mapping(self) -> dict[str, object]:
        return {
            "name": self.name,
            "seed": self.seed,
            "track": self.track,
            "laps": self.laps,
            "duration_s": self.duration_s,
            "physics_dt_s": self.physics_dt_s,
            "telemetry_period_s": self.telemetry_period_s,
            "vehicle_radius_m": self.vehicle_radius_m,
            "wind_world_m_s": self.wind_world_m_s.tolist(),
            "wind_jitter_std_m_s": self.wind_jitter_std_m_s,
            "force_world_n": self.force_world_n.tolist(),
            "torque_body_nm": self.torque_body_nm.tolist(),
            "motor_efficiency": self.motor_efficiency.tolist(),
            "motor_efficiency_jitter": self.motor_efficiency_jitter,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> Scenario:
        """Build a scenario from the repository's compact TOML mapping."""
        try:
            return cls(
                name=str(mapping["name"]),
                seed=_seed(mapping["seed"]),
                track=str(mapping.get("track", "technical-eight")),
                laps=int(mapping.get("laps", 1)),
                duration_s=float(mapping.get("duration_s", 20.0)),
                physics_dt_s=float(mapping.get("physics_dt_s", 0.002)),
                telemetry_period_s=float(mapping.get("telemetry_period_s", 0.05)),
                vehicle_radius_m=float(mapping.get("vehicle_radius_m", 0.13)),
                wind_world_m_s=mapping.get("wind_world_m_s", [0.0, 0.0, 0.0]),
                wind_jitter_std_m_s=float(mapping.get("wind_jitter_std_m_s", 0.0)),
                force_world_n=mapping.get("force_world_n", [0.0, 0.0, 0.0]),
                torque_body_nm=mapping.get("torque_body_nm", [0.0, 0.0, 0.0]),
                motor_efficiency=mapping.get("motor_efficiency", [1.0, 1.0, 1.0, 1.0]),
                motor_efficiency_jitter=float(mapping.get("motor_efficiency_jitter", 0.0)),
            )
        except KeyError as exc:
            raise ValueError(f"scenario is missing {exc.args[0]!r}") from exc


def load_scenario(path: str | Path) -> Scenario:
    """Load a TOML scenario; bare names resolve in :func:`default_scenarios_dir`."""
    candidate = Path(path)
    if candidate.suffix == "" and not candidate.exists():
        candidate = default_scenarios_dir() / f"{candidate.name}.toml"
    try:
        with candidate.open("rb") as handle:
            decoded: Any = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"scenario file not found: {candidate}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"scenario TOML must decode to an object: {candidate}")
    return Scenario.from_mapping(decoded)


__all__ = [
    "Scenario",
    "ScenarioRealization",
    "default_scenarios_dir",
    "load_scenario",
]
