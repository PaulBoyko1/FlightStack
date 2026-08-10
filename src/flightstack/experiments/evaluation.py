"""Paired deterministic evaluation and a compact robustness-grid builder."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.experiments.runner import EpisodeResult, PilotFactory, run_episode
from flightstack.experiments.scenario import Scenario
from flightstack.sim.vehicle import VehicleConfig

ScenarioFactory: TypeAlias = Callable[[int], Scenario]
Vector = NDArray[np.float64]


def _values(values: ArrayLike, name: str) -> Vector:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def _seed(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class ConfidenceInterval:
    """Deterministic percentile-bootstrap interval for a sample mean."""

    lower: float
    upper: float
    confidence: float
    bootstrap_samples: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "bootstrap_samples": self.bootstrap_samples,
        }


def bootstrap_mean_confidence_interval(
    values: ArrayLike,
    *,
    seed: int = 0,
    bootstrap_samples: int = 1_000,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Return a seeded nonparametric bootstrap confidence interval for a mean."""
    sample = _values(values, "values")
    if isinstance(bootstrap_samples, bool) or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be finite and in (0, 1)")
    count = int(bootstrap_samples)
    if count != bootstrap_samples:
        raise ValueError("bootstrap_samples must be a positive integer")
    _seed(seed)
    if sample.size == 1:
        mean = float(sample[0])
        return ConfidenceInterval(mean, mean, float(confidence), count)
    generator = np.random.default_rng(seed)
    resampled_means = np.empty(count, dtype=np.float64)
    for index in range(count):
        indices = generator.integers(0, sample.size, size=sample.size)
        resampled_means[index] = float(np.mean(sample[indices]))
    tail = (1.0 - confidence) / 2.0
    return ConfidenceInterval(
        lower=float(np.quantile(resampled_means, tail)),
        upper=float(np.quantile(resampled_means, 1.0 - tail)),
        confidence=float(confidence),
        bootstrap_samples=count,
    )


@dataclass(frozen=True)
class PilotAggregate:
    """Per-pilot summary over a common set of bounded scenarios."""

    pilot_name: str
    episode_count: int
    completions: int
    completion_rate: float
    mean_elapsed_time_s: float
    median_elapsed_time_s: float
    elapsed_time_stddev_s: float
    elapsed_time_mean_ci: ConfidenceInterval

    def to_mapping(self) -> dict[str, object]:
        return {
            "pilot_name": self.pilot_name,
            "episode_count": self.episode_count,
            "completions": self.completions,
            "completion_rate": self.completion_rate,
            "mean_elapsed_time_s": self.mean_elapsed_time_s,
            "median_elapsed_time_s": self.median_elapsed_time_s,
            "elapsed_time_stddev_s": self.elapsed_time_stddev_s,
            "elapsed_time_mean_ci": self.elapsed_time_mean_ci.to_mapping(),
        }


@dataclass(frozen=True)
class PairedComparison:
    """Contender-minus-baseline outcome deltas computed on matching seeds.

    Elapsed-time deltas are reported only for seeds on which *both* pilots
    completed the course.  A crash or timeout therefore changes completion
    outcomes but can never be mistaken for a faster finish.
    """

    baseline_name: str
    contender_name: str
    matched_pairs: int
    mean_elapsed_time_delta_s: float | None
    median_elapsed_time_delta_s: float | None
    elapsed_time_delta_stddev_s: float | None
    completion_rate_delta: float
    elapsed_time_delta_mean_ci: ConfidenceInterval | None
    completed_pairs: int | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "contender_name": self.contender_name,
            "matched_pairs": self.matched_pairs,
            "completed_pairs": self.completed_pairs,
            "mean_elapsed_time_delta_s": self.mean_elapsed_time_delta_s,
            "median_elapsed_time_delta_s": self.median_elapsed_time_delta_s,
            "elapsed_time_delta_stddev_s": self.elapsed_time_delta_stddev_s,
            "completion_rate_delta": self.completion_rate_delta,
            "elapsed_time_delta_mean_ci": (
                None
                if self.elapsed_time_delta_mean_ci is None
                else self.elapsed_time_delta_mean_ci.to_mapping()
            ),
        }


@dataclass(frozen=True)
class PairedEvaluation:
    """Episodes plus per-pilot and paired-seed aggregate statistics."""

    baseline_name: str
    episodes: tuple[EpisodeResult, ...]
    aggregates: Mapping[str, PilotAggregate]
    comparisons: Mapping[str, PairedComparison]

    def to_mapping(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "aggregates": {
                name: aggregate.to_mapping() for name, aggregate in self.aggregates.items()
            },
            "comparisons": {
                name: comparison.to_mapping() for name, comparison in self.comparisons.items()
            },
            "episodes": [episode.to_mapping() for episode in self.episodes],
        }


def _standard_deviation(values: Vector) -> float:
    return 0.0 if values.size < 2 else float(np.std(values, ddof=1))


def _aggregate(
    pilot_name: str,
    results: list[EpisodeResult],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> PilotAggregate:
    elapsed = _values([result.metrics.elapsed_time_s for result in results], "elapsed times")
    completions = sum(result.metrics.completed for result in results)
    return PilotAggregate(
        pilot_name=pilot_name,
        episode_count=len(results),
        completions=completions,
        completion_rate=float(completions / len(results)),
        mean_elapsed_time_s=float(np.mean(elapsed)),
        median_elapsed_time_s=float(np.median(elapsed)),
        elapsed_time_stddev_s=_standard_deviation(elapsed),
        elapsed_time_mean_ci=bootstrap_mean_confidence_interval(
            elapsed,
            seed=bootstrap_seed,
            bootstrap_samples=bootstrap_samples,
        ),
    )


def paired_evaluate(
    pilots: Mapping[str, PilotFactory],
    seeds: Iterable[int],
    scenario_factory: ScenarioFactory,
    *,
    baseline_name: str | None = None,
    vehicle_config: VehicleConfig | None = None,
    bootstrap_seed: int = 0,
    bootstrap_samples: int = 1_000,
) -> PairedEvaluation:
    """Evaluate fresh pilots on exactly matching seeded scenario realizations.

    It deliberately does not label a factory as human, classical, or learned;
    callers supply names and factories.  The result records completion rates
    separately from elapsed-time deltas.  Only pairs in which both pilots
    finish contribute a time delta, so crashes and timeouts cannot masquerade
    as fast laps.
    """
    pilot_items = tuple(pilots.items())
    if not pilot_items:
        raise ValueError("pilots must contain at least one named factory")
    if any(not isinstance(name, str) or not name for name, _factory in pilot_items):
        raise ValueError("pilot names must be nonempty strings")
    normalized_seeds = tuple(_seed(seed) for seed in seeds)
    if not normalized_seeds:
        raise ValueError("seeds must contain at least one value")
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique for paired evaluation")
    selected_baseline = pilot_items[0][0] if baseline_name is None else baseline_name
    if selected_baseline not in pilots:
        raise ValueError("baseline_name must name one supplied pilot")
    _seed(bootstrap_seed)

    episodes: list[EpisodeResult] = []
    by_name: dict[str, list[EpisodeResult]] = {name: [] for name, _factory in pilot_items}
    by_name_seed: dict[tuple[str, int], EpisodeResult] = {}
    for seed in normalized_seeds:
        scenario = scenario_factory(seed)
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario_factory must return a Scenario")
        paired_scenario = scenario.with_seed(seed)
        for name, factory in pilot_items:
            result = run_episode(
                paired_scenario,
                factory,
                pilot_name=name,
                vehicle_config=vehicle_config,
            )
            episodes.append(result)
            by_name[name].append(result)
            by_name_seed[(name, seed)] = result

    aggregates = {
        name: _aggregate(
            name,
            results,
            bootstrap_seed=bootstrap_seed + index,
            bootstrap_samples=bootstrap_samples,
        )
        for index, (name, results) in enumerate(by_name.items())
    }
    comparisons: dict[str, PairedComparison] = {}
    for index, (name, _factory) in enumerate(pilot_items):
        if name == selected_baseline:
            continue
        baseline_results = [by_name_seed[(selected_baseline, seed)] for seed in normalized_seeds]
        contender_results = [by_name_seed[(name, seed)] for seed in normalized_seeds]
        finished_deltas = [
            contender.metrics.elapsed_time_s - baseline.metrics.elapsed_time_s
            for baseline, contender in zip(baseline_results, contender_results, strict=True)
            if baseline.metrics.completed and contender.metrics.completed
        ]
        deltas = (
            _values(finished_deltas, "paired completed elapsed-time deltas")
            if finished_deltas
            else None
        )
        completion_delta = float(
            np.mean([result.metrics.completed for result in contender_results])
            - np.mean([result.metrics.completed for result in baseline_results])
        )
        comparisons[name] = PairedComparison(
            baseline_name=selected_baseline,
            contender_name=name,
            matched_pairs=len(normalized_seeds),
            completed_pairs=len(finished_deltas),
            mean_elapsed_time_delta_s=None if deltas is None else float(np.mean(deltas)),
            median_elapsed_time_delta_s=None if deltas is None else float(np.median(deltas)),
            elapsed_time_delta_stddev_s=None if deltas is None else _standard_deviation(deltas),
            completion_rate_delta=completion_delta,
            elapsed_time_delta_mean_ci=(
                None
                if deltas is None
                else bootstrap_mean_confidence_interval(
                    deltas,
                    seed=bootstrap_seed + len(pilot_items) + index,
                    bootstrap_samples=bootstrap_samples,
                )
            ),
        )
    return PairedEvaluation(
        baseline_name=selected_baseline,
        episodes=tuple(episodes),
        aggregates=aggregates,
        comparisons=comparisons,
    )


@dataclass(frozen=True)
class RobustnessCase:
    """One generated wind/motor condition in a Cartesian robustness grid."""

    wind_speed_m_s: float
    motor_efficiency_scale: float
    scenario: Scenario

    def to_mapping(self) -> dict[str, object]:
        return {
            "wind_speed_m_s": self.wind_speed_m_s,
            "motor_efficiency_scale": self.motor_efficiency_scale,
            "scenario": self.scenario.to_mapping(),
        }


def _unit_direction(value: ArrayLike) -> Vector:
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("wind_direction_world must be a finite vector with shape (3,)")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("wind_direction_world must be nonzero")
    return np.asarray(direction / norm, dtype=np.float64)


def build_robustness_grid(
    base_scenario: Scenario,
    *,
    wind_speeds_m_s: Iterable[float],
    motor_efficiency_scales: Iterable[float],
    seeds: Iterable[int],
    wind_direction_world: ArrayLike = (1.0, 0.0, 0.0),
) -> tuple[RobustnessCase, ...]:
    """Expand a small Cartesian grid while preserving every other condition.

    Wind speeds replace the base steady wind along the supplied direction.
    Motor scales multiply the base efficiency vector (and are clipped to the
    physical interval).  Both choices are explicit in every returned scenario.
    """
    if not isinstance(base_scenario, Scenario):
        raise TypeError("base_scenario must be a Scenario")
    direction = _unit_direction(wind_direction_world)
    wind_levels = tuple(float(value) for value in wind_speeds_m_s)
    motor_levels = tuple(float(value) for value in motor_efficiency_scales)
    grid_seeds = tuple(_seed(seed) for seed in seeds)
    if not wind_levels or not motor_levels or not grid_seeds:
        raise ValueError("wind levels, motor levels, and seeds must all be nonempty")
    if any(not np.isfinite(value) or value < 0.0 for value in wind_levels):
        raise ValueError("wind_speeds_m_s must be finite and nonnegative")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in motor_levels):
        raise ValueError("motor_efficiency_scales must be finite and in [0, 1]")
    cases: list[RobustnessCase] = []
    for wind in wind_levels:
        for scale in motor_levels:
            for seed in grid_seeds:
                scenario = replace(
                    base_scenario,
                    name=(f"{base_scenario.name}-wind{wind:g}-eff{scale:g}-seed{seed}"),
                    seed=seed,
                    wind_world_m_s=direction * wind,
                    motor_efficiency=np.clip(base_scenario.motor_efficiency * scale, 0.0, 1.0),
                )
                cases.append(RobustnessCase(wind, scale, scenario))
    return tuple(cases)


__all__ = [
    "ConfidenceInterval",
    "PairedComparison",
    "PairedEvaluation",
    "PilotAggregate",
    "RobustnessCase",
    "ScenarioFactory",
    "bootstrap_mean_confidence_interval",
    "build_robustness_grid",
    "paired_evaluate",
]
