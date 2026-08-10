"""Deterministic headless experiment, evaluation, and robustness helpers."""

from flightstack.experiments.evaluation import (
    ConfidenceInterval,
    PairedComparison,
    PairedEvaluation,
    PilotAggregate,
    RobustnessCase,
    bootstrap_mean_confidence_interval,
    build_robustness_grid,
    paired_evaluate,
)
from flightstack.experiments.runner import (
    EpisodeArtifacts,
    EpisodeMetrics,
    EpisodeProvenance,
    EpisodeResult,
    PilotFactory,
    TelemetrySample,
    checkpoint_model_identity,
    run_episode,
)
from flightstack.experiments.scenario import (
    Scenario,
    ScenarioRealization,
    default_scenarios_dir,
    load_scenario,
)

__all__ = [
    "ConfidenceInterval",
    "EpisodeArtifacts",
    "EpisodeMetrics",
    "EpisodeProvenance",
    "EpisodeResult",
    "PairedComparison",
    "PairedEvaluation",
    "PilotAggregate",
    "PilotFactory",
    "RobustnessCase",
    "Scenario",
    "ScenarioRealization",
    "TelemetrySample",
    "bootstrap_mean_confidence_interval",
    "build_robustness_grid",
    "checkpoint_model_identity",
    "default_scenarios_dir",
    "load_scenario",
    "paired_evaluate",
    "run_episode",
]
