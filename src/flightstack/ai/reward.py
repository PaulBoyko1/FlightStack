"""Inspectable delta-based reward calculation for the state race task."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from flightstack.ai.config import RewardConfig
from flightstack.race import Collision, GatePassed, LapCompleted, RaceEvent

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class RewardTerms:
    """Every contribution exposed to training telemetry and experiment logs."""

    progress: float
    gate_pass: float
    lap_complete: float
    collision: float
    out_of_bounds: float
    action_delta: float
    angular_rate: float
    time: float

    @property
    def total(self) -> float:
        return float(sum(asdict(self).values()))

    def to_mapping(self) -> dict[str, float]:
        result = {name: float(value) for name, value in asdict(self).items()}
        result["total"] = self.total
        return result


def race_reward(
    *,
    previous_distance_m: float,
    current_distance_m: float,
    previous_action: Vector,
    action: Vector,
    body_rate_rad_s: Vector,
    max_body_rate_rad_s: Vector,
    events: Iterable[RaceEvent],
    out_of_bounds: bool,
    control_dt_s: float,
    config: RewardConfig,
) -> RewardTerms:
    """Calculate a transparent event-aware reward without stationary exploits.

    Progress is a *decrease* in distance, so moving toward the currently
    required gate is positive.  A valid gate pass uses a zero distance for the
    final segment and separately earns its event bonus; the environment then
    starts the following segment with the next gate's real distance.
    """
    values = (previous_distance_m, current_distance_m, control_dt_s)
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("distances and control_dt_s must be nonnegative finite values")
    before = np.asarray(previous_action, dtype=np.float64)
    after = np.asarray(action, dtype=np.float64)
    body_rate = np.asarray(body_rate_rad_s, dtype=np.float64)
    max_rate = np.asarray(max_body_rate_rad_s, dtype=np.float64)
    expected_shapes = ((before, (4,)), (after, (4,)), (body_rate, (3,)), (max_rate, (3,)))
    if any(array.shape != expected for array, expected in expected_shapes):
        raise ValueError("actions must be 4-vectors and body rates must be 3-vectors")
    if not all(np.all(np.isfinite(array)) for array in (before, after, body_rate, max_rate)):
        raise ValueError("actions and body rates must be finite")
    if np.any(max_rate <= 0.0):
        raise ValueError("max_body_rate_rad_s must be positive")
    recorded = tuple(events)
    progress = config.progress_per_m * (previous_distance_m - current_distance_m)
    gate_pass = config.gate_pass * sum(isinstance(event, GatePassed) for event in recorded)
    lap_complete = config.lap_complete * sum(
        isinstance(event, LapCompleted) for event in recorded
    )
    collision = config.collision * sum(isinstance(event, Collision) for event in recorded)
    action_delta = config.action_delta * float(np.square(after - before).sum())
    normalized_rate = body_rate / max_rate
    angular_rate = config.angular_rate * float(np.square(normalized_rate).sum())
    return RewardTerms(
        progress=float(progress),
        gate_pass=float(gate_pass),
        lap_complete=float(lap_complete),
        collision=float(collision),
        out_of_bounds=float(config.out_of_bounds if out_of_bounds else 0.0),
        action_delta=float(action_delta),
        angular_rate=float(angular_rate),
        time=float(config.time_per_s * control_dt_s),
    )
