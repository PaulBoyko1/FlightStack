"""Cascaded quaternion attitude and body-rate controller."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.control.pid import PIDTerms, VectorPID
from flightstack.math.quaternion import rotation_vector_error

Vector = NDArray[np.float64]


def _gain(value: ArrayLike, name: str) -> Vector:
    gain = np.asarray(value, dtype=np.float64)
    if gain.shape != (3,) or not np.all(np.isfinite(gain)):
        raise ValueError(f"{name} must be a finite 3-vector")
    if np.any(gain < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return gain


def _rate_limit(value: ArrayLike | float) -> Vector:
    limit = np.asarray(value, dtype=np.float64)
    if limit.ndim == 0:
        limit = np.full(3, float(limit), dtype=np.float64)
    if limit.shape != (3,) or not np.all(np.isfinite(limit)) or np.any(limit <= 0.0):
        raise ValueError("max_rate_rad_s must be positive and finite")
    return limit


class AttitudeController:
    """Quaternion P outer loop feeding a rate PID inner loop."""

    def __init__(
        self,
        *,
        attitude_kp: ArrayLike,
        rate_pid: VectorPID,
        max_rate_rad_s: ArrayLike | float,
    ) -> None:
        self.attitude_kp = _gain(attitude_kp, "attitude_kp")
        self.max_rate_rad_s = _rate_limit(max_rate_rad_s)
        self.rate_pid = rate_pid

    def reset(self) -> None:
        self.rate_pid.reset()

    def desired_body_rate(self, current_q: ArrayLike, target_q: ArrayLike) -> Vector:
        rate = self.attitude_kp * rotation_vector_error(current_q, target_q)
        return np.asarray(np.clip(rate, -self.max_rate_rad_s, self.max_rate_rad_s), dtype=np.float64)

    def update(
        self,
        current_q: ArrayLike,
        target_q: ArrayLike,
        body_rate: ArrayLike,
        dt: float,
    ) -> tuple[Vector, PIDTerms]:
        desired_rate = self.desired_body_rate(current_q, target_q)
        terms = self.rate_pid.update(desired_rate, body_rate, dt)
        return desired_rate, terms
