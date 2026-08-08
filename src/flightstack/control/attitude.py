"""Cascaded quaternion attitude and body-rate controller."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.control.pid import PIDTerms, VectorPID
from flightstack.math.quaternion import rotation_vector_error

Vector = NDArray[np.float64]


class AttitudeController:
    """Quaternion P outer loop feeding a rate PID inner loop."""

    def __init__(
        self,
        *,
        attitude_kp: ArrayLike,
        rate_pid: VectorPID,
        max_rate_rad_s: ArrayLike | float,
    ) -> None:
        self.attitude_kp = np.asarray(attitude_kp, dtype=float)
        if self.attitude_kp.shape != (3,) or np.any(self.attitude_kp < 0.0):
            raise ValueError("attitude_kp must be a nonnegative 3-vector")
        max_rate = np.asarray(max_rate_rad_s, dtype=float)
        if max_rate.ndim == 0:
            max_rate = np.full(3, float(max_rate))
        if max_rate.shape != (3,) or np.any(max_rate <= 0.0):
            raise ValueError("max_rate_rad_s must be positive")
        self.max_rate_rad_s = max_rate
        self.rate_pid = rate_pid

    def reset(self) -> None:
        self.rate_pid.reset()

    def desired_body_rate(self, current_q: ArrayLike, target_q: ArrayLike) -> Vector:
        rate = self.attitude_kp * rotation_vector_error(current_q, target_q)
        return np.clip(rate, -self.max_rate_rad_s, self.max_rate_rad_s)

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
