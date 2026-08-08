"""Vector PID controller with derivative-on-measurement and anti-windup."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]


def _gain(value: ArrayLike, name: str) -> Vector:
    gain = np.asarray(value, dtype=float)
    if gain.shape != (3,) or not np.all(np.isfinite(gain)):
        raise ValueError(f"{name} must be a finite 3-vector")
    if np.any(gain < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return gain


def _limit(value: ArrayLike | float, name: str) -> Vector:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(3, float(arr))
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return arr


@dataclass(frozen=True)
class PIDTerms:
    proportional: Vector
    integral: Vector
    derivative: Vector
    output: Vector


class VectorPID:
    """Three-axis PID with filtered derivative-on-measurement and conditional integration."""

    def __init__(self, kp: ArrayLike, ki: ArrayLike, kd: ArrayLike, *, output_limit: ArrayLike | float, integral_limit: ArrayLike | float, derivative_cutoff_hz: float = 35.0) -> None:
        self.kp = _gain(kp, "kp")
        self.ki = _gain(ki, "ki")
        self.kd = _gain(kd, "kd")
        self.output_limit = _limit(output_limit, "output_limit")
        self.integral_limit = _limit(integral_limit, "integral_limit")
        if derivative_cutoff_hz <= 0.0 or not np.isfinite(derivative_cutoff_hz):
            raise ValueError("derivative_cutoff_hz must be positive and finite")
        self.derivative_cutoff_hz = float(derivative_cutoff_hz)
        self.integral = np.zeros(3)
        self._previous_measurement: Vector | None = None
        self._filtered_derivative = np.zeros(3)

    def reset(self) -> None:
        self.integral = np.zeros(3)
        self._previous_measurement = None
        self._filtered_derivative = np.zeros(3)

    def update(self, setpoint: ArrayLike, measurement: ArrayLike, dt: float) -> PIDTerms:
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        target = np.asarray(setpoint, dtype=float)
        measured = np.asarray(measurement, dtype=float)
        if target.shape != (3,) or measured.shape != (3,):
            raise ValueError("setpoint and measurement must be 3-vectors")
        if not np.all(np.isfinite(target)) or not np.all(np.isfinite(measured)):
            raise ValueError("setpoint and measurement must be finite")
        error = target - measured
        p = self.kp * error
        raw_derivative = np.zeros(3) if self._previous_measurement is None else -(measured - self._previous_measurement) / dt
        self._previous_measurement = measured.copy()
        tau = 1.0 / (2.0 * np.pi * self.derivative_cutoff_hz)
        alpha = dt / (tau + dt)
        self._filtered_derivative += alpha * (raw_derivative - self._filtered_derivative)
        d = self.kd * self._filtered_derivative
        proposed_integral = np.clip(self.integral + error * dt, -self.integral_limit, self.integral_limit)
        unsaturated = p + self.ki * proposed_integral + d
        pushing_high = (unsaturated > self.output_limit) & (error > 0.0)
        pushing_low = (unsaturated < -self.output_limit) & (error < 0.0)
        accept = ~(pushing_high | pushing_low)
        self.integral = np.where(accept, proposed_integral, self.integral)
        i = self.ki * self.integral
        output = np.clip(p + i + d, -self.output_limit, self.output_limit)
        return PIDTerms(p, i, d, output)
