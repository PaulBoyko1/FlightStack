"""Simple deterministic IMU model for estimator/controller tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.math.quaternion import rotate_inverse

Vector = NDArray[np.float64]


@dataclass(frozen=True)
class IMUSample:
    timestamp_s: float
    gyro_rad_s: Vector
    accel_m_s2: Vector


class IMUSimulator:
    def __init__(
        self,
        *,
        gyro_bias_rad_s: ArrayLike = (0.0, 0.0, 0.0),
        gyro_noise_std: float = 0.0,
        accel_noise_std: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.gyro_bias = np.asarray(gyro_bias_rad_s, dtype=np.float64)
        if self.gyro_bias.shape != (3,) or not np.all(np.isfinite(self.gyro_bias)):
            raise ValueError("gyro_bias_rad_s must be a finite 3-vector")
        if (
            gyro_noise_std < 0.0
            or accel_noise_std < 0.0
            or not np.isfinite(gyro_noise_std)
            or not np.isfinite(accel_noise_std)
        ):
            raise ValueError("noise standard deviations must be finite and nonnegative")
        self.gyro_noise_std = float(gyro_noise_std)
        self.accel_noise_std = float(accel_noise_std)
        self.rng = np.random.default_rng(seed)

    def sample(
        self,
        timestamp_s: float,
        attitude_q: ArrayLike,
        body_rate: ArrayLike,
    ) -> IMUSample:
        try:
            timestamp = float(timestamp_s)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp_s must be finite") from exc
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite")
        omega = np.asarray(body_rate, dtype=np.float64)
        if omega.shape != (3,) or not np.all(np.isfinite(omega)):
            raise ValueError("body_rate must be a finite 3-vector")
        gyro = omega + self.gyro_bias + self.rng.normal(0.0, self.gyro_noise_std, 3)
        accel_world = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
        accel = rotate_inverse(attitude_q, accel_world)
        accel += self.rng.normal(0.0, self.accel_noise_std, 3)
        return IMUSample(timestamp, gyro, accel)
