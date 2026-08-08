"""Lightweight quaternion complementary attitude estimator."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.math.quaternion import integrate_body_rate, normalize, rotate_inverse

Vector = NDArray[np.float64]


class ComplementaryAttitudeEstimator:
    """Gyro propagation with accelerometer gravity correction.

    Yaw remains gyro-driven because no magnetometer is modeled. The correction is applied
    as a body-rate feedback term, similar to the proportional part of a Mahony filter.
    """

    def __init__(
        self,
        *,
        accel_correction_gain: float = 1.8,
        initial_q: ArrayLike = (1, 0, 0, 0),
    ) -> None:
        if accel_correction_gain < 0.0 or not np.isfinite(accel_correction_gain):
            raise ValueError("accel_correction_gain must be finite and nonnegative")
        self.gain = float(accel_correction_gain)
        self.attitude = normalize(initial_q)

    def reset(self, attitude_q: ArrayLike = (1, 0, 0, 0)) -> None:
        self.attitude = normalize(attitude_q)

    def update(self, gyro_rad_s: ArrayLike, accel_m_s2: ArrayLike, dt: float) -> Vector:
        gyro = np.asarray(gyro_rad_s, dtype=float)
        accel = np.asarray(accel_m_s2, dtype=float)
        if gyro.shape != (3,) or accel.shape != (3,):
            raise ValueError("gyro and accel must be 3-vectors")
        accel_norm = float(np.linalg.norm(accel))
        correction = np.zeros(3)
        if accel_norm > 1e-6:
            measured_up_body = accel / accel_norm
            predicted_up_body = rotate_inverse(self.attitude, [0.0, 0.0, 1.0])
            correction = self.gain * np.cross(measured_up_body, predicted_up_body)
        self.attitude = integrate_body_rate(self.attitude, gyro + correction, dt)
        return self.attitude.copy()
