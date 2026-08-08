"""Rotational rigid-body plant for controller development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.math.quaternion import integrate_body_rate, normalize

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass
class RigidBodyState:
    attitude: Vector
    body_rate: Vector


class RigidBody:
    def __init__(
        self,
        inertia: ArrayLike,
        *,
        attitude: ArrayLike | None = None,
        body_rate: ArrayLike | None = None,
    ) -> None:
        matrix = np.asarray(inertia, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("inertia must be a finite 3x3 matrix")
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
            raise ValueError("inertia must be symmetric")
        if np.min(np.linalg.eigvalsh(matrix)) <= 0.0:
            raise ValueError("inertia must be positive definite")
        self.inertia: Matrix = matrix
        self.state = RigidBodyState(
            normalize([1.0, 0.0, 0.0, 0.0] if attitude is None else attitude),
            np.asarray(
                np.zeros(3) if body_rate is None else body_rate,
                dtype=np.float64,
            ),
        )
        if self.state.body_rate.shape != (3,) or not np.all(np.isfinite(self.state.body_rate)):
            raise ValueError("body_rate must be a finite 3-vector")

    def angular_acceleration(self, torque_body: ArrayLike) -> Vector:
        torque = np.asarray(torque_body, dtype=np.float64)
        if torque.shape != (3,) or not np.all(np.isfinite(torque)):
            raise ValueError("torque_body must be a finite 3-vector")
        omega = self.state.body_rate
        gyroscopic = np.cross(omega, self.inertia @ omega)
        acceleration = np.linalg.solve(self.inertia, torque - gyroscopic)
        return np.asarray(acceleration, dtype=np.float64)

    def step(self, torque_body: ArrayLike, dt: float) -> RigidBodyState:
        if dt <= 0.0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        alpha = self.angular_acceleration(torque_body)
        self.state.body_rate = self.state.body_rate + alpha * dt
        self.state.attitude = integrate_body_rate(self.state.attitude, self.state.body_rate, dt)
        return RigidBodyState(self.state.attitude.copy(), self.state.body_rate.copy())

    def rotational_energy(self) -> float:
        omega = self.state.body_rate
        return float(0.5 * omega @ self.inertia @ omega)

    def angular_momentum_body(self) -> Vector:
        return np.asarray(self.inertia @ self.state.body_rate, dtype=np.float64)
