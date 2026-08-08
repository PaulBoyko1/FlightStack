"""Closed-loop attitude simulation and telemetry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.control.attitude import AttitudeController
from flightstack.math.quaternion import geodesic_angle
from flightstack.sim.rigid_body import RigidBody

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


@dataclass(frozen=True)
class Telemetry:
    time_s: Vector
    attitude: Matrix
    body_rate: Matrix
    desired_rate: Matrix
    torque: Matrix
    attitude_error_rad: Vector

    @property
    def final_error_deg(self) -> float:
        return float(np.rad2deg(self.attitude_error_rad[-1]))

    @property
    def peak_rate_rad_s(self) -> float:
        return float(np.max(np.linalg.norm(self.body_rate, axis=1)))


def simulate_attitude_step(
    plant: RigidBody,
    controller: AttitudeController,
    target_q: ArrayLike,
    *,
    duration_s: float = 4.0,
    dt: float = 0.002,
) -> Telemetry:
    if duration_s <= 0.0 or dt <= 0.0:
        raise ValueError("duration_s and dt must be positive")
    target = np.asarray(target_q, dtype=float)
    steps = int(round(duration_s / dt))
    if steps < 1:
        raise ValueError("duration must include at least one simulation step")

    time_s = np.arange(steps + 1, dtype=float) * dt
    attitude = np.empty((steps + 1, 4))
    body_rate = np.empty((steps + 1, 3))
    desired_rate = np.empty((steps + 1, 3))
    torque = np.empty((steps + 1, 3))
    error = np.empty(steps + 1)

    attitude[0] = plant.state.attitude
    body_rate[0] = plant.state.body_rate
    desired_rate[0] = controller.desired_body_rate(plant.state.attitude, target)
    torque[0] = np.zeros(3)
    error[0] = geodesic_angle(plant.state.attitude, target)

    controller.reset()
    for index in range(1, steps + 1):
        desired, terms = controller.update(
            plant.state.attitude,
            target,
            plant.state.body_rate,
            dt,
        )
        state = plant.step(terms.output, dt)
        attitude[index] = state.attitude
        body_rate[index] = state.body_rate
        desired_rate[index] = desired
        torque[index] = terms.output
        error[index] = geodesic_angle(state.attitude, target)

    return Telemetry(time_s, attitude, body_rate, desired_rate, torque, error)
