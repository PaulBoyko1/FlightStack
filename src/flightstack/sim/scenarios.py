"""Reference simulation scenarios and controller gains."""

from __future__ import annotations

import numpy as np

from flightstack.control.attitude import AttitudeController
from flightstack.control.pid import VectorPID
from flightstack.math.quaternion import from_euler
from flightstack.sim.rigid_body import RigidBody
from flightstack.sim.runner import Telemetry, simulate_attitude_step


def reference_controller() -> AttitudeController:
    return AttitudeController(attitude_kp=np.array([5.0, 5.0, 4.0]), max_rate_rad_s=np.deg2rad([260.0, 260.0, 220.0]), rate_pid=VectorPID(kp=np.array([0.13, 0.13, 0.11]), ki=np.array([0.025, 0.025, 0.020]), kd=np.array([0.004, 0.004, 0.003]), output_limit=np.array([0.24, 0.24, 0.15]), integral_limit=np.array([0.6, 0.6, 0.6]), derivative_cutoff_hz=25.0))


def run_reference_step(*, duration_s: float = 4.0, dt: float = 0.002) -> Telemetry:
    plant = RigidBody(np.diag([0.018, 0.020, 0.035]), attitude=from_euler(*np.deg2rad([20.0, -12.0, 8.0])))
    target = from_euler(*np.deg2rad([30.0, 25.0, -20.0]))
    return simulate_attitude_step(plant, reference_controller(), target, duration_s=duration_s, dt=dt)
