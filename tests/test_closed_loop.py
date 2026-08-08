import numpy as np
import pytest

from flightstack.math.quaternion import from_euler
from flightstack.sim.rigid_body import RigidBody
from flightstack.sim.runner import simulate_attitude_step
from flightstack.sim.scenarios import reference_controller, run_reference_step


def test_reference_nonidentity_target_converges() -> None:
    telemetry = run_reference_step()
    assert telemetry.final_error_deg < 1.0
    assert np.linalg.norm(telemetry.body_rate[-1]) < 0.03


@pytest.mark.parametrize(
    ("initial_deg", "target_deg"),
    [
        ([0, 0, 0], [45, 45, 45]),
        ([60, 20, 40], [-30, 45, 70]),
        ([-45, 30, -120], [20, -25, 90]),
    ],
)
def test_multiaxis_targets_converge(initial_deg: list[float], target_deg: list[float]) -> None:
    plant = RigidBody(
        np.diag([0.018, 0.020, 0.035]),
        attitude=from_euler(*np.deg2rad(initial_deg)),
    )
    target = from_euler(*np.deg2rad(target_deg))
    telemetry = simulate_attitude_step(
        plant,
        reference_controller(),
        target,
        duration_s=5.0,
        dt=0.002,
    )
    assert telemetry.final_error_deg < 1.5
    assert np.linalg.norm(telemetry.body_rate[-1]) < 0.05
