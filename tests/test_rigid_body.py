import numpy as np
import pytest

from flightstack.sim.rigid_body import RigidBody


def test_rejects_nonsymmetric_inertia() -> None:
    inertia = np.array([[1.0, 0.2, 0], [0.0, 1.0, 0], [0, 0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        RigidBody(inertia)


def test_rejects_nonpositive_inertia() -> None:
    with pytest.raises(ValueError, match="positive definite"):
        RigidBody(np.diag([1.0, 1.0, -1.0]))


def test_zero_torque_zero_rate_stays_still() -> None:
    body = RigidBody(np.diag([1.0, 2.0, 3.0]))
    state = body.step([0, 0, 0], 0.01)
    np.testing.assert_allclose(state.body_rate, 0.0)
    np.testing.assert_allclose(state.attitude, [1, 0, 0, 0])


def test_torque_free_spherical_body_conserves_rate_and_energy() -> None:
    body = RigidBody(np.eye(3) * 0.02, body_rate=[0.3, -0.4, 0.5])
    initial_rate = body.state.body_rate.copy()
    initial_energy = body.rotational_energy()
    for _ in range(1000):
        body.step([0, 0, 0], 0.001)
    np.testing.assert_allclose(body.state.body_rate, initial_rate, atol=1e-12)
    assert body.rotational_energy() == pytest.approx(initial_energy, abs=1e-12)
