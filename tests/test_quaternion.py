import numpy as np
import pytest

from flightstack.math.quaternion import conjugate, error_vector, from_axis_angle, from_euler, geodesic_angle, integrate_body_rate, multiply, normalize, rotate


def test_q_and_negative_q_represent_same_attitude() -> None:
    q = from_euler(0.4, -0.3, 0.7)
    assert geodesic_angle(q, -q) == pytest.approx(0.0, abs=1e-12)


def test_multiply_by_conjugate_is_identity() -> None:
    q = from_euler(0.2, 0.4, -0.6)
    np.testing.assert_allclose(normalize(multiply(q, conjugate(q))), [1, 0, 0, 0], atol=1e-12)


def test_constant_rate_integration_matches_axis_angle() -> None:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 1.2])
    for _ in range(1000):
        q = integrate_body_rate(q, omega, 0.001)
    expected = from_axis_angle([0, 0, 1], 1.2)
    assert geodesic_angle(q, expected) < 1e-9
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)


def test_nonidentity_target_error_is_body_frame() -> None:
    current = from_euler(*np.deg2rad([20, -12, 8]))
    target = from_euler(*np.deg2rad([30, 25, -20]))
    step = error_vector(current, target)
    moved = integrate_body_rate(current, step, 1e-3)
    assert geodesic_angle(moved, target) < geodesic_angle(current, target)


def test_rotate_quarter_turn_about_z() -> None:
    q = from_axis_angle([0, 0, 1], np.pi / 2)
    np.testing.assert_allclose(rotate(q, [1, 0, 0]), [0, 1, 0], atol=1e-12)


def test_zero_quaternion_rejected() -> None:
    with pytest.raises(ValueError):
        normalize([0, 0, 0, 0])
