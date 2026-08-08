import numpy as np

from flightstack.control.pid import VectorPID


def make_pid() -> VectorPID:
    return VectorPID(
        [2, 2, 2],
        [1, 1, 1],
        [0.1, 0.1, 0.1],
        output_limit=1.0,
        integral_limit=0.5,
        derivative_cutoff_hz=20,
    )


def test_pid_saturates_and_prevents_windup() -> None:
    pid = make_pid()
    for _ in range(100):
        terms = pid.update([10, 0, 0], [0, 0, 0], 0.01)
    assert terms.output[0] == 1.0
    assert pid.integral[0] < 0.01


def test_derivative_on_measurement_has_no_setpoint_kick() -> None:
    pid = make_pid()
    pid.update([0, 0, 0], [0, 0, 0], 0.01)
    terms = pid.update([1, 0, 0], [0, 0, 0], 0.01)
    np.testing.assert_allclose(terms.derivative, 0.0, atol=1e-12)


def test_integrator_unwinds_after_error_reverses() -> None:
    pid = VectorPID([0, 0, 0], [1, 1, 1], [0, 0, 0], output_limit=2, integral_limit=1)
    for _ in range(20):
        pid.update([1, 0, 0], [0, 0, 0], 0.01)
    before = pid.integral[0]
    for _ in range(10):
        pid.update([-1, 0, 0], [0, 0, 0], 0.01)
    assert 0 < pid.integral[0] < before
