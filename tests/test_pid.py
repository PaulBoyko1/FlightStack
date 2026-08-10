import csv
from pathlib import Path

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


def test_python_pid_matches_shared_cpp_golden_fixture() -> None:
    fixture = Path(__file__).parent / "data" / "python_cpp_rate_pid_v1.csv"
    pid = VectorPID(
        [1.2, 1.0, 0.8],
        [0.4, 0.3, 0.2],
        [0.05, 0.04, 0.03],
        output_limit=[2.0, 2.0, 1.5],
        integral_limit=[0.8, 0.8, 0.6],
        derivative_cutoff_hz=35.0,
    )

    with fixture.open(newline="", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            setpoint = [float(row[f"setpoint_{axis}"]) for axis in "xyz"]
            measurement = [float(row[f"measurement_{axis}"]) for axis in "xyz"]
            expected_output = [float(row[f"output_{axis}"]) for axis in "xyz"]
            expected_integral = [float(row[f"integral_state_{axis}"]) for axis in "xyz"]
            terms = pid.update(setpoint, measurement, float(row["dt"]))
            np.testing.assert_allclose(terms.output, expected_output, rtol=0.0, atol=1e-12)
            np.testing.assert_allclose(pid.integral, expected_integral, rtol=0.0, atol=1e-12)
