"""Validate the Python reference runtime against the shared Rust fixture."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from flightstack.sim.vehicle import (
    Disturbance,
    FixedStepRuntime,
    FlightState,
    PilotCommand,
    VehicleConfig,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = REPOSITORY_ROOT / "tests" / "data" / "python_rust_6dof_ctbr_v1.toml"


def _fixture() -> Mapping[str, Any]:
    with FIXTURE_PATH.open("rb") as fixture_file:
        return cast(Mapping[str, Any], tomllib.load(fixture_file))


def _array(source: Mapping[str, Any], name: str, size: int) -> np.ndarray:
    value = np.asarray(source[name], dtype=np.float64)
    if value.shape != (size,):
        raise AssertionError(f"fixture {name} must have shape ({size},), got {value.shape}")
    return value


def _state(source: Mapping[str, Any]) -> FlightState:
    return FlightState(
        sim_time_s=float(source["sim_time_s"]),
        position_world_m=_array(source, "position_world_m", 3),
        velocity_world_m_s=_array(source, "velocity_world_m_s", 3),
        q_body_to_world_wxyz=_array(source, "q_body_to_world_wxyz", 4),
        body_rate_rad_s=_array(source, "body_rate_rad_s", 3),
        motor_thrust_n=_array(source, "motor_thrust_n", 4),
    )


def _disturbance(source: Mapping[str, Any]) -> Disturbance:
    return Disturbance(
        force_world_n=_array(source, "force_world_n", 3),
        torque_body_nm=_array(source, "torque_body_nm", 3),
        wind_world_m_s=_array(source, "wind_world_m_s", 3),
        motor_efficiency=_array(source, "motor_efficiency", 4),
    )


def _command(source: Mapping[str, Any]) -> PilotCommand:
    return PilotCommand(
        collective_thrust_n=float(source["collective_thrust_n"]),
        body_rate_rad_s=_array(source, "body_rate_rad_s", 3),
    )


def _assert_state_close(
    actual: FlightState,
    expected: FlightState,
    *,
    state_abs: float,
    quaternion_abs: float,
) -> None:
    assert actual.sim_time_s == pytest.approx(expected.sim_time_s, abs=state_abs)
    np.testing.assert_allclose(
        actual.position_world_m, expected.position_world_m, rtol=0.0, atol=state_abs
    )
    np.testing.assert_allclose(
        actual.velocity_world_m_s, expected.velocity_world_m_s, rtol=0.0, atol=state_abs
    )
    expected_quaternion = expected.q_body_to_world_wxyz
    if float(np.dot(actual.q_body_to_world_wxyz, expected_quaternion)) < 0.0:
        expected_quaternion = -expected_quaternion
    np.testing.assert_allclose(
        actual.q_body_to_world_wxyz, expected_quaternion, rtol=0.0, atol=quaternion_abs
    )
    np.testing.assert_allclose(
        actual.body_rate_rad_s, expected.body_rate_rad_s, rtol=0.0, atol=state_abs
    )
    np.testing.assert_allclose(
        actual.motor_thrust_n, expected.motor_thrust_n, rtol=0.0, atol=state_abs
    )


def test_python_fixed_step_runtime_matches_shared_6dof_ctbr_fixture() -> None:
    fixture = _fixture()
    assert fixture["name"] == "python-rust-6dof-ctbr-disturbance-v1"
    config_path = REPOSITORY_ROOT / str(fixture["vehicle_config_path"])
    config = VehicleConfig.from_toml(config_path)
    runtime = FixedStepRuntime(
        config,
        dt=float(fixture["dt_s"]),
        state=_state(cast(Mapping[str, Any], fixture["initial_state"])),
    )
    disturbance = _disturbance(cast(Mapping[str, Any], fixture["disturbance"]))
    commands = cast(list[Mapping[str, Any]], fixture["commands"])
    expected_steps = cast(list[Mapping[str, Any]], fixture["expected_steps"])
    tolerances = cast(Mapping[str, Any], fixture["tolerances"])

    assert len(commands) == len(expected_steps) == 6
    for command, expected in zip(commands, expected_steps, strict=True):
        actual_state, _, _ = runtime.step(_command(command), disturbance)
        _assert_state_close(
            actual_state,
            _state(expected),
            state_abs=float(tolerances["state_abs"]),
            quaternion_abs=float(tolerances["quaternion_abs"]),
        )
