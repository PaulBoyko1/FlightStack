"""Deterministic 6DOF reference vehicle for FlightStack.

The original :mod:`flightstack.sim.rigid_body` remains the focused rotational
plant used by the legacy attitude-controller tests.  This module composes the
same quaternion semantics into a full multirotor model with translation,
individual motors, mixer, drag, and reproducible disturbances.

Frames are deliberately boring and explicit:

* world is right-handed ENU-like, +Z up;
* body is FLU, +X forward, +Y left, +Z up;
* attitude is scalar-first ``q_body_to_world``;
* body rates and moments are expressed in body coordinates.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.control.pid import PIDTerms, VectorPID
from flightstack.math.quaternion import integrate_body_rate, normalize, rotate, rotate_inverse

Vector = NDArray[np.float64]
Matrix = NDArray[np.float64]


def _vector(value: ArrayLike, size: int, name: str) -> Vector:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return result


def _matrix(value: ArrayLike, shape: tuple[int, int], name: str) -> Matrix:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    return result


def _positive(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def default_vehicle_config_path() -> Path:
    """Return the tracked, single source of reference-vehicle parameters."""
    return Path(__file__).resolve().parents[3] / "config" / "vehicles" / "flightstack_5in.toml"


@dataclass(frozen=True)
class VehicleConfig:
    """Validated physical and control parameters loaded from one TOML file."""

    name: str
    version: str
    mass_kg: float
    inertia_kg_m2: Matrix
    motor_position_body_m: Matrix
    motor_spin_direction: Vector
    motor_min_thrust_n: float
    motor_max_thrust_n: float
    motor_tau_rise_s: float
    motor_tau_fall_s: float
    thrust_to_reaction_torque_m: float
    linear_drag_n_per_m_s: Vector
    angular_drag_nm_per_rad_s: Vector
    gravity_m_s2: float
    rate_kp: Vector
    rate_ki: Vector
    rate_kd: Vector
    rate_torque_limit_nm: Vector
    rate_integral_limit: Vector
    max_body_rate_rad_s: Vector

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("vehicle name and version must be nonempty")
        _positive(self.mass_kg, "mass_kg")
        inertia = _matrix(self.inertia_kg_m2, (3, 3), "inertia_kg_m2")
        if not np.allclose(inertia, inertia.T, rtol=1e-10, atol=1e-12):
            raise ValueError("inertia_kg_m2 must be symmetric")
        if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
            raise ValueError("inertia_kg_m2 must be positive definite")
        _matrix(self.motor_position_body_m, (4, 3), "motor_position_body_m")
        spins = _vector(self.motor_spin_direction, 4, "motor_spin_direction")
        if not np.all(np.isin(spins, (-1.0, 1.0))):
            raise ValueError("motor_spin_direction must contain only -1 or +1")
        minimum = _nonnegative(self.motor_min_thrust_n, "motor_min_thrust_n")
        maximum = _positive(self.motor_max_thrust_n, "motor_max_thrust_n")
        if maximum <= minimum:
            raise ValueError("motor_max_thrust_n must exceed motor_min_thrust_n")
        _positive(self.motor_tau_rise_s, "motor_tau_rise_s")
        _positive(self.motor_tau_fall_s, "motor_tau_fall_s")
        _nonnegative(self.thrust_to_reaction_torque_m, "thrust_to_reaction_torque_m")
        if np.any(_vector(self.linear_drag_n_per_m_s, 3, "linear_drag_n_per_m_s") < 0.0):
            raise ValueError("linear_drag_n_per_m_s must be nonnegative")
        if np.any(_vector(self.angular_drag_nm_per_rad_s, 3, "angular_drag_nm_per_rad_s") < 0.0):
            raise ValueError("angular_drag_nm_per_rad_s must be nonnegative")
        _positive(self.gravity_m_s2, "gravity_m_s2")
        for name, value in (
            ("rate_kp", self.rate_kp),
            ("rate_ki", self.rate_ki),
            ("rate_kd", self.rate_kd),
            ("rate_torque_limit_nm", self.rate_torque_limit_nm),
            ("rate_integral_limit", self.rate_integral_limit),
            ("max_body_rate_rad_s", self.max_body_rate_rad_s),
        ):
            vector = _vector(value, 3, name)
            if np.any(vector < 0.0):
                raise ValueError(f"{name} must be nonnegative")
        if np.any(self.rate_torque_limit_nm <= 0.0) or np.any(self.max_body_rate_rad_s <= 0.0):
            raise ValueError("control limits must be positive")

    @property
    def hover_thrust_n(self) -> float:
        return self.mass_kg * self.gravity_m_s2

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.to_mapping(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-safe canonical mapping used in run provenance."""
        return {
            "name": self.name,
            "version": self.version,
            "mass_kg": self.mass_kg,
            "inertia_kg_m2": self.inertia_kg_m2.tolist(),
            "motor_position_body_m": self.motor_position_body_m.tolist(),
            "motor_spin_direction": self.motor_spin_direction.tolist(),
            "motor_min_thrust_n": self.motor_min_thrust_n,
            "motor_max_thrust_n": self.motor_max_thrust_n,
            "motor_tau_rise_s": self.motor_tau_rise_s,
            "motor_tau_fall_s": self.motor_tau_fall_s,
            "thrust_to_reaction_torque_m": self.thrust_to_reaction_torque_m,
            "linear_drag_n_per_m_s": self.linear_drag_n_per_m_s.tolist(),
            "angular_drag_nm_per_rad_s": self.angular_drag_nm_per_rad_s.tolist(),
            "gravity_m_s2": self.gravity_m_s2,
            "control": {
                "rate_kp": self.rate_kp.tolist(),
                "rate_ki": self.rate_ki.tolist(),
                "rate_kd": self.rate_kd.tolist(),
                "rate_torque_limit_nm": self.rate_torque_limit_nm.tolist(),
                "rate_integral_limit": self.rate_integral_limit.tolist(),
                "max_body_rate_rad_s": self.max_body_rate_rad_s.tolist(),
            },
        }

    @classmethod
    def from_toml(cls, path: str | Path | None = None) -> VehicleConfig:
        source = default_vehicle_config_path() if path is None else Path(path)
        with source.open("rb") as handle:
            data: Mapping[str, Any] = tomllib.load(handle)
        control = data.get("control")
        if not isinstance(control, dict):
            raise ValueError("vehicle config must contain a [control] table")
        try:
            return cls(
                name=str(data["name"]),
                version=str(data["version"]),
                mass_kg=float(data["mass_kg"]),
                inertia_kg_m2=_matrix(data["inertia_kg_m2"], (3, 3), "inertia_kg_m2"),
                motor_position_body_m=_matrix(
                    data["motor_position_body_m"], (4, 3), "motor_position_body_m"
                ),
                motor_spin_direction=_vector(
                    data["motor_spin_direction"], 4, "motor_spin_direction"
                ),
                motor_min_thrust_n=float(data["motor_min_thrust_n"]),
                motor_max_thrust_n=float(data["motor_max_thrust_n"]),
                motor_tau_rise_s=float(data["motor_tau_rise_s"]),
                motor_tau_fall_s=float(data["motor_tau_fall_s"]),
                thrust_to_reaction_torque_m=float(data["thrust_to_reaction_torque_m"]),
                linear_drag_n_per_m_s=_vector(
                    data["linear_drag_n_per_m_s"], 3, "linear_drag_n_per_m_s"
                ),
                angular_drag_nm_per_rad_s=_vector(
                    data["angular_drag_nm_per_rad_s"], 3, "angular_drag_nm_per_rad_s"
                ),
                gravity_m_s2=float(data["gravity_m_s2"]),
                rate_kp=_vector(control["rate_kp"], 3, "control.rate_kp"),
                rate_ki=_vector(control["rate_ki"], 3, "control.rate_ki"),
                rate_kd=_vector(control["rate_kd"], 3, "control.rate_kd"),
                rate_torque_limit_nm=_vector(
                    control["rate_torque_limit_nm"], 3, "control.rate_torque_limit_nm"
                ),
                rate_integral_limit=_vector(
                    control["rate_integral_limit"], 3, "control.rate_integral_limit"
                ),
                max_body_rate_rad_s=_vector(
                    control["max_body_rate_rad_s"], 3, "control.max_body_rate_rad_s"
                ),
            )
        except KeyError as exc:
            raise ValueError(f"vehicle config is missing {exc.args[0]!r}") from exc


@dataclass
class FlightState:
    """Canonical state of one 6DOF FlightStack vehicle."""

    sim_time_s: float
    position_world_m: Vector
    velocity_world_m_s: Vector
    q_body_to_world_wxyz: Vector
    body_rate_rad_s: Vector
    motor_thrust_n: Vector

    def __post_init__(self) -> None:
        self.sim_time_s = _nonnegative(self.sim_time_s, "sim_time_s")
        self.position_world_m = _vector(self.position_world_m, 3, "position_world_m")
        self.velocity_world_m_s = _vector(self.velocity_world_m_s, 3, "velocity_world_m_s")
        self.q_body_to_world_wxyz = normalize(self.q_body_to_world_wxyz)
        self.body_rate_rad_s = _vector(self.body_rate_rad_s, 3, "body_rate_rad_s")
        thrust = _vector(self.motor_thrust_n, 4, "motor_thrust_n")
        if np.any(thrust < 0.0):
            raise ValueError("motor_thrust_n must be nonnegative")
        self.motor_thrust_n = thrust

    @classmethod
    def hovering(cls, config: VehicleConfig, *, altitude_m: float = 1.0) -> FlightState:
        return cls(
            sim_time_s=0.0,
            position_world_m=np.array([0.0, 0.0, altitude_m], dtype=np.float64),
            velocity_world_m_s=np.zeros(3, dtype=np.float64),
            q_body_to_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            body_rate_rad_s=np.zeros(3, dtype=np.float64),
            motor_thrust_n=np.full(4, config.hover_thrust_n / 4.0, dtype=np.float64),
        )

    def copy(self) -> FlightState:
        return FlightState(
            self.sim_time_s,
            self.position_world_m.copy(),
            self.velocity_world_m_s.copy(),
            self.q_body_to_world_wxyz.copy(),
            self.body_rate_rad_s.copy(),
            self.motor_thrust_n.copy(),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "sim_time_s": self.sim_time_s,
            "position_world_m": self.position_world_m.tolist(),
            "velocity_world_m_s": self.velocity_world_m_s.tolist(),
            "q_body_to_world_wxyz": self.q_body_to_world_wxyz.tolist(),
            "body_rate_rad_s": self.body_rate_rad_s.tolist(),
            "motor_thrust_n": self.motor_thrust_n.tolist(),
        }


@dataclass(frozen=True)
class PilotCommand:
    """Shared collective-thrust/body-rate (CTBR) command for every pilot."""

    collective_thrust_n: float
    body_rate_rad_s: Vector

    def __post_init__(self) -> None:
        _nonnegative(self.collective_thrust_n, "collective_thrust_n")
        _vector(self.body_rate_rad_s, 3, "body_rate_rad_s")

    @classmethod
    def hover(cls, config: VehicleConfig) -> PilotCommand:
        return cls(config.hover_thrust_n, np.zeros(3, dtype=np.float64))


@dataclass(frozen=True)
class Disturbance:
    """Deterministic, scenario-owned effects applied during a vehicle step."""

    force_world_n: Vector
    torque_body_nm: Vector
    wind_world_m_s: Vector
    motor_efficiency: Vector

    def __post_init__(self) -> None:
        _vector(self.force_world_n, 3, "force_world_n")
        _vector(self.torque_body_nm, 3, "torque_body_nm")
        _vector(self.wind_world_m_s, 3, "wind_world_m_s")
        efficiency = _vector(self.motor_efficiency, 4, "motor_efficiency")
        if np.any(efficiency < 0.0) or np.any(efficiency > 1.0):
            raise ValueError("motor_efficiency must be in [0, 1]")

    @classmethod
    def calm(cls) -> Disturbance:
        return cls(
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.ones(4, dtype=np.float64),
        )


@dataclass(frozen=True)
class MixerResult:
    motor_target_thrust_n: Vector
    achieved_collective_thrust_n: float
    achieved_torque_body_nm: Vector
    saturated: bool


class QuadMixer:
    """Explicit CTBR torque allocation for the configured four-rotor layout."""

    def __init__(self, config: VehicleConfig) -> None:
        self.config = config
        positions = config.motor_position_body_m
        # Rows are collective, roll (+X), pitch (+Y), yaw (+Z).  A motor thrust
        # force is [0, 0, T], so r x F = [r_y*T, -r_x*T, 0].
        self.allocation: Matrix = np.vstack(
            (
                np.ones(4, dtype=np.float64),
                positions[:, 1],
                -positions[:, 0],
                config.motor_spin_direction * config.thrust_to_reaction_torque_m,
            )
        )
        if abs(float(np.linalg.det(self.allocation))) < 1e-12:
            raise ValueError("motor geometry/spin directions do not form a full-rank mixer")
        self._allocation_inverse: Matrix = np.asarray(
            np.linalg.inv(self.allocation), dtype=np.float64
        )

    def mix(self, collective_thrust_n: float, torque_body_nm: ArrayLike) -> MixerResult:
        collective = _nonnegative(collective_thrust_n, "collective_thrust_n")
        torque = _vector(torque_body_nm, 3, "torque_body_nm")
        requested = np.concatenate(([collective], torque))
        unconstrained = self._allocation_inverse @ requested
        clamped = np.clip(
            unconstrained,
            self.config.motor_min_thrust_n,
            self.config.motor_max_thrust_n,
        )
        achieved = self.allocation @ clamped
        return MixerResult(
            motor_target_thrust_n=np.asarray(clamped, dtype=np.float64),
            achieved_collective_thrust_n=float(achieved[0]),
            achieved_torque_body_nm=np.asarray(achieved[1:], dtype=np.float64),
            saturated=not bool(np.allclose(unconstrained, clamped, rtol=1e-12, atol=1e-12)),
        )


class BodyRateController:
    """The Python reference implementation of the CTBR rate-control seam.

    It intentionally delegates to the existing derivative-on-measurement,
    conditional-anti-windup :class:`VectorPID`, keeping controller behavior
    comparable with the original FlightStack laboratory and future C++ bridge.
    """

    def __init__(self, config: VehicleConfig) -> None:
        self.config = config
        self.pid = VectorPID(
            config.rate_kp,
            config.rate_ki,
            config.rate_kd,
            output_limit=config.rate_torque_limit_nm,
            integral_limit=config.rate_integral_limit,
            derivative_cutoff_hz=35.0,
        )

    def reset(self) -> None:
        self.pid.reset()

    def update(self, command: PilotCommand, body_rate_rad_s: ArrayLike, dt: float) -> PIDTerms:
        target = np.clip(
            _vector(command.body_rate_rad_s, 3, "command.body_rate_rad_s"),
            -self.config.max_body_rate_rad_s,
            self.config.max_body_rate_rad_s,
        )
        return self.pid.update(target, body_rate_rad_s, dt)


class Multirotor:
    """Transparent deterministic 6DOF multirotor reference plant."""

    def __init__(self, config: VehicleConfig, state: FlightState | None = None) -> None:
        self.config = config
        self.state = FlightState.hovering(config) if state is None else state.copy()
        if np.any(self.state.motor_thrust_n > config.motor_max_thrust_n):
            raise ValueError("initial motor thrust exceeds motor_max_thrust_n")
        self.mixer = QuadMixer(config)

    def reset(self, state: FlightState | None = None) -> FlightState:
        self.state = FlightState.hovering(self.config) if state is None else state.copy()
        return self.state.copy()

    def motor_force_and_moment(self) -> tuple[Vector, Vector]:
        thrust = self.state.motor_thrust_n
        force_body = np.array([0.0, 0.0, np.sum(thrust)], dtype=np.float64)
        arms = np.cross(
            self.config.motor_position_body_m,
            np.column_stack((np.zeros(4), np.zeros(4), thrust)),
        ).sum(axis=0)
        reaction = np.array(
            [
                0.0,
                0.0,
                np.sum(
                    self.config.motor_spin_direction
                    * self.config.thrust_to_reaction_torque_m
                    * thrust
                ),
            ],
            dtype=np.float64,
        )
        return force_body, np.asarray(arms + reaction, dtype=np.float64)

    def _update_motors(
        self,
        motor_target_thrust_n: ArrayLike,
        disturbance: Disturbance,
        dt: float,
    ) -> None:
        target = _vector(motor_target_thrust_n, 4, "motor_target_thrust_n")
        target = np.clip(
            target,
            self.config.motor_min_thrust_n,
            self.config.motor_max_thrust_n,
        )
        effective_target = target * disturbance.motor_efficiency
        current = self.state.motor_thrust_n
        tau = np.where(
            effective_target > current,
            self.config.motor_tau_rise_s,
            self.config.motor_tau_fall_s,
        )
        alpha = 1.0 - np.exp(-dt / tau)
        self.state.motor_thrust_n = np.clip(
            current + alpha * (effective_target - current),
            self.config.motor_min_thrust_n,
            self.config.motor_max_thrust_n,
        )

    def step_motor_targets(
        self,
        motor_target_thrust_n: ArrayLike,
        dt: float,
        disturbance: Disturbance | None = None,
    ) -> FlightState:
        """Advance one exact-motor/semi-implicit 6DOF integration step."""
        step_s = _positive(dt, "dt")
        effect = Disturbance.calm() if disturbance is None else disturbance
        self._update_motors(motor_target_thrust_n, effect, step_s)

        force_body, moment_from_motors = self.motor_force_and_moment()
        force_thrust_world = rotate(self.state.q_body_to_world_wxyz, force_body)
        relative_velocity_body = rotate_inverse(
            self.state.q_body_to_world_wxyz,
            self.state.velocity_world_m_s - effect.wind_world_m_s,
        )
        drag_body = -self.config.linear_drag_n_per_m_s * relative_velocity_body
        drag_world = rotate(self.state.q_body_to_world_wxyz, drag_body)
        gravity_world = np.array([0.0, 0.0, -self.config.mass_kg * self.config.gravity_m_s2])
        acceleration_world = (
            force_thrust_world + gravity_world + drag_world + effect.force_world_n
        ) / self.config.mass_kg
        self.state.velocity_world_m_s = self.state.velocity_world_m_s + acceleration_world * step_s
        self.state.position_world_m = (
            self.state.position_world_m + self.state.velocity_world_m_s * step_s
        )

        omega = self.state.body_rate_rad_s
        gyroscopic = np.cross(omega, self.config.inertia_kg_m2 @ omega)
        drag_moment = -self.config.angular_drag_nm_per_rad_s * omega
        angular_acceleration = np.linalg.solve(
            self.config.inertia_kg_m2,
            moment_from_motors + drag_moment + effect.torque_body_nm - gyroscopic,
        )
        self.state.body_rate_rad_s = omega + angular_acceleration * step_s
        self.state.q_body_to_world_wxyz = integrate_body_rate(
            self.state.q_body_to_world_wxyz,
            self.state.body_rate_rad_s,
            step_s,
        )
        self.state.sim_time_s += step_s
        return self.state.copy()

    def step_command(
        self,
        command: PilotCommand,
        controller: BodyRateController,
        dt: float,
        disturbance: Disturbance | None = None,
    ) -> tuple[FlightState, MixerResult, PIDTerms]:
        """Run the shared CTBR -> PID -> mixer -> motors -> physics pipeline."""
        terms = controller.update(command, self.state.body_rate_rad_s, dt)
        mixed = self.mixer.mix(command.collective_thrust_n, terms.output)
        state = self.step_motor_targets(mixed.motor_target_thrust_n, dt, disturbance)
        return state, mixed, terms


class FixedStepRuntime:
    """Fixed-rate owner of a vehicle/control chain, independent of wall clock."""

    def __init__(
        self,
        config: VehicleConfig,
        *,
        dt: float = 0.002,
        state: FlightState | None = None,
    ) -> None:
        self.dt = _positive(dt, "dt")
        self.vehicle = Multirotor(config, state)
        self.controller = BodyRateController(config)

    @property
    def state(self) -> FlightState:
        return self.vehicle.state.copy()

    def reset(self, state: FlightState | None = None) -> FlightState:
        self.controller.reset()
        return self.vehicle.reset(state)

    def step(
        self,
        command: PilotCommand,
        disturbance: Disturbance | None = None,
    ) -> tuple[FlightState, MixerResult, PIDTerms]:
        return self.vehicle.step_command(command, self.controller, self.dt, disturbance)
