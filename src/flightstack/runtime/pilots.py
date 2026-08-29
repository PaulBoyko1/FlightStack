"""Shared pilot contracts and responsive, frame-safe human input shaping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from flightstack.math.quaternion import (
    from_rotation_matrix,
    rotate,
    rotation_vector_error,
)
from flightstack.race import RaceState
from flightstack.sim.vehicle import FlightState, PilotCommand, VehicleConfig

Vector = NDArray[np.float64]


class PilotKind(StrEnum):
    """The comparable high-level pilots supported by FlightStack."""

    HUMAN = "human"
    CLASSICAL = "classical"
    LEARNED = "learned"


class RaceView(Protocol):
    """Minimal race information pilots may consume without owning race state."""

    @property
    def next_gate_index(self) -> int | None:
        """Return the next physical gate, or ``None`` while the run is idle."""


class Pilot(Protocol):
    """Every pilot emits the same CTBR command at the same control seam."""

    kind: PilotKind

    def reset(self, initial_state: FlightState) -> None: ...

    def command(self, state: FlightState, race: RaceState, dt: float) -> PilotCommand: ...


def deadzone_expo(value: float, *, deadzone: float, expo: float) -> float:
    """Normalize a signed stick with a continuous deadzone and cubic expo."""
    scalar = float(value)
    if not np.isfinite(scalar) or not np.isfinite(deadzone) or not np.isfinite(expo):
        raise ValueError("stick value, deadzone, and expo must be finite")
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    if not 0.0 <= expo <= 1.0:
        raise ValueError("expo must be in [0, 1]")
    clipped = float(np.clip(scalar, -1.0, 1.0))
    magnitude = abs(clipped)
    if magnitude <= deadzone:
        return 0.0
    normalized = (magnitude - deadzone) / (1.0 - deadzone)
    curved = (1.0 - expo) * normalized + expo * normalized**3
    return float(np.copysign(curved, clipped))


def hover_centered_collective(throttle: float, config: VehicleConfig) -> float:
    """Map a human throttle in ``[0, 1]`` with 0.5 at physical hover.

    This low-level mapping remains useful for direct/acro experiments.  The
    default interactive HumanPilot is intentionally game-style and instead
    closes velocity/attitude loops before emitting the same CTBR contract.
    """
    value = float(throttle)
    if not np.isfinite(value):
        raise ValueError("throttle must be finite")
    value = float(np.clip(value, 0.0, 1.0))
    total_max = 4.0 * config.motor_max_thrust_n
    hover = config.hover_thrust_n
    if value <= 0.5:
        return 2.0 * value * hover
    return hover + 2.0 * (value - 0.5) * (total_max - hover)


@dataclass(frozen=True)
class ManualInput:
    """Normalized game-style human controls captured by the browser.

    ``pitch`` is forward/back intent, ``roll`` is right/left intent, ``yaw``
    is turn intent, and ``throttle`` is centered vertical-speed intent once the
    craft is airborne.  The names remain protocol-compatible with the original
    transmitter-style client while the default HumanPilot provides stabilized
    movement.
    """

    throttle: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("throttle", self.throttle),
            ("roll", self.roll),
            ("pitch", self.pitch),
            ("yaw", self.yaw),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ManualControlConfig:
    deadzone: float = 0.07
    expo: float = 0.18
    yaw_rate_scale: float = 0.90
    max_vertical_speed_m_s: float = 2.2
    vertical_speed_kp: float = 3.2
    max_vertical_accel_m_s2: float = 4.0
    max_horizontal_speed_m_s: float = 22.0
    horizontal_speed_kp: float = 3.2
    max_horizontal_accel_m_s2: float = 14.0
    attitude_kp: float = 5.0
    max_stabilized_rate_rad_s: float = 4.0
    takeoff_max_speed_m_s: float = 1.4
    takeoff_altitude_kp: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        if not 0.0 <= self.expo <= 1.0:
            raise ValueError("expo must be in [0, 1]")
        for name, value in (
            ("yaw_rate_scale", self.yaw_rate_scale),
            ("max_vertical_speed_m_s", self.max_vertical_speed_m_s),
            ("vertical_speed_kp", self.vertical_speed_kp),
            ("max_vertical_accel_m_s2", self.max_vertical_accel_m_s2),
            ("max_horizontal_speed_m_s", self.max_horizontal_speed_m_s),
            ("horizontal_speed_kp", self.horizontal_speed_kp),
            ("max_horizontal_accel_m_s2", self.max_horizontal_accel_m_s2),
            ("attitude_kp", self.attitude_kp),
            ("max_stabilized_rate_rad_s", self.max_stabilized_rate_rad_s),
            ("takeoff_max_speed_m_s", self.takeoff_max_speed_m_s),
            ("takeoff_altitude_kp", self.takeoff_altitude_kp),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")


class HumanPilot:
    """Stabilized keyboard/gamepad pilot sharing the normal CTBR control seam."""

    kind = PilotKind.HUMAN

    def __init__(
        self,
        vehicle: VehicleConfig,
        controls: ManualControlConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.controls = ManualControlConfig() if controls is None else controls
        self.input = ManualInput()

    def set_input(self, input_state: ManualInput) -> None:
        self.input = input_state

    def reset(self, initial_state: FlightState) -> None:
        del initial_state
        self.input = ManualInput()

    @staticmethod
    def _horizontal_unit(vector: Vector, fallback: Vector) -> Vector:
        horizontal = np.array([vector[0], vector[1], 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(horizontal))
        if norm < 1e-8:
            return fallback.copy()
        return horizontal / norm

    @staticmethod
    def _clip_norm(vector: Vector, limit: float) -> Vector:
        norm = float(np.linalg.norm(vector))
        if norm <= limit or norm < 1e-12:
            return vector
        return vector * (limit / norm)

    def _game_command(
        self,
        state: FlightState,
        *,
        forward_input: float,
        right_input: float,
        yaw_input: float,
        target_vertical_speed_m_s: float,
    ) -> PilotCommand:
        """Close world-velocity and attitude loops, then emit body rates + thrust."""
        q = state.q_body_to_world_wxyz
        forward_world = self._horizontal_unit(
            rotate(q, np.array([1.0, 0.0, 0.0], dtype=np.float64)),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        left_world = self._horizontal_unit(
            rotate(q, np.array([0.0, 1.0, 0.0], dtype=np.float64)),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )

        desired_horizontal_velocity = self.controls.max_horizontal_speed_m_s * (
            forward_input * forward_world - right_input * left_world
        )
        horizontal_error = desired_horizontal_velocity[:2] - state.velocity_world_m_s[:2]
        horizontal_accel = np.array(
            [
                horizontal_error[0] * self.controls.horizontal_speed_kp,
                horizontal_error[1] * self.controls.horizontal_speed_kp,
                0.0,
            ],
            dtype=np.float64,
        )
        horizontal_accel = self._clip_norm(
            horizontal_accel, self.controls.max_horizontal_accel_m_s2
        )

        vertical_error = target_vertical_speed_m_s - float(state.velocity_world_m_s[2])
        vertical_accel = float(
            np.clip(
                vertical_error * self.controls.vertical_speed_kp,
                -self.controls.max_vertical_accel_m_s2,
                self.controls.max_vertical_accel_m_s2,
            )
        )
        desired_specific_force = np.array(
            [
                horizontal_accel[0],
                horizontal_accel[1],
                self.vehicle.gravity_m_s2 + vertical_accel,
            ],
            dtype=np.float64,
        )
        force_norm = float(np.linalg.norm(desired_specific_force))
        if force_norm < 1e-9:
            desired_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            collective = 0.0
        else:
            desired_up = desired_specific_force / force_norm
            # Keep the vertical force target correct while attitude slews into
            # an aggressive horizontal acceleration. Applying the final vector
            # magnitude while the craft is still upright creates an unwanted
            # altitude pop; compensating against the current body-Z projection
            # lets horizontal authority build as the vehicle actually tilts.
            current_up = rotate(q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
            vertical_projection = max(float(current_up[2]), 0.25)
            desired_vertical_specific_force = max(
                0.0, self.vehicle.gravity_m_s2 + vertical_accel
            )
            collective = (
                self.vehicle.mass_kg
                * desired_vertical_specific_force
                / vertical_projection
            )

        # Preserve the current horizontal heading while tilting toward the
        # acceleration vector.  Q/E adds yaw rate separately below.
        desired_left = np.cross(desired_up, forward_world)
        desired_left_norm = float(np.linalg.norm(desired_left))
        if desired_left_norm < 1e-8:
            desired_left = left_world
        else:
            desired_left = desired_left / desired_left_norm
        desired_forward = np.cross(desired_left, desired_up)
        desired_forward = desired_forward / max(float(np.linalg.norm(desired_forward)), 1e-8)
        desired_rotation = np.column_stack((desired_forward, desired_left, desired_up))
        target_q = from_rotation_matrix(desired_rotation)
        attitude_error = rotation_vector_error(q, target_q)
        rates = attitude_error * self.controls.attitude_kp
        rates[2] += (
            yaw_input
            * self.vehicle.max_body_rate_rad_s[2]
            * self.controls.yaw_rate_scale
        )
        rate_limit = np.minimum(
            self.vehicle.max_body_rate_rad_s,
            np.full(3, self.controls.max_stabilized_rate_rad_s, dtype=np.float64),
        )
        rates = np.clip(rates, -rate_limit, rate_limit)
        collective = float(
            np.clip(collective, 0.0, 4.0 * self.vehicle.motor_max_thrust_n)
        )
        return PilotCommand(collective_thrust_n=collective, body_rate_rad_s=rates)

    def takeoff_command(self, state: FlightState, target_altitude_m: float) -> PilotCommand:
        """Automatically rise from the pad and settle at the requested hover altitude."""
        altitude_error = max(0.0, float(target_altitude_m) - float(state.position_world_m[2]))
        target_vz = min(
            self.controls.takeoff_max_speed_m_s,
            altitude_error * self.controls.takeoff_altitude_kp,
        )
        return self._game_command(
            state,
            forward_input=0.0,
            right_input=0.0,
            yaw_input=0.0,
            target_vertical_speed_m_s=target_vz,
        )

    def command(self, state: FlightState, race: RaceView, dt: float) -> PilotCommand:
        del race, dt
        forward = deadzone_expo(
            self.input.pitch, deadzone=self.controls.deadzone, expo=self.controls.expo
        )
        right = deadzone_expo(
            self.input.roll, deadzone=self.controls.deadzone, expo=self.controls.expo
        )
        yaw = deadzone_expo(
            self.input.yaw, deadzone=self.controls.deadzone, expo=self.controls.expo
        )
        throttle = float(np.clip(self.input.throttle, 0.0, 1.0))
        target_vz = (throttle - 0.5) * 2.0 * self.controls.max_vertical_speed_m_s
        return self._game_command(
            state,
            forward_input=forward,
            right_input=right,
            yaw_input=yaw,
            target_vertical_speed_m_s=target_vz,
        )
