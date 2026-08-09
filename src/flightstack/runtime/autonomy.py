"""A small deterministic racing baseline using FlightStack's shared CTBR seam."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from flightstack.math.quaternion import (
    from_rotation_matrix,
    rotate,
    rotation_vector_error,
)
from flightstack.race import RaceState
from flightstack.runtime.pilots import PilotKind
from flightstack.sim.vehicle import FlightState, PilotCommand, VehicleConfig

Vector = NDArray[np.float64]


def _unit(value: Vector, fallback: Vector) -> Vector:
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        return fallback.copy()
    return np.asarray(value / norm, dtype=np.float64)


@dataclass(frozen=True)
class ClassicalPilotConfig:
    """Conservative guidance gains for the reference technical-eight course."""

    cruise_speed_m_s: float = 4.5
    position_gain_s2: float = 1.4
    velocity_gain_s: float = 2.1
    max_acceleration_m_s2: float = 7.0
    max_tilt_rad: float = float(np.deg2rad(38.0))
    attitude_gain_s: float = 3.2
    target_altitude_m: float = 1.5

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_tilt_rad >= np.pi / 2.0:
            raise ValueError("max_tilt_rad must be below pi/2")


class ClassicalRacePilot:
    """Position/velocity guidance -> thrust vector -> attitude -> CTBR command.

    It is intentionally a transparent benchmark, not a full trajectory
    optimizer.  The important property is that its collective and body-rate
    requests traverse the exact same rate controller, mixer, motor dynamics,
    and vehicle as manual and learned pilots.
    """

    kind = PilotKind.CLASSICAL

    def __init__(
        self,
        vehicle: VehicleConfig,
        config: ClassicalPilotConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.config = ClassicalPilotConfig() if config is None else config

    def reset(self, initial_state: FlightState) -> None:
        del initial_state

    def command(self, state: FlightState, race: RaceState, dt: float) -> PilotCommand:
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        target = self._target_position(state, race)
        offset = target - state.position_world_m
        distance = float(np.linalg.norm(offset))
        direction = _unit(offset, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        desired_speed = min(self.config.cruise_speed_m_s, 0.75 + distance * 1.15)
        desired_velocity = direction * desired_speed
        acceleration = (
            self.config.position_gain_s2 * offset
            + self.config.velocity_gain_s * (desired_velocity - state.velocity_world_m_s)
        )
        acceleration_norm = float(np.linalg.norm(acceleration))
        if acceleration_norm > self.config.max_acceleration_m_s2:
            acceleration *= self.config.max_acceleration_m_s2 / acceleration_norm

        requested_force = self.vehicle.mass_kg * (
            acceleration + np.array([0.0, 0.0, self.vehicle.gravity_m_s2], dtype=np.float64)
        )
        requested_force = self._limit_tilt(requested_force)
        collective = float(
            np.clip(
                np.linalg.norm(requested_force),
                self.vehicle.motor_min_thrust_n * 4.0,
                self.vehicle.motor_max_thrust_n * 4.0,
            )
        )
        desired_q = self._attitude_from_thrust(requested_force, direction, state)
        desired_rates = self.config.attitude_gain_s * rotation_vector_error(
            state.q_body_to_world_wxyz, desired_q
        )
        desired_rates = np.clip(
            desired_rates,
            -self.vehicle.max_body_rate_rad_s,
            self.vehicle.max_body_rate_rad_s,
        )
        return PilotCommand(collective, np.asarray(desired_rates, dtype=np.float64))

    def _target_position(self, state: FlightState, race: RaceState) -> Vector:
        gate = race.next_gate
        if gate is None:
            return np.array(
                [
                    state.position_world_m[0],
                    state.position_world_m[1],
                    self.config.target_altitude_m,
                ],
                dtype=np.float64,
            )
        return np.asarray(gate.center_world_m, dtype=np.float64)

    def _limit_tilt(self, force_world: Vector) -> Vector:
        minimum_vertical = self.vehicle.mass_kg * self.vehicle.gravity_m_s2 * 0.2
        vertical = max(float(force_world[2]), minimum_vertical)
        lateral = force_world[:2].copy()
        lateral_norm = float(np.linalg.norm(lateral))
        lateral_limit = vertical * float(np.tan(self.config.max_tilt_rad))
        if lateral_norm > lateral_limit:
            lateral *= lateral_limit / lateral_norm
        return np.array([lateral[0], lateral[1], vertical], dtype=np.float64)

    def _attitude_from_thrust(
        self,
        force_world: Vector,
        travel_direction: Vector,
        state: FlightState,
    ) -> Vector:
        body_z_world = _unit(force_world, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        heading = travel_direction.copy()
        heading[2] = 0.0
        if float(np.linalg.norm(heading)) < 1e-8:
            heading = rotate(state.q_body_to_world_wxyz, [1.0, 0.0, 0.0])
            heading[2] = 0.0
        heading = _unit(heading, np.array([1.0, 0.0, 0.0], dtype=np.float64))
        body_y_world = np.cross(body_z_world, heading)
        if float(np.linalg.norm(body_y_world)) < 1e-8:
            body_y_world = np.cross(body_z_world, np.array([0.0, 1.0, 0.0]))
        body_y_world = _unit(body_y_world, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        body_x_world = _unit(
            np.cross(body_y_world, body_z_world), np.array([1.0, 0.0, 0.0], dtype=np.float64)
        )
        rotation = np.column_stack((body_x_world, body_y_world, body_z_world))
        return from_rotation_matrix(rotation)
