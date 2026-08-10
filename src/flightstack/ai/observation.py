"""Local, sign-invariant state observations for FlightStack racing policies."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from flightstack.ai.config import ObservationConfig
from flightstack.math.quaternion import rotate_inverse
from flightstack.race import Gate, RaceState
from flightstack.sim.vehicle import FlightState, VehicleConfig

Vector = NDArray[np.float64]

OBSERVATION_SCHEMA_VERSION = "flightstack-race-observation-v1"
OBSERVATION_DIMENSION = 27
OBSERVATION_COMPONENTS: Final[tuple[str, ...]] = (
    "body_velocity_x",
    "body_velocity_y",
    "body_velocity_z",
    "body_rate_x",
    "body_rate_y",
    "body_rate_z",
    "next_gate_vector_x",
    "next_gate_vector_y",
    "next_gate_vector_z",
    "next_gate_normal_x",
    "next_gate_normal_y",
    "next_gate_normal_z",
    "following_gate_vector_x",
    "following_gate_vector_y",
    "following_gate_vector_z",
    "following_gate_normal_x",
    "following_gate_normal_y",
    "following_gate_normal_z",
    "world_up_in_body_x",
    "world_up_in_body_y",
    "world_up_in_body_z",
    "previous_action_thrust",
    "previous_action_roll_rate",
    "previous_action_pitch_rate",
    "previous_action_yaw_rate",
    "next_gate_distance",
    "world_speed",
)


def _clip_unit(value: Vector) -> Vector:
    return np.asarray(np.clip(value, -1.0, 1.0), dtype=np.float64)


def _following_gate(race: RaceState) -> Gate | None:
    """Return the order entry after the current target, wrapping a lap."""
    if race.next_gate is None:
        return None
    following_order = (race.next_gate_order_index + 1) % race.track.gate_passes_per_lap
    return race.track.gate_for_order_index(following_order)


def distance_to_next_gate(state: FlightState, race: RaceState) -> float:
    """Return the physical distance to the active ordered gate, or zero when done."""
    gate = race.next_gate
    if gate is None:
        return 0.0
    return float(np.linalg.norm(gate.center_world_m - state.position_world_m))


def build_observation(
    state: FlightState,
    race: RaceState,
    vehicle: VehicleConfig,
    config: ObservationConfig,
    previous_action: Vector | None = None,
) -> NDArray[np.float32]:
    """Build the versioned 27-float observation documented in TOML.

    All geometry enters in body coordinates.  That makes the policy invariant
    to a global course translation/yaw and avoids passing an ambiguous raw
    quaternion (`q` and `-q` encode the same attitude).
    """
    if config.schema_version != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(
            "unsupported observation schema "
            f"{config.schema_version!r}; expected {OBSERVATION_SCHEMA_VERSION!r}"
        )
    if not isinstance(race, RaceState):
        raise TypeError("race must be a RaceState")
    if not isinstance(vehicle, VehicleConfig):
        raise TypeError("vehicle must be a VehicleConfig")
    q = state.q_body_to_world_wxyz
    body_velocity = _clip_unit(
        np.asarray(
            rotate_inverse(q, state.velocity_world_m_s) / config.body_velocity_scale_m_s,
            dtype=np.float64,
        )
    )
    body_rate = _clip_unit(
        np.asarray(state.body_rate_rad_s / vehicle.max_body_rate_rad_s, dtype=np.float64)
    )
    next_gate = race.next_gate
    following = _following_gate(race)
    if next_gate is None:
        next_vector = np.zeros(3, dtype=np.float64)
        next_normal = np.zeros(3, dtype=np.float64)
    else:
        next_vector = _clip_unit(
            np.asarray(
                rotate_inverse(q, next_gate.center_world_m - state.position_world_m)
                / config.gate_vector_scale_m,
                dtype=np.float64,
            )
        )
        next_normal = _clip_unit(np.asarray(rotate_inverse(q, next_gate.normal_world)))
    if following is None:
        following_vector = np.zeros(3, dtype=np.float64)
        following_normal = np.zeros(3, dtype=np.float64)
    else:
        following_vector = _clip_unit(
            np.asarray(
                rotate_inverse(q, following.center_world_m - state.position_world_m)
                / config.gate_vector_scale_m,
                dtype=np.float64,
            )
        )
        following_normal = _clip_unit(np.asarray(rotate_inverse(q, following.normal_world)))
    body_up = _clip_unit(np.asarray(rotate_inverse(q, [0.0, 0.0, 1.0]), dtype=np.float64))
    prior = np.zeros(4, dtype=np.float64) if previous_action is None else previous_action
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != (4,) or not np.all(np.isfinite(prior)):
        raise ValueError("previous_action must be a finite vector with shape (4,)")
    distance = min(distance_to_next_gate(state, race) / config.distance_scale_m, 1.0)
    speed = min(float(np.linalg.norm(state.velocity_world_m_s)) / config.speed_scale_m_s, 1.0)
    observation = np.concatenate(
        (
            body_velocity,
            body_rate,
            next_vector,
            next_normal,
            following_vector,
            following_normal,
            body_up,
            _clip_unit(prior),
            np.array([distance, speed], dtype=np.float64),
        )
    )
    if observation.shape != (OBSERVATION_DIMENSION,):
        raise RuntimeError("internal observation layout does not match its schema")
    return np.asarray(observation, dtype=np.float32)
