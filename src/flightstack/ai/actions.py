"""Versioned normalized-action mapping onto FlightStack's CTBR seam."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.sim.vehicle import PilotCommand, VehicleConfig

ACTION_SCHEMA_VERSION = "flightstack-ctbr-action-v1"
ACTION_DIMENSION = 4
Vector = NDArray[np.float64]


def normalized_action(action: ArrayLike) -> Vector:
    """Validate and clip one policy action in ``[thrust, roll, pitch, yaw]``.

    The explicit clip makes a policy's occasional numerical overshoot safe at
    the only learned-to-physical command boundary.  It never changes the
    caller's array in place.
    """
    result = np.asarray(action, dtype=np.float64)
    if result.shape != (ACTION_DIMENSION,) or not np.all(np.isfinite(result)):
        raise ValueError("normalized action must be a finite vector with shape (4,)")
    return np.asarray(np.clip(result, -1.0, 1.0), dtype=np.float64)


def action_to_command(action: ArrayLike, vehicle: VehicleConfig) -> PilotCommand:
    """Map normalized policy action to hover-centred collective + body rates.

    ``action[0] == 0`` is exactly physical hover.  Positive and negative
    thrust authority use the true *total vehicle* limits, while angular terms
    scale against the shared low-level rate-controller limits.  This is the
    only default learned-pilot actuator mapping in FlightStack v1.
    """
    if not isinstance(vehicle, VehicleConfig):
        raise TypeError("vehicle must be a VehicleConfig")
    clipped = normalized_action(action)
    total_min = vehicle.motor_min_thrust_n * 4.0
    total_max = vehicle.motor_max_thrust_n * 4.0
    hover = vehicle.hover_thrust_n
    thrust_action = float(clipped[0])
    if thrust_action >= 0.0:
        collective = hover + thrust_action * (total_max - hover)
    else:
        collective = hover + thrust_action * (hover - total_min)
    collective = float(np.clip(collective, total_min, total_max))
    rates = np.asarray(clipped[1:] * vehicle.max_body_rate_rad_s, dtype=np.float64)
    return PilotCommand(collective_thrust_n=collective, body_rate_rad_s=rates)
