"""Gate geometry and small collision queries for FlightStack races.

The reference simulator uses an ENU-like world frame (+Z up).  A gate's
``normal_world`` points in its *positive* passing direction: a forward pass
travels from the negative side of the gate plane to the positive side.  Its
``right_world`` and ``up_world`` complete a right-handed gate-local basis:
``normal x right == up``.

Gate passing deliberately uses a swept segment, rather than a distance test,
so a fast vehicle cannot tunnel through an angled gate between physics ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector: TypeAlias = NDArray[np.float64]

_BASIS_TOLERANCE = 1e-8
_INTERSECTION_TOLERANCE = 1e-12


def _vector3(value: ArrayLike, name: str) -> Vector:
    """Return a finite, immutable world-space vector."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    result = vector.copy()
    result.setflags(write=False)
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    """Validate a nonnegative finite scalar."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _finite_positive(value: float, name: str) -> float:
    """Validate a positive finite scalar."""
    result = _finite_nonnegative(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _passing_direction(value: int, name: str = "direction") -> int:
    """Validate a signed gate passing direction."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be either -1 or +1")
    try:
        direction = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be either -1 or +1") from exc
    if direction not in (-1, 1):
        raise ValueError(f"{name} must be either -1 or +1")
    return direction


@dataclass(frozen=True)
class Gate:
    """A rectangular race-gate aperture represented in a world-space basis.

    ``half_width_m`` measures along ``right_world`` and ``half_height_m``
    measures along ``up_world``.  The optional frame dimensions are used only
    by :func:`gate_frame_collision`; they do not shrink the pass aperture.
    """

    center_world_m: Vector
    normal_world: Vector
    right_world: Vector
    up_world: Vector
    half_width_m: float
    half_height_m: float
    gate_id: str = ""
    frame_thickness_m: float = 0.08
    frame_depth_m: float = 0.08

    def __post_init__(self) -> None:
        center = _vector3(self.center_world_m, "center_world_m")
        normal = _vector3(self.normal_world, "normal_world")
        right = _vector3(self.right_world, "right_world")
        up = _vector3(self.up_world, "up_world")

        for name, vector in (("normal_world", normal), ("right_world", right), ("up_world", up)):
            length = float(np.linalg.norm(vector))
            if length <= _INTERSECTION_TOLERANCE:
                raise ValueError(f"{name} must have nonzero length")

        normal = _immutable_unit(normal)
        right = _immutable_unit(right)
        up = _immutable_unit(up)
        if (
            abs(float(np.dot(normal, right))) > _BASIS_TOLERANCE
            or abs(float(np.dot(normal, up))) > _BASIS_TOLERANCE
            or abs(float(np.dot(right, up))) > _BASIS_TOLERANCE
        ):
            raise ValueError("gate basis vectors must be orthogonal")
        if not np.isclose(
            float(np.dot(np.cross(normal, right), up)),
            1.0,
            rtol=0.0,
            atol=_BASIS_TOLERANCE,
        ):
            raise ValueError("gate basis must be right-handed: normal x right == up")

        if not isinstance(self.gate_id, str):
            raise ValueError("gate_id must be a string")

        object.__setattr__(self, "center_world_m", center)
        object.__setattr__(self, "normal_world", normal)
        object.__setattr__(self, "right_world", right)
        object.__setattr__(self, "up_world", up)
        object.__setattr__(
            self,
            "half_width_m",
            _finite_positive(self.half_width_m, "half_width_m"),
        )
        object.__setattr__(
            self,
            "half_height_m",
            _finite_positive(self.half_height_m, "half_height_m"),
        )
        object.__setattr__(
            self,
            "frame_thickness_m",
            _finite_nonnegative(self.frame_thickness_m, "frame_thickness_m"),
        )
        object.__setattr__(
            self,
            "frame_depth_m",
            _finite_nonnegative(self.frame_depth_m, "frame_depth_m"),
        )

    @property
    def id(self) -> str:
        """Short alias convenient for JSON/UI consumers."""
        return self.gate_id

    @property
    def width_m(self) -> float:
        """Full clear aperture width."""
        return 2.0 * self.half_width_m

    @property
    def height_m(self) -> float:
        """Full clear aperture height."""
        return 2.0 * self.half_height_m

    def local_coordinates(self, position_world_m: ArrayLike) -> Vector:
        """Transform a world point into ``[normal, right, up]`` gate coordinates."""
        relative = _vector3(position_world_m, "position_world_m") - self.center_world_m
        result = np.array(
            [
                np.dot(relative, self.normal_world),
                np.dot(relative, self.right_world),
                np.dot(relative, self.up_world),
            ],
            dtype=np.float64,
        )
        result.setflags(write=False)
        return result

    def world_coordinates(self, local_position_m: ArrayLike) -> Vector:
        """Transform a ``[normal, right, up]`` gate point into world coordinates."""
        local = _vector3(local_position_m, "local_position_m")
        result = (
            self.center_world_m
            + local[0] * self.normal_world
            + local[1] * self.right_world
            + local[2] * self.up_world
        )
        result = np.asarray(result, dtype=np.float64)
        result.setflags(write=False)
        return result

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-safe representation of the gate."""
        return {
            "id": self.gate_id,
            "center_world_m": self.center_world_m.tolist(),
            "normal_world": self.normal_world.tolist(),
            "right_world": self.right_world.tolist(),
            "up_world": self.up_world.tolist(),
            "half_width_m": self.half_width_m,
            "half_height_m": self.half_height_m,
            "frame_thickness_m": self.frame_thickness_m,
            "frame_depth_m": self.frame_depth_m,
        }


def _immutable_unit(vector: Vector) -> Vector:
    """Normalize a nonzero vector and make the result immutable."""
    result = np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class GateCrossing:
    """The exact result of intersecting a motion segment with a gate plane."""

    gate_id: str
    position_world_m: Vector
    segment_fraction: float
    direction: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_world_m",
            _vector3(self.position_world_m, "position_world_m"),
        )
        fraction = float(self.segment_fraction)
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("segment_fraction must be finite and in [0, 1]")
        object.__setattr__(self, "segment_fraction", fraction)
        object.__setattr__(self, "direction", _passing_direction(self.direction))


def swept_gate_intersection(
    previous_world_m: ArrayLike,
    current_world_m: ArrayLike,
    gate: Gate,
    *,
    direction: int = 1,
) -> GateCrossing | None:
    """Return a valid directed swept crossing of ``gate``, if one occurred.

    A forward pass (``direction=+1``) crosses from the negative to positive
    normal side.  A reverse pass crosses in the opposite direction.  The
    aperture test is performed at the interpolated plane intersection, which
    remains correct for high-speed and non-axis-aligned motion.
    """
    if not isinstance(gate, Gate):
        raise TypeError("gate must be a Gate")
    previous = _vector3(previous_world_m, "previous_world_m")
    current = _vector3(current_world_m, "current_world_m")
    pass_direction = _passing_direction(direction)

    previous_local = gate.local_coordinates(previous)
    current_local = gate.local_coordinates(current)
    previous_plane_distance = float(previous_local[0])
    current_plane_distance = float(current_local[0])

    if pass_direction > 0:
        crosses_in_direction = previous_plane_distance < 0.0 and current_plane_distance >= 0.0
    else:
        crosses_in_direction = previous_plane_distance > 0.0 and current_plane_distance <= 0.0
    if not crosses_in_direction:
        return None

    denominator = previous_plane_distance - current_plane_distance
    if abs(denominator) <= _INTERSECTION_TOLERANCE:
        return None
    fraction = previous_plane_distance / denominator
    if not 0.0 <= fraction <= 1.0:
        return None

    intersection_local = previous_local + fraction * (current_local - previous_local)
    in_aperture = (
        abs(float(intersection_local[1])) <= gate.half_width_m + _INTERSECTION_TOLERANCE
        and abs(float(intersection_local[2])) <= gate.half_height_m + _INTERSECTION_TOLERANCE
    )
    if not in_aperture:
        return None

    position = previous + fraction * (current - previous)
    return GateCrossing(
        gate_id=gate.gate_id,
        position_world_m=position,
        segment_fraction=fraction,
        direction=pass_direction,
    )


def swept_gate_crossing(
    previous_world_m: ArrayLike,
    current_world_m: ArrayLike,
    gate: Gate,
    *,
    direction: int = 1,
) -> Vector | None:
    """Return the crossing position for the source-pack-compatible simple API.

    Use :func:`swept_gate_intersection` when segment timing or direction metadata
    is needed.  This helper intentionally mirrors the compact source-pack seed
    which returns only the crossing point.
    """
    crossing = swept_gate_intersection(
        previous_world_m,
        current_world_m,
        gate,
        direction=direction,
    )
    return None if crossing is None else crossing.position_world_m.copy()


def gate_passed(
    previous_world_m: ArrayLike,
    current_world_m: ArrayLike,
    gate: Gate,
    *,
    direction: int = 1,
) -> bool:
    """Return whether the motion segment made a valid directed gate pass."""
    return swept_gate_intersection(
        previous_world_m,
        current_world_m,
        gate,
        direction=direction,
    ) is not None


def ground_collision(
    position_world_m: ArrayLike,
    *,
    vehicle_radius_m: float = 0.0,
    ground_height_m: float = 0.0,
) -> bool:
    """Test a spherical vehicle against the horizontal ground plane."""
    position = _vector3(position_world_m, "position_world_m")
    radius = _finite_nonnegative(vehicle_radius_m, "vehicle_radius_m")
    try:
        height = float(ground_height_m)
    except (TypeError, ValueError) as exc:
        raise ValueError("ground_height_m must be finite") from exc
    if not np.isfinite(height):
        raise ValueError("ground_height_m must be finite")
    return bool(position[2] - radius <= height)


def gate_frame_collision(
    position_world_m: ArrayLike,
    gate: Gate,
    *,
    vehicle_radius_m: float = 0.0,
) -> bool:
    """Test a spherical vehicle against the four rectangular bars of a gate.

    This is intentionally a small, deterministic helper for the Python
    reference path.  It models frame rails rather than treating a valid
    aperture pass as a collision, and can be replaced by a full CCD geometry
    backend without changing pass semantics.
    """
    if not isinstance(gate, Gate):
        raise TypeError("gate must be a Gate")
    local = gate.local_coordinates(position_world_m)
    radius = _finite_nonnegative(vehicle_radius_m, "vehicle_radius_m")
    thickness = gate.frame_thickness_m
    depth = gate.frame_depth_m

    # A zero-size frame is useful for purely logical gates and has no geometry.
    if thickness == 0.0 or depth == 0.0:
        return False

    # Gate-local axes are [normal/depth, right/width, up/height].  Four AABBs
    # form the vertical and horizontal frame rails.  A sphere intersects a box
    # when the distance from its center to that box is no greater than radius.
    rails = (
        (
            np.array([0.0, gate.half_width_m + thickness / 2.0, 0.0]),
            np.array([depth / 2.0, thickness / 2.0, gate.half_height_m + thickness]),
        ),
        (
            np.array([0.0, -gate.half_width_m - thickness / 2.0, 0.0]),
            np.array([depth / 2.0, thickness / 2.0, gate.half_height_m + thickness]),
        ),
        (
            np.array([0.0, 0.0, gate.half_height_m + thickness / 2.0]),
            np.array([depth / 2.0, gate.half_width_m + thickness, thickness / 2.0]),
        ),
        (
            np.array([0.0, 0.0, -gate.half_height_m - thickness / 2.0]),
            np.array([depth / 2.0, gate.half_width_m + thickness, thickness / 2.0]),
        ),
    )
    for center, half_extent in rails:
        separation = np.maximum(np.abs(local - center) - half_extent, 0.0)
        if float(np.linalg.norm(separation)) <= radius:
            return True
    return False


# Readable aliases for callers that prefer predicate-style names.
is_ground_collision = ground_collision
is_gate_frame_collision = gate_frame_collision


__all__ = [
    "Gate",
    "GateCrossing",
    "Vector",
    "gate_frame_collision",
    "gate_passed",
    "ground_collision",
    "is_gate_frame_collision",
    "is_ground_collision",
    "swept_gate_crossing",
    "swept_gate_intersection",
]
