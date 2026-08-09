"""Quaternion utilities using scalar-first [w, x, y, z] convention.

The attitude quaternion maps body-frame vectors into the world frame. Body angular
rate therefore integrates on the right: q_next = q * exp(omega_body * dt / 2).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

Vector = NDArray[np.float64]


def _vec(value: ArrayLike, size: int, name: str) -> Vector:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (size,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return out


def _finite_scalar(value: float, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def normalize(q: ArrayLike) -> Vector:
    quat = _vec(q, 4, "quaternion")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion norm is zero or non-finite")
    return quat / norm


def conjugate(q: ArrayLike) -> Vector:
    w, x, y, z = _vec(q, 4, "quaternion")
    return np.array([w, -x, -y, -z], dtype=np.float64)


def multiply(lhs: ArrayLike, rhs: ArrayLike) -> Vector:
    w1, x1, y1, z1 = _vec(lhs, 4, "lhs")
    w2, x2, y2, z2 = _vec(rhs, 4, "rhs")
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise ValueError("quaternion product must be finite")
    return result


def from_axis_angle(axis: ArrayLike, angle_rad: float) -> Vector:
    axis_vec = _vec(axis, 3, "axis")
    angle = _finite_scalar(angle_rad, "angle_rad")
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-12:
        if abs(angle) < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        raise ValueError("nonzero rotation requires a nonzero axis")
    half = 0.5 * angle
    result = np.concatenate(
        ([np.cos(half)], axis_vec / axis_norm * np.sin(half))
    )
    return np.asarray(result, dtype=np.float64)


def from_rotation_vector(rotation: ArrayLike) -> Vector:
    rot = _vec(rotation, 3, "rotation")
    angle = float(np.linalg.norm(rot))
    if angle < 1e-12:
        return normalize(np.concatenate(([1.0], 0.5 * rot)))
    return from_axis_angle(rot / angle, angle)


def from_euler(roll: float, pitch: float, yaw: float) -> Vector:
    """Create body-to-world quaternion from intrinsic XYZ / roll-pitch-yaw angles."""
    roll_value = _finite_scalar(roll, "roll")
    pitch_value = _finite_scalar(pitch, "pitch")
    yaw_value = _finite_scalar(yaw, "yaw")
    cr, sr = np.cos(roll_value / 2.0), np.sin(roll_value / 2.0)
    cp, sp = np.cos(pitch_value / 2.0), np.sin(pitch_value / 2.0)
    cy, sy = np.cos(yaw_value / 2.0), np.sin(yaw_value / 2.0)
    return normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def to_rotation_matrix(q: ArrayLike) -> NDArray[np.float64]:
    w, x, y, z = normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotate(q: ArrayLike, vector_body: ArrayLike) -> Vector:
    return to_rotation_matrix(q) @ _vec(vector_body, 3, "vector_body")


def rotate_inverse(q: ArrayLike, vector_world: ArrayLike) -> Vector:
    return to_rotation_matrix(q).T @ _vec(vector_world, 3, "vector_world")


def wxyz_to_xyzw(q_wxyz: ArrayLike) -> Vector:
    """Adapt FlightStack's scalar-first quaternion for scalar-last APIs.

    This is intentionally a named boundary rather than a convenience scattered
    through render, ML, or third-party integration code.  Both representations
    retain the same body-to-world rotation semantics.
    """
    w, x, y, z = _vec(q_wxyz, 4, "q_wxyz")
    return np.array([x, y, z, w], dtype=np.float64)


def xyzw_to_wxyz(q_xyzw: ArrayLike) -> Vector:
    """Adapt a scalar-last body-to-world quaternion into FlightStack form."""
    x, y, z, w = _vec(q_xyzw, 4, "q_xyzw")
    return np.array([w, x, y, z], dtype=np.float64)


def relative_body_error(current: ArrayLike, target: ArrayLike) -> Vector:
    """Return shortest-path current->target relative quaternion expressed in body frame."""
    q_error = normalize(multiply(conjugate(normalize(current)), normalize(target)))
    if q_error[0] < 0.0:
        q_error = -q_error
    return q_error


def rotation_vector_error(current: ArrayLike, target: ArrayLike) -> Vector:
    """Exact shortest current->target rotation vector expressed in body coordinates."""
    q_error = relative_body_error(current, target)
    vector = q_error[1:]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return np.asarray(2.0 * vector, dtype=np.float64)
    angle = float(2.0 * np.arctan2(vector_norm, float(q_error[0])))
    return np.asarray(vector / vector_norm * angle, dtype=np.float64)


def error_vector(current: ArrayLike, target: ArrayLike) -> Vector:
    """Backward-compatible alias for the exact body-frame rotation-vector error."""
    return rotation_vector_error(current, target)


def geodesic_angle(current: ArrayLike, target: ArrayLike) -> float:
    """Exact shortest angular distance between two attitudes, in radians."""
    q_error = relative_body_error(current, target)
    return float(2.0 * np.arctan2(np.linalg.norm(q_error[1:]), abs(q_error[0])))


def integrate_body_rate(q: ArrayLike, omega_body: ArrayLike, dt: float) -> Vector:
    """Integrate constant body rate exactly over dt using the quaternion exponential."""
    step = _finite_scalar(dt, "dt")
    if step <= 0.0:
        raise ValueError("dt must be positive and finite")
    omega = _vec(omega_body, 3, "omega_body")
    delta = from_rotation_vector(omega * step)
    return normalize(multiply(normalize(q), delta))
