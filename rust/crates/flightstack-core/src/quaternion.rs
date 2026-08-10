//! Explicit scalar-first quaternion operations for FlightStack's body-to-world
//! convention.

use crate::{ensure_finite, ensure_finite_vec3, ContractError, QuatWxyz, Vec3};

const MIN_QUATERNION_NORM: f64 = 1.0e-12;

/// Convert FlightStack storage `[w, x, y, z]` to scalar-last `[x, y, z, w]`.
///
/// This changes storage order only; it does not change frame semantics.
pub fn wxyz_to_xyzw(q_wxyz: QuatWxyz) -> QuatWxyz {
    [q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]
}

/// Convert a scalar-last body-to-world quaternion into FlightStack storage.
pub fn xyzw_to_wxyz(q_xyzw: QuatWxyz) -> QuatWxyz {
    [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
}

/// Euclidean norm of a finite scalar-first quaternion.
pub fn norm(q: QuatWxyz) -> Result<f64, ContractError> {
    ensure_finite_quaternion(q)?;
    let scale = q.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Ok(0.0);
    }
    let scaled_norm = q
        .iter()
        .map(|value| {
            let scaled = value / scale;
            scaled * scaled
        })
        .sum::<f64>()
        .sqrt();
    let magnitude = scale * scaled_norm;
    if magnitude.is_finite() {
        Ok(magnitude)
    } else {
        Err(ContractError::new(
            "quaternion norm is too large to represent as f64",
        ))
    }
}

/// Normalize a scalar-first quaternion, rejecting zero/non-finite input.
pub fn normalize(q: QuatWxyz) -> Result<QuatWxyz, ContractError> {
    ensure_finite_quaternion(q)?;
    // Scale before squaring so finite inputs close to `f64::MAX` do not
    // overflow. The normalized result remains representable even if the input
    // norm itself would be larger than an `f64` can express.
    let scale = q.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if scale < MIN_QUATERNION_NORM {
        return Err(ContractError::new("quaternion norm is zero or too small"));
    }
    let scaled_norm = q
        .iter()
        .map(|value| {
            let scaled = value / scale;
            scaled * scaled
        })
        .sum::<f64>()
        .sqrt();
    if !scaled_norm.is_finite() || scaled_norm < MIN_QUATERNION_NORM {
        return Err(ContractError::new("quaternion norm is zero or non-finite"));
    }
    Ok([
        (q[0] / scale) / scaled_norm,
        (q[1] / scale) / scaled_norm,
        (q[2] / scale) / scaled_norm,
        (q[3] / scale) / scaled_norm,
    ])
}

/// Return the conjugate of a scalar-first quaternion.
pub fn conjugate(q: QuatWxyz) -> Result<QuatWxyz, ContractError> {
    ensure_finite_quaternion(q)?;
    Ok([q[0], -q[1], -q[2], -q[3]])
}

/// Hamilton product of two scalar-first quaternions.
pub fn multiply(lhs: QuatWxyz, rhs: QuatWxyz) -> Result<QuatWxyz, ContractError> {
    ensure_finite_quaternion(lhs)?;
    ensure_finite_quaternion(rhs)?;
    let product = [
        lhs[0] * rhs[0] - lhs[1] * rhs[1] - lhs[2] * rhs[2] - lhs[3] * rhs[3],
        lhs[0] * rhs[1] + lhs[1] * rhs[0] + lhs[2] * rhs[3] - lhs[3] * rhs[2],
        lhs[0] * rhs[2] - lhs[1] * rhs[3] + lhs[2] * rhs[0] + lhs[3] * rhs[1],
        lhs[0] * rhs[3] + lhs[1] * rhs[2] - lhs[2] * rhs[1] + lhs[3] * rhs[0],
    ];
    if product.iter().all(|value| value.is_finite()) {
        Ok(product)
    } else {
        Err(ContractError::new("quaternion product must be finite"))
    }
}

/// Build a scalar-first quaternion from an axis-angle rotation in radians.
pub fn from_axis_angle(axis: Vec3, angle_rad: f64) -> Result<QuatWxyz, ContractError> {
    ensure_finite_vec3(axis, "axis")?;
    ensure_finite(angle_rad, "angle_rad")?;
    let axis_norm = vector_norm(axis);
    if axis_norm < MIN_QUATERNION_NORM {
        if angle_rad.abs() < MIN_QUATERNION_NORM {
            return Ok([1.0, 0.0, 0.0, 0.0]);
        }
        return Err(ContractError::new(
            "a nonzero rotation requires a nonzero axis",
        ));
    }
    let half_angle = angle_rad * 0.5;
    let scale = half_angle.sin() / axis_norm;
    normalize([
        half_angle.cos(),
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
    ])
}

/// Build a scalar-first quaternion from a body-frame rotation vector.
pub fn from_rotation_vector(rotation_rad: Vec3) -> Result<QuatWxyz, ContractError> {
    ensure_finite_vec3(rotation_rad, "rotation_rad")?;
    let angle = vector_norm(rotation_rad);
    if angle < MIN_QUATERNION_NORM {
        return normalize([
            1.0,
            0.5 * rotation_rad[0],
            0.5 * rotation_rad[1],
            0.5 * rotation_rad[2],
        ]);
    }
    from_axis_angle(
        [
            rotation_rad[0] / angle,
            rotation_rad[1] / angle,
            rotation_rad[2] / angle,
        ],
        angle,
    )
}

/// Rotate a body-frame vector into the world frame.
pub fn rotate_body_to_world(
    q_body_to_world: QuatWxyz,
    vector_body: Vec3,
) -> Result<Vec3, ContractError> {
    ensure_finite_vec3(vector_body, "vector_body")?;
    let [w, x, y, z] = normalize(q_body_to_world)?;
    let matrix = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ],
        [
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ],
        [
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    Ok(matrix_vector_product(matrix, vector_body))
}

/// Rotate a world-frame vector into the body frame.
pub fn rotate_world_to_body(
    q_body_to_world: QuatWxyz,
    vector_world: Vec3,
) -> Result<Vec3, ContractError> {
    ensure_finite_vec3(vector_world, "vector_world")?;
    let [w, x, y, z] = normalize(q_body_to_world)?;
    let matrix_transpose = [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ];
    Ok(matrix_vector_product(matrix_transpose, vector_world))
}

/// Exactly integrate a constant body rate over a positive timestep.
///
/// Body-rate integration is right-multiplicative because the stored
/// quaternion maps body vectors to the world frame:
/// `q_next = q_body_to_world * Exp(omega_body * dt)`.
pub fn integrate_body_rate(
    q_body_to_world: QuatWxyz,
    body_rate_rad_s: Vec3,
    dt_s: f64,
) -> Result<QuatWxyz, ContractError> {
    ensure_finite_vec3(body_rate_rad_s, "body_rate_rad_s")?;
    ensure_finite(dt_s, "dt_s")?;
    if dt_s <= 0.0 {
        return Err(ContractError::new("dt_s must be positive"));
    }
    let delta = from_rotation_vector([
        body_rate_rad_s[0] * dt_s,
        body_rate_rad_s[1] * dt_s,
        body_rate_rad_s[2] * dt_s,
    ])?;
    normalize(multiply(normalize(q_body_to_world)?, delta)?)
}

fn vector_norm(vector: Vec3) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn ensure_finite_quaternion(q: QuatWxyz) -> Result<(), ContractError> {
    if q.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ContractError::new(
            "quaternion must contain only finite values",
        ))
    }
}

fn matrix_vector_product(matrix: [[f64; 3]; 3], vector: Vec3) -> Vec3 {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1.0e-12;

    fn assert_vector_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected) {
            assert!(
                (actual_value - expected_value).abs() < EPSILON,
                "{actual_value} did not equal {expected_value}"
            );
        }
    }

    #[test]
    fn scalar_order_adapter_round_trips_identity_and_principal_axes() {
        let quaternions = [
            [1.0, 0.0, 0.0, 0.0],
            from_axis_angle([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2).expect("x rotation"),
            from_axis_angle([0.0, 1.0, 0.0], std::f64::consts::FRAC_PI_2).expect("y rotation"),
            from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2).expect("z rotation"),
        ];
        for quaternion in quaternions {
            assert_vector_close(&xyzw_to_wxyz(wxyz_to_xyzw(quaternion)), &quaternion);
        }
    }

    #[test]
    fn scalar_order_adapter_preserves_arbitrary_and_negative_quaternion() {
        let quaternion = normalize([0.31, -0.27, 0.49, 0.76]).expect("normalizable");
        let round_trip = xyzw_to_wxyz(wxyz_to_xyzw(quaternion));
        assert_vector_close(&round_trip, &quaternion);
        let vector_body = [0.3, -0.2, 0.7];
        assert_vector_close(
            &rotate_body_to_world(round_trip, vector_body).expect("adapted rotation"),
            &rotate_body_to_world(quaternion, vector_body).expect("canonical rotation"),
        );
        let negative = quaternion.map(|value| -value);
        assert_vector_close(&xyzw_to_wxyz(wxyz_to_xyzw(negative)), &negative);
    }

    #[test]
    fn normalization_handles_huge_finite_components_without_overflow() {
        let huge = [f64::MAX, -f64::MAX, f64::MAX, -f64::MAX];
        let normalized = normalize(huge).expect("finite quaternion is normalizable");
        assert_vector_close(&normalized, &[0.5, -0.5, 0.5, -0.5]);
        assert_vector_close(&[norm(normalized).expect("unit norm")], &[1.0]);
        assert!(norm(huge).is_err(), "unrepresentable norm is rejected");
    }

    #[test]
    fn body_to_world_rotation_has_expected_principal_axis_parity() {
        let q_x =
            from_axis_angle([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2).expect("x rotation");
        assert_vector_close(
            &rotate_body_to_world(q_x, [0.0, 1.0, 0.0]).expect("rotation"),
            &[0.0, 0.0, 1.0],
        );

        let q_y =
            from_axis_angle([0.0, 1.0, 0.0], std::f64::consts::FRAC_PI_2).expect("y rotation");
        assert_vector_close(
            &rotate_body_to_world(q_y, [0.0, 0.0, 1.0]).expect("rotation"),
            &[1.0, 0.0, 0.0],
        );

        let q_z =
            from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2).expect("z rotation");
        assert_vector_close(
            &rotate_body_to_world(q_z, [1.0, 0.0, 0.0]).expect("rotation"),
            &[0.0, 1.0, 0.0],
        );
    }

    #[test]
    fn body_rate_integration_is_right_multiplicative() {
        let initial = from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)
            .expect("initial rotation");
        let next = integrate_body_rate(initial, [std::f64::consts::PI, 0.0, 0.0], 0.5)
            .expect("integration");
        let expected = multiply(
            initial,
            from_axis_angle([1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2).expect("body delta"),
        )
        .expect("product");
        assert_vector_close(&next, &normalize(expected).expect("normalized expected"));
    }
}
