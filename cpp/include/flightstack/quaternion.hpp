#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

namespace flightstack {
using Vec3 = std::array<double, 3>;
using Quat = std::array<double, 4>;

inline double norm3(const Vec3& v) { return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]); }
inline Quat normalize(const Quat& q) {
    const double n = std::sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    if (n < 1e-12) throw std::invalid_argument("zero quaternion");
    return {q[0]/n, q[1]/n, q[2]/n, q[3]/n};
}
inline Quat conjugate(const Quat& q) { return {q[0], -q[1], -q[2], -q[3]}; }
inline Quat multiply(const Quat& a, const Quat& b) {
    return {
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]
    };
}
inline Quat relative_body_error(const Quat& current, const Quat& target) {
    auto error = normalize(multiply(conjugate(normalize(current)), normalize(target)));
    if (error[0] < 0.0) for (double& value : error) value = -value;
    return error;
}
inline Vec3 rotation_vector_error(const Quat& current, const Quat& target) {
    const auto q = relative_body_error(current, target);
    const Vec3 vector{q[1], q[2], q[3]};
    const double n = norm3(vector);
    if (n < 1e-12) return {2.0*q[1], 2.0*q[2], 2.0*q[3]};
    const double angle = 2.0 * std::atan2(n, q[0]);
    return {angle*vector[0]/n, angle*vector[1]/n, angle*vector[2]/n};
}
inline Quat integrate_body_rate(const Quat& q, const Vec3& omega, double dt) {
    if (!(dt > 0.0)) throw std::invalid_argument("dt must be positive");
    const Vec3 rotation{omega[0]*dt, omega[1]*dt, omega[2]*dt};
    const double angle = norm3(rotation);
    Quat delta{};
    if (angle < 1e-12) {
        delta = normalize({1.0, 0.5*rotation[0], 0.5*rotation[1], 0.5*rotation[2]});
    } else {
        const double half = 0.5 * angle;
        const double scale = std::sin(half) / angle;
        delta = {std::cos(half), rotation[0]*scale, rotation[1]*scale, rotation[2]*scale};
    }
    return normalize(multiply(normalize(q), delta));
}
} // namespace flightstack
