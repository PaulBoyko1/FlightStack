#pragma once

#include <algorithm>
#include <array>
#include "flightstack/quaternion.hpp"

namespace flightstack {
struct AttitudeP {
    Vec3 gain{5.0, 5.0, 4.0};
    Vec3 max_rate{4.537856055, 4.537856055, 3.839724354};
    Vec3 desired_body_rate(const Quat& current, const Quat& target) const {
        const auto error = rotation_vector_error(current, target);
        Vec3 rate{};
        for (int i = 0; i < 3; ++i) rate[i] = std::clamp(gain[i] * error[i], -max_rate[i], max_rate[i]);
        return rate;
    }
};
} // namespace flightstack
