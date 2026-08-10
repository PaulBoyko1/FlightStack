#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "flightstack/quaternion.hpp"

namespace flightstack {
struct AttitudeP {
    Vec3 gain{5.0, 5.0, 4.0};
    Vec3 max_rate{4.537856055, 4.537856055, 3.839724354};

    Vec3 desired_body_rate(const Quat& current, const Quat& target) const {
        if (!is_finite(gain) || !is_finite(max_rate)) {
            throw std::invalid_argument("controller gains and limits must be finite");
        }
        const auto error = rotation_vector_error(current, target);
        Vec3 rate{};
        for (int i = 0; i < 3; ++i) {
            if (gain[i] < 0.0 || max_rate[i] <= 0.0) {
                throw std::invalid_argument("controller gains must be nonnegative and limits positive");
            }
            rate[i] = std::clamp(gain[i] * error[i], -max_rate[i], max_rate[i]);
        }
        return rate;
    }
};

struct PIDTerms {
    Vec3 proportional{};
    Vec3 integral{};
    Vec3 derivative{};
    Vec3 output{};
};

class VectorPID {
public:
    VectorPID(
        Vec3 kp,
        Vec3 ki,
        Vec3 kd,
        Vec3 output_limit,
        Vec3 integral_limit,
        double derivative_cutoff_hz = 35.0
    )
        : kp_(kp),
          ki_(ki),
          kd_(kd),
          output_limit_(output_limit),
          integral_limit_(integral_limit),
          derivative_cutoff_hz_(derivative_cutoff_hz) {
        validate_configuration();
    }

    void reset() {
        integral_ = {0.0, 0.0, 0.0};
        previous_measurement_ = {0.0, 0.0, 0.0};
        filtered_derivative_ = {0.0, 0.0, 0.0};
        has_previous_measurement_ = false;
    }

    PIDTerms update(const Vec3& setpoint, const Vec3& measurement, double dt) {
        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::invalid_argument("dt must be positive and finite");
        }
        if (!is_finite(setpoint) || !is_finite(measurement)) {
            throw std::invalid_argument("setpoint and measurement must be finite");
        }

        Vec3 error{};
        Vec3 proportional{};
        Vec3 raw_derivative{};
        for (int i = 0; i < 3; ++i) {
            error[i] = setpoint[i] - measurement[i];
            proportional[i] = kp_[i] * error[i];
            raw_derivative[i] = has_previous_measurement_
                ? -(measurement[i] - previous_measurement_[i]) / dt
                : 0.0;
        }
        previous_measurement_ = measurement;
        has_previous_measurement_ = true;

        constexpr double pi = 3.141592653589793238462643383279502884;
        const double tau = 1.0 / (2.0 * pi * derivative_cutoff_hz_);
        const double alpha = dt / (tau + dt);

        Vec3 derivative{};
        Vec3 proposed_integral{};
        Vec3 unsaturated{};
        for (int i = 0; i < 3; ++i) {
            filtered_derivative_[i] += alpha * (raw_derivative[i] - filtered_derivative_[i]);
            derivative[i] = kd_[i] * filtered_derivative_[i];
            proposed_integral[i] = std::clamp(
                integral_[i] + error[i] * dt,
                -integral_limit_[i],
                integral_limit_[i]
            );
            unsaturated[i] =
                proportional[i] + ki_[i] * proposed_integral[i] + derivative[i];

            const bool pushing_high = unsaturated[i] > output_limit_[i] && error[i] > 0.0;
            const bool pushing_low = unsaturated[i] < -output_limit_[i] && error[i] < 0.0;
            if (!(pushing_high || pushing_low)) {
                integral_[i] = proposed_integral[i];
            }
        }

        PIDTerms terms{};
        terms.proportional = proportional;
        terms.derivative = derivative;
        for (int i = 0; i < 3; ++i) {
            terms.integral[i] = ki_[i] * integral_[i];
            terms.output[i] = std::clamp(
                terms.proportional[i] + terms.integral[i] + terms.derivative[i],
                -output_limit_[i],
                output_limit_[i]
            );
        }
        return terms;
    }

    const Vec3& integral_state() const { return integral_; }

private:
    void validate_configuration() const {
        if (!is_finite(kp_) || !is_finite(ki_) || !is_finite(kd_)
            || !is_finite(output_limit_) || !is_finite(integral_limit_)) {
            throw std::invalid_argument("PID gains and limits must be finite");
        }
        if (!std::isfinite(derivative_cutoff_hz_) || derivative_cutoff_hz_ <= 0.0) {
            throw std::invalid_argument("derivative cutoff must be positive and finite");
        }
        for (int i = 0; i < 3; ++i) {
            if (kp_[i] < 0.0 || ki_[i] < 0.0 || kd_[i] < 0.0) {
                throw std::invalid_argument("PID gains must be nonnegative");
            }
            if (output_limit_[i] <= 0.0 || integral_limit_[i] <= 0.0) {
                throw std::invalid_argument("PID limits must be positive");
            }
        }
    }

    Vec3 kp_{};
    Vec3 ki_{};
    Vec3 kd_{};
    Vec3 output_limit_{};
    Vec3 integral_limit_{};
    double derivative_cutoff_hz_{};
    Vec3 integral_{0.0, 0.0, 0.0};
    Vec3 previous_measurement_{0.0, 0.0, 0.0};
    Vec3 filtered_derivative_{0.0, 0.0, 0.0};
    bool has_previous_measurement_{false};
};
} // namespace flightstack
