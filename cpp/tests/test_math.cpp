#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "flightstack/controller.hpp"

namespace {
void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

template <typename Callable>
void require_invalid_argument(Callable&& operation, const char* message) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    } catch (...) {
    }
    require(false, message);
}
} // namespace

int main() {
    using namespace flightstack;

    const Quat identity{1.0, 0.0, 0.0, 0.0};
    const Vec3 omega{0.0, 0.0, 1.2};
    auto q = identity;
    for (int i = 0; i < 1000; ++i) {
        q = integrate_body_rate(q, omega, 0.001);
    }
    require(std::abs(q[0] - std::cos(0.6)) < 1e-10, "constant-rate quaternion scalar");
    require(std::abs(q[3] - std::sin(0.6)) < 1e-10, "constant-rate quaternion z");

    const Quat target{std::cos(0.25), std::sin(0.25), 0.0, 0.0};
    const auto error = rotation_vector_error(identity, target);
    require(std::abs(error[0] - 0.5) < 1e-10, "rotation-vector error x");
    require(std::abs(error[1]) < 1e-12, "rotation-vector error y");
    require(std::abs(error[2]) < 1e-12, "rotation-vector error z");

    AttitudeP controller;
    const auto rate = controller.desired_body_rate(identity, target);
    require(std::abs(rate[0] - 2.5) < 1e-10, "attitude P rate command");

    // C++ rate PID mirrors the Python VectorPID seam: derivative is taken on
    // measurement, the derivative is low-pass filtered, and integration is
    // conditional when the requested output would push farther into saturation.
    VectorPID rate_pid(
        {1.0, 1.0, 1.0},
        {0.5, 0.5, 0.5},
        {0.25, 0.25, 0.25},
        {10.0, 10.0, 10.0},
        {2.0, 2.0, 2.0},
        35.0
    );
    const auto pid_first = rate_pid.update({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.01);
    require(std::abs(pid_first.proportional[0] - 1.0) < 1e-12, "PID proportional term");
    require(std::abs(pid_first.integral[0] - 0.005) < 1e-12, "PID integral term");
    require(std::abs(pid_first.derivative[0]) < 1e-12, "PID first derivative is zero");
    require(std::abs(pid_first.output[0] - 1.005) < 1e-12, "PID first output");

    // Changing only the setpoint must not create derivative kick.
    const auto pid_setpoint_step = rate_pid.update({2.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.01);
    require(
        std::abs(pid_setpoint_step.derivative[0]) < 1e-12,
        "PID derivative is measurement-only"
    );
    const auto pid_measurement_step = rate_pid.update({2.0, 0.0, 0.0}, {0.2, 0.0, 0.0}, 0.01);
    require(pid_measurement_step.derivative[0] < 0.0, "PID derivative opposes rising measurement");

    VectorPID saturating_pid(
        {2.0, 2.0, 2.0},
        {1.0, 1.0, 1.0},
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {5.0, 5.0, 5.0},
        35.0
    );
    const auto saturated = saturating_pid.update({1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, 1.0);
    require(std::abs(saturated.output[0] - 1.0) < 1e-12, "PID output saturation");
    require(
        std::abs(saturating_pid.integral_state()[0]) < 1e-12,
        "PID conditional anti-windup"
    );
    saturating_pid.reset();
    require(std::abs(saturating_pid.integral_state()[0]) < 1e-12, "PID reset integral");

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double infinity = std::numeric_limits<double>::infinity();
    require_invalid_argument(
        [&] { normalize({nan, 0.0, 0.0, 0.0}); },
        "non-finite quaternion rejected"
    );
    require_invalid_argument(
        [&] { integrate_body_rate(identity, omega, infinity); },
        "non-finite integration step rejected"
    );
    require_invalid_argument(
        [&] { integrate_body_rate(identity, {nan, 0.0, 0.0}, 0.001); },
        "non-finite angular rate rejected"
    );
    AttitudeP invalid_controller;
    invalid_controller.gain[0] = nan;
    require_invalid_argument(
        [&] { invalid_controller.desired_body_rate(identity, target); },
        "non-finite controller gain rejected"
    );
    require_invalid_argument(
        [&] {
            VectorPID invalid_pid(
                {1.0, 1.0, 1.0},
                {0.0, 0.0, 0.0},
                {0.0, 0.0, 0.0},
                {1.0, 1.0, 1.0},
                {1.0, 1.0, 1.0},
                nan
            );
            (void) invalid_pid;
        },
        "non-finite PID cutoff rejected"
    );
    require_invalid_argument(
        [&] { rate_pid.update({nan, 0.0, 0.0}, {0.0, 0.0, 0.0}, 0.01); },
        "non-finite PID setpoint rejected"
    );

    std::cout << "FlightStack C++ golden-vector tests passed\n";
    return 0;
}
