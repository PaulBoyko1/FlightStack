#include <cmath>
#include <cstdlib>
#include <iostream>
#include "flightstack/controller.hpp"

namespace {
void require(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAILED: " << message << '\n'; std::exit(1); }
}
}

int main() {
    using namespace flightstack;
    const Quat identity{1.0, 0.0, 0.0, 0.0};
    const Vec3 omega{0.0, 0.0, 1.2};
    auto q = identity;
    for (int i = 0; i < 1000; ++i) q = integrate_body_rate(q, omega, 0.001);
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
    std::cout << "FlightStack C++ golden-vector tests passed\n";
    return 0;
}
