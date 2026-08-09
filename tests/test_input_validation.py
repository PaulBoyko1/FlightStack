import numpy as np
import pytest

from flightstack.control.attitude import AttitudeController
from flightstack.control.pid import VectorPID
from flightstack.estimation.complementary import ComplementaryAttitudeEstimator
from flightstack.hil.protocol import Packet, crc16_ccitt
from flightstack.math.quaternion import from_axis_angle, from_euler, integrate_body_rate
from flightstack.sensors.imu import IMUSimulator
from flightstack.sim.rigid_body import RigidBody
from flightstack.sim.runner import simulate_attitude_step


def make_rate_pid() -> VectorPID:
    return VectorPID(
        kp=[1.0, 1.0, 1.0],
        ki=[0.0, 0.0, 0.0],
        kd=[0.0, 0.0, 0.0],
        output_limit=1.0,
        integral_limit=1.0,
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_quaternion_constructors_reject_nonfinite_scalars(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        from_axis_angle([1.0, 0.0, 0.0], invalid)
    with pytest.raises(ValueError, match="finite"):
        from_euler(invalid, 0.0, 0.0)
    with pytest.raises(ValueError, match="finite"):
        integrate_body_rate([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0], invalid)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_attitude_controller_rejects_nonfinite_configuration(invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        AttitudeController(
            attitude_kp=[invalid, 1.0, 1.0],
            rate_pid=make_rate_pid(),
            max_rate_rad_s=1.0,
        )
    with pytest.raises(ValueError, match="finite"):
        AttitudeController(
            attitude_kp=[1.0, 1.0, 1.0],
            rate_pid=make_rate_pid(),
            max_rate_rad_s=invalid,
        )


def test_estimator_rejects_nonfinite_measurements() -> None:
    estimator = ComplementaryAttitudeEstimator()
    with pytest.raises(ValueError, match="finite"):
        estimator.update([np.nan, 0.0, 0.0], [0.0, 0.0, 9.8], 0.01)
    with pytest.raises(ValueError, match="finite"):
        estimator.update([0.0, 0.0, 0.0], [np.inf, 0.0, 9.8], 0.01)


def test_imu_rejects_nonfinite_configuration_and_samples() -> None:
    with pytest.raises(ValueError, match="finite"):
        IMUSimulator(gyro_bias_rad_s=[np.nan, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        IMUSimulator(gyro_noise_std=np.inf)

    imu = IMUSimulator()
    with pytest.raises(ValueError, match="timestamp"):
        imu.sample(np.nan, [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        imu.sample(0.0, [1.0, 0.0, 0.0, 0.0], [np.nan, 0.0, 0.0])


def test_simulation_rejects_nonfinite_duration() -> None:
    controller = AttitudeController(
        attitude_kp=[1.0, 1.0, 1.0],
        rate_pid=make_rate_pid(),
        max_rate_rad_s=1.0,
    )
    with pytest.raises(ValueError, match="positive and finite"):
        simulate_attitude_step(
            RigidBody(np.eye(3)),
            controller,
            [1.0, 0.0, 0.0, 0.0],
            duration_s=np.inf,
        )


def test_hil_packet_contracts_reject_invalid_types_and_ranges() -> None:
    with pytest.raises(ValueError, match="message_id"):
        Packet(True, b"")
    with pytest.raises(TypeError, match="payload"):
        Packet(1, bytearray())
    with pytest.raises(TypeError, match="frame"):
        Packet.decode(bytearray(Packet(1, b"payload").encode()))
    with pytest.raises(ValueError, match="initial"):
        crc16_ccitt(b"payload", initial=0x1_0000)
