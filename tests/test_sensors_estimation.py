import numpy as np
from flightstack.estimation.complementary import ComplementaryAttitudeEstimator
from flightstack.math.quaternion import from_euler, rotate_inverse
from flightstack.sensors.imu import IMUSimulator


def test_imu_is_deterministic_for_seed() -> None:
    a = IMUSimulator(gyro_noise_std=0.01, accel_noise_std=0.02, seed=42)
    b = IMUSimulator(gyro_noise_std=0.01, accel_noise_std=0.02, seed=42)
    sa = a.sample(0.0, [1, 0, 0, 0], [0.1, 0.2, 0.3])
    sb = b.sample(0.0, [1, 0, 0, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(sa.gyro_rad_s, sb.gyro_rad_s)
    np.testing.assert_allclose(sa.accel_m_s2, sb.accel_m_s2)


def test_complementary_filter_reduces_tilt_error_at_rest() -> None:
    true_q = from_euler(0.0, 0.0, 0.0)
    estimator = ComplementaryAttitudeEstimator(accel_correction_gain=2.5, initial_q=from_euler(np.deg2rad(20), np.deg2rad(-15), 0.0))
    imu = IMUSimulator()
    true_up = np.array([0.0, 0.0, 1.0])
    initial_up = rotate_inverse(estimator.attitude, true_up)
    initial_tilt = np.arccos(np.clip(initial_up @ true_up, -1.0, 1.0))
    for i in range(1500):
        sample = imu.sample(i * 0.002, true_q, [0, 0, 0])
        estimator.update(sample.gyro_rad_s, sample.accel_m_s2, 0.002)
    final_up = rotate_inverse(estimator.attitude, true_up)
    final_tilt = np.arccos(np.clip(final_up @ true_up, -1.0, 1.0))
    assert final_tilt < initial_tilt * 0.05
