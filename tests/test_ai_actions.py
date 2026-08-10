import numpy as np
import pytest

from flightstack.ai.actions import ACTION_SCHEMA_VERSION, action_to_command, normalized_action
from flightstack.sim.vehicle import VehicleConfig


def vehicle() -> VehicleConfig:
    return VehicleConfig.from_toml()


def test_zero_normalized_action_is_exact_hover_through_shared_ctbr_contract() -> None:
    config = vehicle()
    command = action_to_command([0.0, 0.0, 0.0, 0.0], config)
    assert ACTION_SCHEMA_VERSION == "flightstack-ctbr-action-v1"
    assert command.collective_thrust_n == pytest.approx(config.hover_thrust_n)
    np.testing.assert_allclose(command.body_rate_rad_s, 0.0)


def test_action_mapping_uses_total_vehicle_limits_and_rate_limits() -> None:
    config = vehicle()
    high = action_to_command([1.0, 1.0, -1.0, 0.5], config)
    low = action_to_command([-1.0, -1.0, 1.0, -0.5], config)
    assert high.collective_thrust_n == pytest.approx(4.0 * config.motor_max_thrust_n)
    assert low.collective_thrust_n == pytest.approx(4.0 * config.motor_min_thrust_n)
    np.testing.assert_allclose(
        high.body_rate_rad_s,
        [config.max_body_rate_rad_s[0], -config.max_body_rate_rad_s[1], 2.0],
    )
    np.testing.assert_allclose(
        low.body_rate_rad_s,
        [-config.max_body_rate_rad_s[0], config.max_body_rate_rad_s[1], -2.0],
    )


def test_action_mapping_clips_without_mutating_caller_data() -> None:
    raw = np.array([2.0, -3.0, 0.2, 1.5])
    clipped = normalized_action(raw)
    np.testing.assert_allclose(clipped, [1.0, -1.0, 0.2, 1.0])
    np.testing.assert_allclose(raw, [2.0, -3.0, 0.2, 1.5])


@pytest.mark.parametrize("bad", ([0.0, 0.0], [0.0, 0.0, 0.0, np.nan]))
def test_action_mapping_rejects_bad_policy_output(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="normalized action"):
        normalized_action(bad)
