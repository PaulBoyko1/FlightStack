import json

import numpy as np
import pytest

from flightstack.ai.actions import ACTION_SCHEMA_VERSION, action_to_command
from flightstack.ai.config import load_racing_ai_config
from flightstack.ai.errors import PolicyNotTrainedError, PolicySchemaError
from flightstack.ai.policy import (
    LearnedPolicyPilot,
    PolicyMetadata,
    load_policy_metadata,
    metadata_path_for_checkpoint,
)
from flightstack.race import Gate, RaceState, Track
from flightstack.sim.vehicle import FlightState, VehicleConfig


class FixedPolicy:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.calls = 0

    def predict(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, None]:
        assert observation.shape == (27,)
        assert deterministic
        self.calls += 1
        return self.action, None


def vehicle() -> VehicleConfig:
    return VehicleConfig.from_toml()


def race() -> RaceState:
    gate = Gate(
        center_world_m=[0.0, 3.0, 1.0],
        normal_world=[0.0, 1.0, 0.0],
        right_world=[-1.0, 0.0, 0.0],
        up_world=[0.0, 0.0, 1.0],
        half_width_m=1.0,
        half_height_m=1.0,
        gate_id="next",
    )
    result = RaceState(Track(name="one", gates=(gate,), gate_order=(1,)))
    result.start()
    return result


def metadata(config: VehicleConfig) -> PolicyMetadata:
    return PolicyMetadata(
        action_schema_version=ACTION_SCHEMA_VERSION,
        observation_schema_version=load_racing_ai_config().observation.schema_version,
        vehicle_config_hash=config.config_hash,
    )


def test_learned_pilot_emits_only_shared_ctbr_command() -> None:
    config = vehicle()
    policy = FixedPolicy(np.array([0.25, -0.2, 0.3, -0.4]))
    pilot = LearnedPolicyPilot(config, policy, metadata=metadata(config))
    command = pilot.command(FlightState.hovering(config), race(), 0.02)
    expected = action_to_command(policy.action, config)
    assert policy.calls == 1
    assert command.collective_thrust_n == pytest.approx(expected.collective_thrust_n)
    np.testing.assert_allclose(command.body_rate_rad_s, expected.body_rate_rad_s)
    np.testing.assert_allclose(pilot.previous_action, policy.action)
    pilot.reset(FlightState.hovering(config))
    np.testing.assert_allclose(pilot.previous_action, 0.0)


def test_learned_pilot_rejects_incompatible_metadata_or_bad_policy_output() -> None:
    config = vehicle()
    incompatible = PolicyMetadata(
        action_schema_version="old-action",
        observation_schema_version=load_racing_ai_config().observation.schema_version,
        vehicle_config_hash=config.config_hash,
    )
    with pytest.raises(PolicySchemaError, match="action schema"):
        LearnedPolicyPilot(config, FixedPolicy(np.zeros(4)), metadata=incompatible)
    pilot = LearnedPolicyPilot(config, FixedPolicy(np.zeros(3)))
    with pytest.raises(ValueError, match="normalized action"):
        pilot.command(FlightState.hovering(config), race(), 0.02)


def test_checkpoint_metadata_is_required_and_versioned(tmp_path) -> None:
    checkpoint = tmp_path / "ppo_model.zip"
    checkpoint.touch()
    sidecar = metadata_path_for_checkpoint(checkpoint)
    with sidecar.open("w", encoding="utf-8") as handle:
        json.dump(metadata(vehicle()).to_mapping(), handle)
    loaded = load_policy_metadata(checkpoint)
    assert loaded.vehicle_config_hash == vehicle().config_hash
    with pytest.raises(PolicyNotTrainedError, match="does not exist"):
        LearnedPolicyPilot.from_checkpoint(tmp_path / "missing.zip")
