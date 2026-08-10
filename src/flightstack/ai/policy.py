"""Checkpoint-backed learned pilot that preserves the canonical CTBR interface."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from flightstack.ai.actions import ACTION_SCHEMA_VERSION, action_to_command, normalized_action
from flightstack.ai.config import ObservationConfig, load_racing_ai_config
from flightstack.ai.errors import (
    TRAIN_EXTRA_COMMAND,
    OptionalTrainingDependencyError,
    PolicyNotTrainedError,
    PolicySchemaError,
)
from flightstack.ai.observation import OBSERVATION_SCHEMA_VERSION, build_observation
from flightstack.race import RaceState
from flightstack.runtime.pilots import PilotKind
from flightstack.sim.vehicle import FlightState, PilotCommand, VehicleConfig

Vector = NDArray[np.float64]


class PredictionPolicy(Protocol):
    """The small Stable-Baselines3-compatible inference surface we require."""

    def predict(self, observation: NDArray[np.float32], *, deterministic: bool) -> object: ...


@dataclass(frozen=True)
class PolicyMetadata:
    """Compatibility facts written beside every exported learned checkpoint."""

    action_schema_version: str
    observation_schema_version: str
    vehicle_config_hash: str
    training: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_schema_version or not self.observation_schema_version:
            raise ValueError("policy schema versions must be nonempty")
        if not self.vehicle_config_hash:
            raise ValueError("vehicle_config_hash must be nonempty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_schema_version": self.action_schema_version,
            "observation_schema_version": self.observation_schema_version,
            "vehicle_config_hash": self.vehicle_config_hash,
            "training": self.training,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PolicyMetadata:
        training = value.get("training", {})
        if not isinstance(training, dict):
            raise PolicySchemaError("checkpoint metadata training field must be an object")
        try:
            return cls(
                action_schema_version=str(value["action_schema_version"]),
                observation_schema_version=str(value["observation_schema_version"]),
                vehicle_config_hash=str(value["vehicle_config_hash"]),
                training=dict(training),
            )
        except KeyError as exc:
            raise PolicySchemaError(f"checkpoint metadata is missing {exc.args[0]!r}") from exc


def normalized_checkpoint_path(path: str | Path) -> Path:
    """Resolve either `model` or `model.zip` to its expected on-disk path."""
    candidate = Path(path).expanduser()
    if candidate.suffix == "" and candidate.with_suffix(".zip").is_file():
        candidate = candidate.with_suffix(".zip")
    return candidate.resolve()


def metadata_path_for_checkpoint(path: str | Path) -> Path:
    """Return the versioned metadata sidecar path for an SB3 checkpoint."""
    checkpoint = normalized_checkpoint_path(path)
    return checkpoint.with_suffix(".metadata.json")


def load_policy_metadata(path: str | Path) -> PolicyMetadata:
    """Load a required sidecar instead of guessing a policy's input contract."""
    metadata_path = metadata_path_for_checkpoint(path)
    if not metadata_path.is_file():
        raise PolicyNotTrainedError(
            "learned checkpoint metadata is missing: "
            f"{metadata_path}. Train/export a FlightStack policy before selecting Learned."
        )
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            raw: object = json.load(handle)
    except json.JSONDecodeError as exc:
        raise PolicySchemaError(f"checkpoint metadata is invalid JSON: {metadata_path}") from exc
    if not isinstance(raw, dict):
        raise PolicySchemaError("checkpoint metadata must be a JSON object")
    return PolicyMetadata.from_mapping(raw)


class LearnedPolicyPilot:
    """A deterministic policy adapter that emits only canonical CTBR commands.

    Use :meth:`from_checkpoint` in a runtime process.  There is intentionally
    no fallback to a hover/classical command: a missing or incompatible model
    raises a named error that UI/server code can present to a user.
    """

    kind = PilotKind.LEARNED

    def __init__(
        self,
        vehicle: VehicleConfig,
        policy: PredictionPolicy | Callable[[NDArray[np.float32]], object],
        *,
        observation_config: ObservationConfig | None = None,
        deterministic: bool = True,
        metadata: PolicyMetadata | None = None,
    ) -> None:
        if not isinstance(vehicle, VehicleConfig):
            raise TypeError("vehicle must be a VehicleConfig")
        self.vehicle = vehicle
        self.policy = policy
        self.observation_config = (
            load_racing_ai_config().observation
            if observation_config is None
            else observation_config
        )
        self.deterministic = bool(deterministic)
        self.metadata = metadata
        if metadata is not None:
            self._validate_metadata(metadata)
        self.previous_action = np.zeros(4, dtype=np.float64)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        vehicle: VehicleConfig | None = None,
        observation_config: ObservationConfig | None = None,
        deterministic: bool = True,
    ) -> LearnedPolicyPilot:
        """Load an SB3 PPO checkpoint and reject missing/incompatible metadata."""
        checkpoint = normalized_checkpoint_path(checkpoint_path)
        if not checkpoint.is_file():
            raise PolicyNotTrainedError(
                f"learned checkpoint does not exist: {checkpoint}. "
                "Train/export a FlightStack policy before selecting Learned."
            )
        selected_vehicle = VehicleConfig.from_toml() if vehicle is None else vehicle
        metadata = load_policy_metadata(checkpoint)
        try:
            from stable_baselines3 import PPO  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise OptionalTrainingDependencyError(
                "Stable-Baselines3 inference is optional; install it with "
                f"`{TRAIN_EXTRA_COMMAND}`."
            ) from exc
        policy: PredictionPolicy = PPO.load(str(checkpoint))
        return cls(
            selected_vehicle,
            policy,
            observation_config=observation_config,
            deterministic=deterministic,
            metadata=metadata,
        )

    def reset(self, initial_state: FlightState) -> None:
        """Reset action history at episode/replay boundaries."""
        del initial_state
        self.previous_action = np.zeros(4, dtype=np.float64)

    def command(self, state: FlightState, race: RaceState, dt: float) -> PilotCommand:
        """Run inference once and map its output through the shared CTBR seam."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not isinstance(race, RaceState):
            raise TypeError("learned pilot requires a RaceState")
        observation = build_observation(
            state,
            race,
            self.vehicle,
            self.observation_config,
            self.previous_action,
        )
        action = self._predict(observation)
        self.previous_action = action
        return action_to_command(action, self.vehicle)

    def _predict(self, observation: NDArray[np.float32]) -> Vector:
        policy = self.policy
        if hasattr(policy, "predict"):
            prediction = policy.predict(observation, deterministic=self.deterministic)
        elif callable(policy):
            prediction = policy(observation)
        else:
            raise PolicyNotTrainedError("learned policy does not provide predict(observation)")
        action = prediction[0] if isinstance(prediction, tuple) else prediction
        return normalized_action(np.asarray(action, dtype=np.float64))

    def _validate_metadata(self, metadata: PolicyMetadata) -> None:
        if metadata.action_schema_version != ACTION_SCHEMA_VERSION:
            raise PolicySchemaError(
                "checkpoint action schema does not match FlightStack: "
                f"{metadata.action_schema_version!r}"
            )
        if metadata.observation_schema_version != self.observation_config.schema_version:
            raise PolicySchemaError(
                "checkpoint observation schema does not match FlightStack: "
                f"{metadata.observation_schema_version!r}"
            )
        if metadata.vehicle_config_hash != self.vehicle.config_hash:
            raise PolicySchemaError(
                "checkpoint vehicle configuration does not match the active FlightStack vehicle"
            )
        if self.observation_config.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise PolicySchemaError("unsupported active observation schema")
