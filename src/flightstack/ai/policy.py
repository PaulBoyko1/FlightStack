"""Checkpoint-backed learned pilot that preserves the canonical CTBR interface."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from flightstack.ai.actions import ACTION_SCHEMA_VERSION, action_to_command, normalized_action
from flightstack.ai.config import ObservationConfig, RacingAIConfig, load_racing_ai_config
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
    ai_config_hash: str
    control_period_s: float
    checkpoint_sha256: str | None = None
    training: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.action_schema_version or not self.observation_schema_version:
            raise ValueError("policy schema versions must be nonempty")
        if not self.vehicle_config_hash or not self.ai_config_hash:
            raise ValueError("vehicle_config_hash and ai_config_hash must be nonempty")
        if not np.isfinite(self.control_period_s) or self.control_period_s <= 0.0:
            raise ValueError("control_period_s must be positive and finite")
        if self.checkpoint_sha256 is not None and (
            len(self.checkpoint_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.checkpoint_sha256)
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")

    def to_mapping(self) -> dict[str, object]:
        return {
            "action_schema_version": self.action_schema_version,
            "observation_schema_version": self.observation_schema_version,
            "vehicle_config_hash": self.vehicle_config_hash,
            "ai_config_hash": self.ai_config_hash,
            "control_period_s": self.control_period_s,
            "checkpoint_sha256": self.checkpoint_sha256,
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
                ai_config_hash=str(value["ai_config_hash"]),
                control_period_s=float(str(value["control_period_s"])),
                checkpoint_sha256=(
                    None
                    if value.get("checkpoint_sha256") is None
                    else str(value["checkpoint_sha256"])
                ),
                training=dict(training),
            )
        except KeyError as exc:
            raise PolicySchemaError(f"checkpoint metadata is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise PolicySchemaError(
                "checkpoint metadata contains an invalid policy contract"
            ) from exc


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


def checkpoint_sha256(path: str | Path) -> str:
    """Return a content hash without loading an untrusted model archive."""
    digest = hashlib.sha256()
    with normalized_checkpoint_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        ai_config: RacingAIConfig | None = None,
        observation_config: ObservationConfig | None = None,
        control_period_s: float | None = None,
        deterministic: bool = True,
        metadata: PolicyMetadata | None = None,
    ) -> None:
        if not isinstance(vehicle, VehicleConfig):
            raise TypeError("vehicle must be a VehicleConfig")
        self.vehicle = vehicle
        self.policy = policy
        selected_ai_config = load_racing_ai_config() if ai_config is None else ai_config
        if not isinstance(selected_ai_config, RacingAIConfig):
            raise TypeError("ai_config must be a RacingAIConfig")
        self.ai_config = selected_ai_config
        self.observation_config = (
            selected_ai_config.observation
            if observation_config is None
            else observation_config
        )
        self.control_period_s = (
            selected_ai_config.environment.control_dt_s
            if control_period_s is None
            else float(control_period_s)
        )
        if not np.isfinite(self.control_period_s) or self.control_period_s <= 0.0:
            raise ValueError("control_period_s must be positive and finite")
        self.deterministic = bool(deterministic)
        self.metadata = metadata
        if metadata is not None:
            self._validate_metadata(metadata)
        self.previous_action = np.zeros(4, dtype=np.float64)
        self._held_command: PilotCommand | None = None
        self._elapsed_since_action_s = 0.0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        vehicle: VehicleConfig | None = None,
        ai_config: RacingAIConfig | None = None,
        observation_config: ObservationConfig | None = None,
        control_period_s: float | None = None,
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
        selected_ai_config = load_racing_ai_config() if ai_config is None else ai_config
        if not isinstance(selected_ai_config, RacingAIConfig):
            raise TypeError("ai_config must be a RacingAIConfig")
        metadata = load_policy_metadata(checkpoint)
        if metadata.checkpoint_sha256 is None:
            raise PolicySchemaError(
                "checkpoint metadata does not include a model content hash; retrain/export it "
                "with the current FlightStack policy contract"
            )
        if checkpoint_sha256(checkpoint) != metadata.checkpoint_sha256:
            raise PolicySchemaError("checkpoint content does not match its FlightStack metadata")
        try:
            stable_baselines: Any = importlib.import_module("stable_baselines3")
        except ModuleNotFoundError as exc:
            raise OptionalTrainingDependencyError(
                "Stable-Baselines3 inference is optional; install it with "
                f"`{TRAIN_EXTRA_COMMAND}`."
            ) from exc
        policy = cast(PredictionPolicy, stable_baselines.PPO.load(str(checkpoint)))
        return cls(
            selected_vehicle,
            policy,
            ai_config=selected_ai_config,
            observation_config=observation_config,
            control_period_s=control_period_s,
            deterministic=deterministic,
            metadata=metadata,
        )

    def reset(self, initial_state: FlightState) -> None:
        """Reset action history and the training-aligned action-rate scheduler."""
        del initial_state
        self.previous_action = np.zeros(4, dtype=np.float64)
        self._held_command = None
        self._elapsed_since_action_s = 0.0

    def command(self, state: FlightState, race: RaceState, dt: float) -> PilotCommand:
        """Hold each inference action at the configured training control rate.

        Training advances one normalized action over ``control_substeps``
        fixed physics ticks.  Interactive and experiment runtimes call pilots
        at each 2 ms physics tick, so this small scheduler preserves exactly
        the deployment rate used by the Gymnasium environment instead of
        accidentally running an exported policy ten times faster.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not isinstance(race, RaceState):
            raise TypeError("learned pilot requires a RaceState")
        if self._held_command is not None:
            self._elapsed_since_action_s += dt
        if (
            self._held_command is None
            or self._elapsed_since_action_s + 1e-12 >= self.control_period_s
        ):
            observation = build_observation(
                state,
                race,
                self.vehicle,
                self.observation_config,
                self.previous_action,
            )
            action = self._predict(observation)
            self.previous_action = action
            self._held_command = action_to_command(action, self.vehicle)
            self._elapsed_since_action_s = max(
                0.0,
                self._elapsed_since_action_s - self.control_period_s,
            )
        return self._held_command

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
        if metadata.ai_config_hash != self.ai_config.config_hash:
            raise PolicySchemaError(
                "checkpoint AI configuration does not match the active FlightStack policy contract"
            )
        if not np.isclose(metadata.control_period_s, self.control_period_s, rtol=0.0, atol=1e-12):
            raise PolicySchemaError(
                "checkpoint control period does not match the active FlightStack policy contract"
            )
        if self.observation_config.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise PolicySchemaError("unsupported active observation schema")
