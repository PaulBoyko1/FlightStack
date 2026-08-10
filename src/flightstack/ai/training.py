"""Optional Stable-Baselines3 PPO entry point for the FlightStack race task."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from flightstack.ai.actions import ACTION_SCHEMA_VERSION
from flightstack.ai.config import RacingAIConfig, load_racing_ai_config
from flightstack.ai.environment import make_gymnasium_env
from flightstack.ai.errors import TRAIN_EXTRA_COMMAND, OptionalTrainingDependencyError
from flightstack.ai.observation import OBSERVATION_SCHEMA_VERSION
from flightstack.ai.policy import PolicyMetadata, checkpoint_sha256, metadata_path_for_checkpoint
from flightstack.race import Track, load_technical_eight
from flightstack.sim.vehicle import VehicleConfig


@dataclass(frozen=True)
class PPOTrainingConfig:
    """Small, explicit PPO configuration suitable for deterministic smoke runs."""

    total_timesteps: int = 25_000
    seed: int = 42
    n_envs: int = 1
    n_steps: int = 128
    batch_size: int = 64
    n_epochs: int = 2
    learning_rate: float = 3e-4

    def __post_init__(self) -> None:
        for name in ("total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite")


@dataclass(frozen=True)
class TrainingResult:
    """Paths and run facts returned after an SB3 training/export transaction."""

    checkpoint_path: Path
    metadata_path: Path
    total_timesteps: int


def _canonical_hash(value: object) -> str:
    """Hash JSON-safe FlightStack contracts in a stable representation."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dependency_versions(sb3: Any) -> dict[str, str]:
    """Record the actual training stack without making it a core dependency."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "stable_baselines3": str(getattr(sb3, "__version__", "unknown")),
    }


def _require_train_dependencies() -> tuple[Any, Any]:
    """Load training-only dependencies lazily with one actionable command."""
    try:
        importlib.import_module("gymnasium")
        sb3 = importlib.import_module("stable_baselines3")
    except ModuleNotFoundError as exc:
        raise OptionalTrainingDependencyError(
            "PPO training is optional; install Gymnasium and Stable-Baselines3 with "
            f"`{TRAIN_EXTRA_COMMAND}`."
        ) from exc
    try:
        vec_env = importlib.import_module("stable_baselines3.common.vec_env")
    except ModuleNotFoundError as exc:
        raise OptionalTrainingDependencyError(
            "Stable-Baselines3 is incomplete; reinstall the train extra with "
            f"`{TRAIN_EXTRA_COMMAND}`."
        ) from exc
    return sb3, vec_env


def train_ppo(
    output_dir: str | Path,
    *,
    training: PPOTrainingConfig | None = None,
    vehicle: VehicleConfig | None = None,
    track: Track | None = None,
    ai_config: RacingAIConfig | None = None,
) -> TrainingResult:
    """Train and export a PPO policy with versioned FlightStack metadata.

    This delegates optimisation and neural-network implementation to
    Stable-Baselines3.  The environment and policy output seam remain owned by
    FlightStack, so a checkpoint cannot silently drift from its vehicle or
    observation/action schema.
    """
    sb3, vec_env = _require_train_dependencies()
    selected_training = PPOTrainingConfig() if training is None else training
    selected_vehicle = VehicleConfig.from_toml() if vehicle is None else vehicle
    selected_ai = load_racing_ai_config() if ai_config is None else ai_config
    selected_track = load_technical_eight() if track is None else track
    if not isinstance(selected_track, Track):
        raise TypeError("track must be a Track")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    def environment_factory(index: int) -> Any:
        return make_gymnasium_env(
            vehicle=selected_vehicle,
            track=selected_track,
            ai_config=selected_ai,
        )

    factories = [
        (lambda index=index: environment_factory(index))
        for index in range(selected_training.n_envs)
    ]
    environment = vec_env.DummyVecEnv(factories)
    ppo_class = sb3.PPO
    model = ppo_class(
        "MlpPolicy",
        environment,
        seed=selected_training.seed,
        verbose=0,
        n_steps=selected_training.n_steps,
        batch_size=min(selected_training.batch_size, selected_training.n_steps),
        n_epochs=selected_training.n_epochs,
        learning_rate=selected_training.learning_rate,
    )
    try:
        model.learn(total_timesteps=selected_training.total_timesteps)
        checkpoint_base = destination / "ppo_model"
        model.save(str(checkpoint_base))
    finally:
        environment.close()
    checkpoint = checkpoint_base.with_suffix(".zip")
    if not checkpoint.is_file():
        raise RuntimeError("Stable-Baselines3 did not produce the expected PPO checkpoint")
    metadata = PolicyMetadata(
        action_schema_version=ACTION_SCHEMA_VERSION,
        observation_schema_version=OBSERVATION_SCHEMA_VERSION,
        vehicle_config_hash=selected_vehicle.config_hash,
        ai_config_hash=selected_ai.config_hash,
        control_period_s=selected_ai.environment.control_dt_s,
        checkpoint_sha256=checkpoint_sha256(checkpoint),
        training={
            "algorithm": "PPO",
            "training_config": asdict(selected_training),
            "ai_config": selected_ai.to_mapping(),
            "environment_schema_version": selected_ai.environment.schema_version,
            "reward_schema_version": selected_ai.reward.schema_version,
            "track": {
                "name": selected_track.name,
                "contract_sha256": _canonical_hash(selected_track.to_mapping()),
            },
            "dependencies": _dependency_versions(sb3),
        },
    )
    metadata_path = metadata_path_for_checkpoint(checkpoint)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata.to_mapping(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return TrainingResult(
        checkpoint_path=checkpoint,
        metadata_path=metadata_path,
        total_timesteps=selected_training.total_timesteps,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a FlightStack state-racing PPO policy")
    parser.add_argument("--output", type=Path, required=True, help="directory for ppo_model.zip")
    parser.add_argument("--timesteps", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a small 256-step PPO smoke training after installing .[train]",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a real, deliberately small training job from ``python -m``."""
    args = _parser().parse_args(argv)
    if args.smoke:
        config = PPOTrainingConfig(
            total_timesteps=256,
            seed=args.seed,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
        )
    else:
        config = PPOTrainingConfig(total_timesteps=args.timesteps, seed=args.seed)
    result = train_ppo(args.output, training=config)
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"metadata: {result.metadata_path}")
    print(f"timesteps: {result.total_timesteps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
