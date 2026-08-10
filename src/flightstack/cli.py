"""FlightStack command-line tools."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from flightstack.sim.scenarios import run_reference_step


def _simulate(args: argparse.Namespace) -> int:
    telemetry = run_reference_step(duration_s=args.duration, dt=args.dt)
    print(f"final attitude error: {telemetry.final_error_deg:.4f} deg")
    print(f"peak body rate: {np.rad2deg(telemetry.peak_rate_rad_s):.2f} deg/s")
    if args.csv is not None:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "time_s",
                    "error_deg",
                    "p_rad_s",
                    "q_rad_s",
                    "r_rad_s",
                    "tx_Nm",
                    "ty_Nm",
                    "tz_Nm",
                ]
            )
            for index, time_s in enumerate(telemetry.time_s):
                writer.writerow(
                    [
                        time_s,
                        np.rad2deg(telemetry.attitude_error_rad[index]),
                        *telemetry.body_rate[index],
                        *telemetry.torque[index],
                    ]
                )
        print(f"telemetry: {path}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    from flightstack.web.server import run

    run(host=args.host, port=args.port, policy_path=args.policy)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    """Run one reproducible headless race and optionally write its artifacts."""
    from flightstack.ai.policy import LearnedPolicyPilot
    from flightstack.experiments import checkpoint_model_identity, load_scenario, run_episode
    from flightstack.runtime.autonomy import ClassicalRacePilot
    from flightstack.sim.vehicle import VehicleConfig

    scenario = load_scenario(args.scenario)
    if args.pilot == "classical":
        result = run_episode(scenario, ClassicalRacePilot, pilot_name="classical")
    else:
        if args.policy is None:
            raise ValueError("--policy is required for a learned evaluation")

        def learned_factory(vehicle: VehicleConfig) -> LearnedPolicyPilot:
            return LearnedPolicyPilot.from_checkpoint(args.policy, vehicle=vehicle)

        result = run_episode(
            scenario,
            learned_factory,
            pilot_name="learned",
            pilot_model_identity=checkpoint_model_identity(args.policy),
        )
    payload = result.to_mapping()
    if args.output is not None:
        artifacts = result.write_artifacts(args.output)
        payload["artifacts"] = {
            "summary": str(artifacts.summary_path),
            "telemetry": str(artifacts.telemetry_path),
            "replay": str(artifacts.replay_path),
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _replay(args: argparse.Namespace) -> int:
    """Inspect a recorded replay and optionally export state frames as CSV."""
    from flightstack.runtime.replay import ReplayPlayer, read_replay

    replay = read_replay(args.path)
    player = ReplayPlayer(replay)
    payload = replay.summary()
    if args.at is not None:
        frame = player.frame_at(args.at, interpolate=args.interpolate)
        payload["requested_time_s"] = args.at
        payload["interpolated"] = bool(args.interpolate)
        payload["frame"] = frame.to_mapping()
    if args.csv is not None:
        destination = player.export_csv(args.csv, sample_period_s=args.sample_period)
        payload["csv"] = str(destination)
        payload["csv_sample_period_s"] = args.sample_period
    elif args.sample_period is not None:
        raise ValueError("--sample-period requires --csv")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _train(args: argparse.Namespace) -> int:
    """Train PPO through the optional, maintained Stable-Baselines3 extra."""
    from flightstack.ai.errors import OptionalTrainingDependencyError
    from flightstack.ai.training import PPOTrainingConfig, train_ppo

    config = (
        PPOTrainingConfig(
            total_timesteps=256,
            seed=args.seed,
            n_envs=1,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
        )
        if args.smoke
        else PPOTrainingConfig(
            total_timesteps=args.timesteps,
            seed=args.seed,
            n_envs=args.n_envs,
        )
    )
    try:
        result = train_ppo(args.output, training=config)
    except OptionalTrainingDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"checkpoint: {result.checkpoint_path}")
    print(f"metadata: {result.metadata_path}")
    print(f"timesteps: {result.total_timesteps}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flightstack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sim = subparsers.add_parser("simulate", help="run the reference attitude step")
    sim.add_argument("--duration", type=float, default=4.0)
    sim.add_argument("--dt", type=float, default=0.002)
    sim.add_argument("--csv", type=str)
    sim.set_defaults(func=_simulate)
    serve = subparsers.add_parser("serve", help="run the authoritative interactive simulator")
    serve.add_argument("--host", type=str, default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--policy",
        type=Path,
        help="validated Stable-Baselines3 FlightStack checkpoint (.zip) for Learned mode",
    )
    serve.set_defaults(func=_serve)
    evaluate = subparsers.add_parser("evaluate", help="run a reproducible headless race episode")
    evaluate.add_argument("--scenario", type=str, default="technical-eight-wind-degraded")
    evaluate.add_argument("--pilot", choices=("classical", "learned"), default="classical")
    evaluate.add_argument("--policy", type=Path, help="checkpoint required when --pilot learned")
    evaluate.add_argument(
        "--output",
        type=Path,
        help="directory for result, telemetry, and replay JSON",
    )
    evaluate.set_defaults(func=_evaluate)
    replay = subparsers.add_parser(
        "replay",
        help="inspect a FlightStack replay-v1 file or export recorded state frames",
    )
    replay.add_argument("path", type=Path, help="replay JSON produced by a FlightStack session")
    replay.add_argument(
        "--at",
        type=float,
        help="simulation time to inspect (clamped to the recorded frame range)",
    )
    replay.add_argument(
        "--interpolate",
        action="store_true",
        help="interpolate continuous state at --at; race/pilot/events remain discrete",
    )
    replay.add_argument("--csv", type=Path, help="write state, CTBR, race, and event data as CSV")
    replay.add_argument(
        "--sample-period",
        type=float,
        help="CSV interpolation period in seconds; omit to export authoritative source frames",
    )
    replay.set_defaults(func=_replay)
    train = subparsers.add_parser(
        "train",
        help="train a state-based PPO race policy (optional extra)",
    )
    train.add_argument(
        "--output",
        type=Path,
        required=True,
        help="directory for PPO checkpoint files",
    )
    train.add_argument("--timesteps", type=int, default=25_000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="parallel environments; values above 1 use subprocess workers",
    )
    train.add_argument(
        "--smoke",
        action="store_true",
        help="run a short 256-step single-environment PPO smoke train",
    )
    train.set_defaults(func=_train)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
