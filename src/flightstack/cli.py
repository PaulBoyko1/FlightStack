"""FlightStack command-line tools."""

from __future__ import annotations

import argparse
import csv
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
            writer.writerow(["time_s", "error_deg", "p_rad_s", "q_rad_s", "r_rad_s", "tx_Nm", "ty_Nm", "tz_Nm"])
            for index, time_s in enumerate(telemetry.time_s):
                writer.writerow([time_s, np.rad2deg(telemetry.attitude_error_rad[index]), *telemetry.body_rate[index], *telemetry.torque[index]])
        print(f"telemetry: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="flightstack")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sim = subparsers.add_parser("simulate", help="run the reference attitude step")
    sim.add_argument("--duration", type=float, default=4.0)
    sim.add_argument("--dt", type=float, default=0.002)
    sim.add_argument("--csv", type=str)
    sim.set_defaults(func=_simulate)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
