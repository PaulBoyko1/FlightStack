"""Portable JSON replay capture for simulator sessions and experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flightstack.runtime.pilots import PilotKind
from flightstack.sim.vehicle import FlightState, PilotCommand

REPLAY_FORMAT_VERSION = "flightstack-replay-v1"


@dataclass(frozen=True)
class ReplayFrame:
    """One sampled authoritative state, command, and race event snapshot."""

    state: FlightState
    pilot: PilotKind
    command: PilotCommand
    race: Mapping[str, object] = field(default_factory=dict)
    events: tuple[Mapping[str, object], ...] = ()

    def to_mapping(self) -> dict[str, object]:
        return {
            "state": self.state.to_mapping(),
            "pilot": self.pilot.value,
            "pilot_command": {
                "collective_thrust_n": self.command.collective_thrust_n,
                "body_rate_rad_s": self.command.body_rate_rad_s.tolist(),
            },
            "race": dict(self.race),
            "events": [dict(event) for event in self.events],
        }


class ReplayRecorder:
    """Accumulate an inspectable, schema-versioned replay in memory.

    Replays are deliberately JSON rather than a bespoke binary format.  The
    fixed-step state remains deterministic; JSON makes the first iteration easy
    to inspect in a bug report, experiment artifact, or browser client.
    """

    def __init__(self, metadata: Mapping[str, object]) -> None:
        self.metadata = dict(metadata)
        self._frames: list[ReplayFrame] = []

    @property
    def frames(self) -> tuple[ReplayFrame, ...]:
        return tuple(self._frames)

    def record(
        self,
        state: FlightState,
        pilot: PilotKind,
        command: PilotCommand,
        *,
        race: Mapping[str, object] | None = None,
        events: tuple[Mapping[str, object], ...] = (),
    ) -> None:
        if self._frames and state.sim_time_s < self._frames[-1].state.sim_time_s:
            raise ValueError("replay frames must be recorded in nondecreasing simulation time")
        self._frames.append(
            ReplayFrame(
                state=state.copy(),
                pilot=pilot,
                command=command,
                race={} if race is None else dict(race),
                events=tuple(dict(event) for event in events),
            )
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "format": REPLAY_FORMAT_VERSION,
            "metadata": self.metadata,
            "frames": [frame.to_mapping() for frame in self._frames],
        }

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def load_replay(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a replay document without re-simulating it."""
    source = Path(path)
    decoded: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict) or decoded.get("format") != REPLAY_FORMAT_VERSION:
        raise ValueError("not a supported FlightStack replay")
    if not isinstance(decoded.get("metadata"), dict) or not isinstance(decoded.get("frames"), list):
        raise ValueError("replay must contain object metadata and list frames")
    return decoded
