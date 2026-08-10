"""Portable, typed replay capture and state-frame playback for FlightStack.

The persisted v1 format is intentionally simple JSON.  It is useful even
without a browser replay UI: this module reconstructs canonical state/CTBR
objects, exposes deterministic frame sampling, and can export a compact CSV
for debugging or plotting.
"""

from __future__ import annotations

import csv
import json
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.math.quaternion import normalize
from flightstack.runtime.pilots import PilotKind
from flightstack.sim.vehicle import FlightState, PilotCommand

REPLAY_FORMAT_VERSION = "flightstack-replay-v1"
Vector: TypeAlias = NDArray[np.float64]
ReplayEventInput: TypeAlias = "ReplayEvent | Mapping[str, object]"


class ReplayFormatError(ValueError):
    """A replay document does not satisfy FlightStack's v1 JSON contract."""


def _json_object(value: object, *, name: str) -> dict[str, object]:
    """Return an independent JSON-safe object while keeping v1 forward-compatible."""
    if not isinstance(value, Mapping):
        raise ReplayFormatError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ReplayFormatError(f"{name} keys must be strings")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        decoded: object = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ReplayFormatError(f"{name} must contain JSON-safe values") from exc
    if not isinstance(decoded, dict):  # Defensive after the Mapping check above.
        raise ReplayFormatError(f"{name} must be an object")
    return cast(dict[str, object], decoded)


def _vector(value: object, size: int, *, name: str) -> Vector:
    try:
        result = np.asarray(cast(ArrayLike, value), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ReplayFormatError(f"{name} must be a finite vector with shape ({size},)") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ReplayFormatError(f"{name} must be a finite vector with shape ({size},)")
    return result.copy()


def _finite_time(value: object, *, name: str) -> float:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise ReplayFormatError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ReplayFormatError(f"{name} must be finite")
    return result


def _copy_command(command: PilotCommand) -> PilotCommand:
    return PilotCommand(
        collective_thrust_n=command.collective_thrust_n,
        body_rate_rad_s=np.asarray(command.body_rate_rad_s, dtype=np.float64).copy(),
    )


def _copy_frame(frame: ReplayFrame) -> ReplayFrame:
    return ReplayFrame(
        state=frame.state.copy(),
        pilot=frame.pilot,
        command=_copy_command(frame.command),
        race=frame.race,
        events=frame.events,
    )


@dataclass(frozen=True)
class ReplayEvent:
    """One JSON-safe race or disturbance event carried by a replay frame.

    V1 deliberately leaves event payload schemas open so additional event kinds
    can be introduced without invalidating old recordings.  ``kind`` is the
    optional persisted ``type`` field and ``data`` contains all remaining
    fields.  Untyped event objects remain valid v1 data.
    """

    kind: str | None
    data: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind is not None and (not isinstance(self.kind, str) or not self.kind):
            raise ReplayFormatError("event.type must be a nonempty string when present")
        payload = _json_object(self.data, name="event data")
        if "type" in payload:
            raise ReplayFormatError("event data must not duplicate the type field")
        object.__setattr__(self, "data", payload)

    @classmethod
    def from_mapping(cls, value: object, *, name: str = "event") -> ReplayEvent:
        mapping = _json_object(value, name=name)
        raw_kind = mapping.pop("type", None)
        if raw_kind is not None and (not isinstance(raw_kind, str) or not raw_kind):
            raise ReplayFormatError(f"{name}.type must be a nonempty string when present")
        return cls(raw_kind, mapping)

    def to_mapping(self) -> dict[str, object]:
        return dict(self.data) if self.kind is None else {"type": self.kind, **dict(self.data)}


def _event_from_input(value: ReplayEventInput, *, name: str) -> ReplayEvent:
    return value if isinstance(value, ReplayEvent) else ReplayEvent.from_mapping(value, name=name)


@dataclass(frozen=True)
class ReplayFrame:
    """One authoritative state, CTBR command, pilot, and race-event snapshot."""

    state: FlightState
    pilot: PilotKind
    command: PilotCommand
    race: Mapping[str, object] = field(default_factory=dict)
    events: tuple[ReplayEvent | Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.state, FlightState):
            raise TypeError("state must be a FlightState")
        if not isinstance(self.pilot, PilotKind):
            raise TypeError("pilot must be a PilotKind")
        if not isinstance(self.command, PilotCommand):
            raise TypeError("command must be a PilotCommand")
        object.__setattr__(self, "state", self.state.copy())
        object.__setattr__(self, "command", _copy_command(self.command))
        object.__setattr__(self, "race", _json_object(self.race, name="frame.race"))
        normalized_events: list[ReplayEvent] = []
        for index, event in enumerate(self.events):
            normalized_events.append(_event_from_input(event, name=f"frame.events[{index}]"))
        object.__setattr__(self, "events", tuple(normalized_events))

    @classmethod
    def from_mapping(cls, value: object, *, name: str = "frame") -> ReplayFrame:
        """Reconstruct one typed frame from its v1 JSON representation."""
        mapping = _json_object(value, name=name)
        try:
            raw_state = _json_object(mapping["state"], name=f"{name}.state")
            state = FlightState(
                sim_time_s=_finite_time(raw_state["sim_time_s"], name=f"{name}.state.sim_time_s"),
                position_world_m=_vector(
                    raw_state["position_world_m"], 3, name=f"{name}.state.position_world_m"
                ),
                velocity_world_m_s=_vector(
                    raw_state["velocity_world_m_s"],
                    3,
                    name=f"{name}.state.velocity_world_m_s",
                ),
                q_body_to_world_wxyz=_vector(
                    raw_state["q_body_to_world_wxyz"],
                    4,
                    name=f"{name}.state.q_body_to_world_wxyz",
                ),
                body_rate_rad_s=_vector(
                    raw_state["body_rate_rad_s"], 3, name=f"{name}.state.body_rate_rad_s"
                ),
                motor_thrust_n=_vector(
                    raw_state["motor_thrust_n"], 4, name=f"{name}.state.motor_thrust_n"
                ),
            )
        except KeyError as exc:
            raise ReplayFormatError(f"{name}.state is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise ReplayFormatError(f"{name}.state is invalid: {exc}") from exc

        raw_pilot = mapping.get("pilot")
        if not isinstance(raw_pilot, str):
            raise ReplayFormatError(f"{name}.pilot must be a PilotKind string")
        try:
            pilot = PilotKind(raw_pilot)
        except ValueError as exc:
            allowed = ", ".join(kind.value for kind in PilotKind)
            raise ReplayFormatError(f"{name}.pilot must be one of {allowed}") from exc

        try:
            raw_command = _json_object(mapping["pilot_command"], name=f"{name}.pilot_command")
            command = PilotCommand(
                collective_thrust_n=_finite_time(
                    raw_command["collective_thrust_n"],
                    name=f"{name}.pilot_command.collective_thrust_n",
                ),
                body_rate_rad_s=_vector(
                    raw_command["body_rate_rad_s"],
                    3,
                    name=f"{name}.pilot_command.body_rate_rad_s",
                ),
            )
        except KeyError as exc:
            raise ReplayFormatError(f"{name}.pilot_command is missing {exc.args[0]!r}") from exc
        except (TypeError, ValueError) as exc:
            raise ReplayFormatError(f"{name}.pilot_command is invalid: {exc}") from exc

        race = _json_object(mapping.get("race", {}), name=f"{name}.race")
        raw_events = mapping.get("events", [])
        if not isinstance(raw_events, list):
            raise ReplayFormatError(f"{name}.events must be a list")
        events = tuple(
            ReplayEvent.from_mapping(event, name=f"{name}.events[{index}]")
            for index, event in enumerate(raw_events)
        )
        return cls(state=state, pilot=pilot, command=command, race=race, events=events)

    def to_mapping(self) -> dict[str, object]:
        return {
            "state": self.state.to_mapping(),
            "pilot": self.pilot.value,
            "pilot_command": {
                "collective_thrust_n": self.command.collective_thrust_n,
                "body_rate_rad_s": self.command.body_rate_rad_s.tolist(),
            },
            "race": dict(self.race),
            "events": [
                _event_from_input(event, name=f"frame.events[{index}]").to_mapping()
                for index, event in enumerate(self.events)
            ],
        }


@dataclass(frozen=True)
class ReplayDocument:
    """Validated, typed representation of a FlightStack replay-v1 document."""

    metadata: Mapping[str, object]
    frames: tuple[ReplayFrame, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _json_object(self.metadata, name="metadata"))
        normalized_frames = tuple(self.frames)
        if any(not isinstance(frame, ReplayFrame) for frame in normalized_frames):
            raise TypeError("frames must contain ReplayFrame instances")
        for previous, current in zip(normalized_frames, normalized_frames[1:], strict=False):
            if current.state.sim_time_s < previous.state.sim_time_s:
                raise ReplayFormatError("replay frame times must be nondecreasing")
        object.__setattr__(self, "frames", tuple(_copy_frame(frame) for frame in normalized_frames))

    @classmethod
    def from_mapping(cls, value: object) -> ReplayDocument:
        """Parse a v1 JSON object, rejecting malformed frame contracts early."""
        mapping = _json_object(value, name="replay")
        if mapping.get("format") != REPLAY_FORMAT_VERSION:
            raise ReplayFormatError("not a supported FlightStack replay")
        metadata = _json_object(mapping.get("metadata"), name="replay.metadata")
        raw_frames = mapping.get("frames")
        if not isinstance(raw_frames, list):
            raise ReplayFormatError("replay.frames must be a list")
        frames = tuple(
            ReplayFrame.from_mapping(frame, name=f"replay.frames[{index}]")
            for index, frame in enumerate(raw_frames)
        )
        return cls(metadata=metadata, frames=frames)

    @property
    def start_time_s(self) -> float | None:
        """First recorded simulation time, or ``None`` for a valid empty replay."""
        return None if not self.frames else self.frames[0].state.sim_time_s

    @property
    def end_time_s(self) -> float | None:
        """Last recorded simulation time, or ``None`` for a valid empty replay."""
        return None if not self.frames else self.frames[-1].state.sim_time_s

    @property
    def duration_s(self) -> float:
        """Recorded time span, excluding any unrecorded pre-roll."""
        if not self.frames:
            return 0.0
        return float(self.frames[-1].state.sim_time_s - self.frames[0].state.sim_time_s)

    def to_mapping(self) -> dict[str, object]:
        """Serialize exactly the existing replay-v1 envelope."""
        return {
            "format": REPLAY_FORMAT_VERSION,
            "metadata": dict(self.metadata),
            "frames": [frame.to_mapping() for frame in self.frames],
        }

    def summary(self) -> dict[str, object]:
        """Return compact JSON-safe facts suitable for CLI inspection."""
        pilots = Counter(frame.pilot.value for frame in self.frames)
        events = Counter(
            _event_from_input(event, name=f"frame.events[{index}]").kind or "<untyped>"
            for frame in self.frames
            for index, event in enumerate(frame.events)
        )
        return {
            "format": REPLAY_FORMAT_VERSION,
            "frame_count": len(self.frames),
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "pilots": dict(sorted(pilots.items())),
            "events": dict(sorted(events.items())),
            "metadata": dict(self.metadata),
        }


class ReplayRecorder:
    """Accumulate an inspectable, schema-versioned replay in memory.

    Replays are deliberately JSON rather than a bespoke binary format.  The
    fixed-step state remains deterministic; JSON makes the first iteration easy
    to inspect in a bug report, experiment artifact, or browser client.
    """

    def __init__(self, metadata: Mapping[str, object]) -> None:
        self.metadata = _json_object(metadata, name="metadata")
        self._frames: list[ReplayFrame] = []

    @property
    def frames(self) -> tuple[ReplayFrame, ...]:
        return tuple(_copy_frame(frame) for frame in self._frames)

    def record(
        self,
        state: FlightState,
        pilot: PilotKind,
        command: PilotCommand,
        *,
        race: Mapping[str, object] | None = None,
        events: Iterable[ReplayEventInput] = (),
    ) -> None:
        frame = ReplayFrame(
            state=state,
            pilot=pilot,
            command=command,
            race={} if race is None else race,
            events=tuple(events),
        )
        if self._frames and frame.state.sim_time_s < self._frames[-1].state.sim_time_s:
            raise ValueError("replay frames must be recorded in nondecreasing simulation time")
        self._frames.append(frame)

    def document(self) -> ReplayDocument:
        """Return the current recording as a typed replay-v1 document."""
        return ReplayDocument(self.metadata, tuple(self._frames))

    def to_mapping(self) -> dict[str, object]:
        return self.document().to_mapping()

    def write(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return destination


def _slerp(lhs: ArrayLike, rhs: ArrayLike, fraction: float) -> Vector:
    """Shortest-arc quaternion interpolation in FlightStack's scalar-first order."""
    left = normalize(lhs)
    right = normalize(rhs)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return normalize((1.0 - fraction) * left + fraction * right)
    angle = float(np.arccos(dot))
    sin_angle = float(np.sin(angle))
    return normalize(
        (np.sin((1.0 - fraction) * angle) / sin_angle) * left
        + (np.sin(fraction * angle) / sin_angle) * right
    )


def _interpolate_frame(left: ReplayFrame, right: ReplayFrame, fraction: float) -> ReplayFrame:
    """Interpolate state only; discrete pilot/race/event data remains left-held."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("interpolation fraction must be strictly between zero and one")
    lhs = left.state
    rhs = right.state
    state = FlightState(
        sim_time_s=lhs.sim_time_s + fraction * (rhs.sim_time_s - lhs.sim_time_s),
        position_world_m=(1.0 - fraction) * lhs.position_world_m + fraction * rhs.position_world_m,
        velocity_world_m_s=(1.0 - fraction) * lhs.velocity_world_m_s
        + fraction * rhs.velocity_world_m_s,
        q_body_to_world_wxyz=_slerp(lhs.q_body_to_world_wxyz, rhs.q_body_to_world_wxyz, fraction),
        body_rate_rad_s=(1.0 - fraction) * lhs.body_rate_rad_s + fraction * rhs.body_rate_rad_s,
        motor_thrust_n=(1.0 - fraction) * lhs.motor_thrust_n + fraction * rhs.motor_thrust_n,
    )
    return ReplayFrame(
        state=state,
        pilot=left.pilot,
        command=left.command,
        race=left.race,
        events=(),
    )


class ReplayPlayer:
    """Deterministic sampling over recorded state frames without re-simulation.

    Exact sampling uses zero-order hold: ``frame_at(t)`` returns the last
    authoritative recorded frame at or before ``t``.  With ``interpolate=True``
    it linearly interpolates continuous state fields and SLERPs attitude between
    adjacent frames.  Pilot, CTBR command, race snapshot, and events stay
    discrete/left-held; interpolated frames intentionally have no events.
    """

    def __init__(self, replay: ReplayDocument) -> None:
        if not isinstance(replay, ReplayDocument):
            raise TypeError("replay must be a ReplayDocument")
        self.replay = replay
        self._times = tuple(frame.state.sim_time_s for frame in replay.frames)

    @property
    def frames(self) -> tuple[ReplayFrame, ...]:
        """Return independent copies of the authoritative recorded frames."""
        return tuple(_copy_frame(frame) for frame in self.replay.frames)

    def frame_at(self, time_s: float, *, interpolate: bool = False) -> ReplayFrame:
        """Return a clamped source frame or an interpolated state at ``time_s``."""
        if not self.replay.frames:
            raise ReplayFormatError("cannot play an empty replay")
        time = _finite_time(time_s, name="time_s")
        first = self.replay.frames[0]
        last = self.replay.frames[-1]
        if time <= first.state.sim_time_s:
            return _copy_frame(first)
        if time >= last.state.sim_time_s:
            return _copy_frame(last)

        right_index = bisect_right(self._times, time)
        left = self.replay.frames[right_index - 1]
        right = self.replay.frames[right_index]
        if not interpolate or time <= left.state.sim_time_s:
            return _copy_frame(left)
        span = right.state.sim_time_s - left.state.sim_time_s
        if span <= 0.0:
            return _copy_frame(left)
        return _interpolate_frame(left, right, (time - left.state.sim_time_s) / span)

    def sampled_frames(
        self,
        period_s: float,
        *,
        start_time_s: float | None = None,
        end_time_s: float | None = None,
    ) -> Iterator[ReplayFrame]:
        """Yield deterministic interpolated samples on an inclusive regular grid."""
        period = _finite_time(period_s, name="period_s")
        if period <= 0.0:
            raise ValueError("period_s must be positive and finite")
        if not self.replay.frames:
            return
        first = self.replay.frames[0].state.sim_time_s
        last = self.replay.frames[-1].state.sim_time_s
        requested_start = (
            first if start_time_s is None else _finite_time(start_time_s, name="start_time_s")
        )
        requested_end = last if end_time_s is None else _finite_time(end_time_s, name="end_time_s")
        start = float(np.clip(requested_start, first, last))
        end = float(np.clip(requested_end, first, last))
        if end < start:
            raise ValueError("end_time_s must not precede start_time_s")
        count = int(np.floor((end - start) / period + 1e-12))
        for index in range(count + 1):
            yield self.frame_at(start + index * period, interpolate=True)
        sampled_end = start + count * period
        if not np.isclose(sampled_end, end, rtol=0.0, atol=1e-12):
            yield self.frame_at(end, interpolate=True)

    def export_csv(
        self,
        path: str | Path,
        *,
        sample_period_s: float | None = None,
    ) -> Path:
        """Write source frames or a regular interpolated state timeline as CSV."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frames: Iterable[ReplayFrame]
        if sample_period_s is None:
            frames = self.frames
        else:
            frames = self.sampled_frames(sample_period_s)
        fieldnames = (
            "sim_time_s",
            "pilot",
            "position_x_m",
            "position_y_m",
            "position_z_m",
            "velocity_x_m_s",
            "velocity_y_m_s",
            "velocity_z_m_s",
            "q_w",
            "q_x",
            "q_y",
            "q_z",
            "body_rate_p_rad_s",
            "body_rate_q_rad_s",
            "body_rate_r_rad_s",
            "motor_0_thrust_n",
            "motor_1_thrust_n",
            "motor_2_thrust_n",
            "motor_3_thrust_n",
            "collective_thrust_n",
            "command_p_rad_s",
            "command_q_rad_s",
            "command_r_rad_s",
            "race_json",
            "events_json",
        )
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for frame in frames:
                state = frame.state
                command = frame.command
                writer.writerow(
                    {
                        "sim_time_s": state.sim_time_s,
                        "pilot": frame.pilot.value,
                        "position_x_m": state.position_world_m[0],
                        "position_y_m": state.position_world_m[1],
                        "position_z_m": state.position_world_m[2],
                        "velocity_x_m_s": state.velocity_world_m_s[0],
                        "velocity_y_m_s": state.velocity_world_m_s[1],
                        "velocity_z_m_s": state.velocity_world_m_s[2],
                        "q_w": state.q_body_to_world_wxyz[0],
                        "q_x": state.q_body_to_world_wxyz[1],
                        "q_y": state.q_body_to_world_wxyz[2],
                        "q_z": state.q_body_to_world_wxyz[3],
                        "body_rate_p_rad_s": state.body_rate_rad_s[0],
                        "body_rate_q_rad_s": state.body_rate_rad_s[1],
                        "body_rate_r_rad_s": state.body_rate_rad_s[2],
                        "motor_0_thrust_n": state.motor_thrust_n[0],
                        "motor_1_thrust_n": state.motor_thrust_n[1],
                        "motor_2_thrust_n": state.motor_thrust_n[2],
                        "motor_3_thrust_n": state.motor_thrust_n[3],
                        "collective_thrust_n": command.collective_thrust_n,
                        "command_p_rad_s": command.body_rate_rad_s[0],
                        "command_q_rad_s": command.body_rate_rad_s[1],
                        "command_r_rad_s": command.body_rate_rad_s[2],
                        "race_json": json.dumps(frame.race, sort_keys=True),
                        "events_json": json.dumps(
                            [
                                _event_from_input(
                                    event,
                                    name=f"frame.events[{index}]",
                                ).to_mapping()
                                for index, event in enumerate(frame.events)
                            ],
                            sort_keys=True,
                        ),
                    }
                )
        return destination


def read_replay(path: str | Path) -> ReplayDocument:
    """Read and fully validate a v1 JSON replay into canonical typed objects."""
    source = Path(path)
    try:
        decoded: object = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReplayFormatError(f"replay JSON is invalid: {source}") from exc
    return ReplayDocument.from_mapping(decoded)


def load_replay(path: str | Path) -> dict[str, Any]:
    """Return a validated v1 replay mapping for backward-compatible callers.

    New callers should prefer :func:`read_replay` and :class:`ReplayPlayer` for
    canonical objects and deterministic state-frame playback.
    """
    return cast(dict[str, Any], read_replay(path).to_mapping())


__all__ = [
    "REPLAY_FORMAT_VERSION",
    "ReplayDocument",
    "ReplayEvent",
    "ReplayFormatError",
    "ReplayFrame",
    "ReplayPlayer",
    "ReplayRecorder",
    "load_replay",
    "read_replay",
]
