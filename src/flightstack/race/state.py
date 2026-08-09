"""Ordered, event-oriented race progress for the Python reference runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from flightstack.race.geometry import Gate, GateCrossing, Vector, swept_gate_intersection
from flightstack.race.track import Track


def _time_s(value: float, name: str = "time_s") -> float:
    """Validate a nonnegative simulation timestamp."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative and finite") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return result


def _gate_index(value: int) -> int:
    """Validate a zero-based physical gate index."""
    if isinstance(value, bool):
        raise ValueError("gate_index must be a nonnegative integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("gate_index must be a nonnegative integer") from exc
    if result < 0 or result != value:
        raise ValueError("gate_index must be a nonnegative integer")
    return result


def _direction(value: int | None) -> int | None:
    """Validate an optional signed gate direction."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("direction must be either -1 or +1")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("direction must be either -1 or +1") from exc
    if result not in (-1, 1):
        raise ValueError("direction must be either -1 or +1")
    return result


def _vector3(value: ArrayLike, name: str) -> Vector:
    """Validate an immutable world-space event vector."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    result = vector.copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class Start:
    """Command event that begins a race run."""

    time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _time_s(self.time_s))


@dataclass(frozen=True)
class Reset:
    """Command event that clears the active run and returns to an idle state."""

    time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _time_s(self.time_s))


@dataclass(frozen=True)
class GatePassed:
    """A verified pass through one physical gate's aperture."""

    gate_index: int
    crossing_position_world_m: Vector
    time_s: float
    direction: int | None = None
    segment_fraction: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gate_index", _gate_index(self.gate_index))
        object.__setattr__(
            self,
            "crossing_position_world_m",
            _vector3(self.crossing_position_world_m, "crossing_position_world_m"),
        )
        object.__setattr__(self, "time_s", _time_s(self.time_s))
        object.__setattr__(self, "direction", _direction(self.direction))
        if self.segment_fraction is not None:
            fraction = float(self.segment_fraction)
            if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
                raise ValueError("segment_fraction must be finite and in [0, 1]")
            object.__setattr__(self, "segment_fraction", fraction)


@dataclass(frozen=True)
class Collision:
    """A collision with a named world object."""

    object_id: str
    time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id:
            raise ValueError("object_id must be a nonempty string")
        object.__setattr__(self, "time_s", _time_s(self.time_s))


@dataclass(frozen=True)
class LapCompleted:
    """Derived event emitted after the final ordered gate of a lap."""

    lap: int
    lap_time_s: float
    time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "lap", _positive_int(self.lap, "lap"))
        lap_time = _time_s(self.lap_time_s, "lap_time_s")
        object.__setattr__(self, "lap_time_s", lap_time)
        object.__setattr__(self, "time_s", _time_s(self.time_s))


@dataclass(frozen=True)
class RaceFinished:
    """Derived event emitted when the requested number of laps is complete."""

    time_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _time_s(self.time_s))


RaceEvent: TypeAlias = Start | Reset | GatePassed | Collision | LapCompleted | RaceFinished


def _positive_int(value: int, name: str) -> int:
    """Validate an integer count."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


class RaceState:
    """Race progress driven by explicit events and swept gate observations.

    A state instance owns one run.  Call :meth:`reset` to begin a new run, then
    :meth:`start`.  :meth:`update_position` checks only the *next ordered gate*
    and advances at most one order entry per motion segment, making wrong-order
    and duplicate passes harmless.
    """

    def __init__(
        self,
        track: Track,
        total_laps: int | None = None,
        *,
        laps: int | None = None,
    ) -> None:
        if not isinstance(track, Track):
            raise TypeError("track must be a Track")
        if total_laps is not None and laps is not None:
            raise ValueError("specify either total_laps or laps, not both")
        requested_laps = total_laps if total_laps is not None else laps
        self.track = track
        self.total_laps = track.default_laps if requested_laps is None else _positive_int(
            requested_laps, "total_laps"
        )
        self.running = False
        self.finished = False
        self.lap = 0
        self.next_gate_order_index = 0
        self.lap_started_at_s: float | None = None
        self.last_gate_at_s: float | None = None
        self.best_lap_s: float | None = None
        self.completed_lap_times_s: list[float] = []
        self.collisions = 0
        self.events: list[RaceEvent] = []
        self._last_event_time_s: float | None = None
        self._repeat_gate_rearm_required = False

    @property
    def laps(self) -> int:
        """Alias for the configured race length."""
        return self.total_laps

    @property
    def next_gate_index(self) -> int | None:
        """The next required *physical* gate index, or ``None`` when idle/done."""
        if not self.running or self.finished:
            return None
        return self.track.gate_sequence[self.next_gate_order_index]

    @property
    def next_gate_direction(self) -> int | None:
        """Required direction of the next physical gate, or ``None`` when idle/done."""
        if not self.running or self.finished:
            return None
        return self.track.gate_directions[self.next_gate_order_index]

    @property
    def next_gate(self) -> Gate | None:
        """The next required :class:`~flightstack.race.geometry.Gate`, if any."""
        index = self.next_gate_index
        return None if index is None else self.track.gates[index]

    @property
    def gates_passed(self) -> int:
        """Total accepted gate-order entries in the active/completed run."""
        if self.lap == 0:
            return 0
        if self.finished:
            return self.total_laps * self.track.gate_passes_per_lap
        return (self.lap - 1) * self.track.gate_passes_per_lap + self.next_gate_order_index

    @property
    def n_gates_passed(self) -> int:
        """LSY-style alias for :attr:`gates_passed`."""
        return self.gates_passed

    @property
    def lap_time_s(self) -> float | None:
        """Elapsed time of the active lap at the latest processed race event."""
        if self.lap_started_at_s is None or self._last_event_time_s is None:
            return None
        return self._last_event_time_s - self.lap_started_at_s

    def reset(self, time_s: float = 0.0) -> tuple[Reset, ...]:
        """Clear all run progress and create a new event log rooted at ``Reset``."""
        event = Reset(time_s)
        self.running = False
        self.finished = False
        self.lap = 0
        self.next_gate_order_index = 0
        self.lap_started_at_s = None
        self.last_gate_at_s = None
        self.best_lap_s = None
        self.completed_lap_times_s = []
        self.collisions = 0
        self.events = [event]
        self._last_event_time_s = event.time_s
        self._repeat_gate_rearm_required = False
        return (event,)

    def start(self, time_s: float = 0.0) -> tuple[Start, ...]:
        """Start an idle run.  A completed run must be reset before restarting."""
        event = Start(time_s)
        if self.running or self.finished:
            return ()
        self._validate_event_time(event.time_s)
        self.running = True
        self.lap = 1
        self.next_gate_order_index = 0
        self.lap_started_at_s = event.time_s
        self.last_gate_at_s = None
        self._last_event_time_s = event.time_s
        self.events.append(event)
        return (event,)

    def apply_event(self, event: RaceEvent) -> tuple[RaceEvent, ...]:
        """Consume a command/observation event and return events actually recorded.

        ``Start``, ``Reset``, ``GatePassed``, and ``Collision`` are input events.
        ``LapCompleted`` and ``RaceFinished`` are emitted by the state machine
        and are intentionally not accepted as externally supplied commands.
        """
        if isinstance(event, Reset):
            return self.reset(event.time_s)
        if isinstance(event, Start):
            return self.start(event.time_s)
        if isinstance(event, GatePassed):
            return self._apply_gate_passed(event)
        if isinstance(event, Collision):
            return self._apply_collision(event)
        if isinstance(event, (LapCompleted, RaceFinished)):
            return ()
        raise TypeError("event must be a FlightStack race event")

    def update_position(
        self,
        previous_position_world_m: ArrayLike,
        current_position_world_m: ArrayLike,
        time_s: float,
        *,
        previous_time_s: float | None = None,
    ) -> tuple[RaceEvent, ...]:
        """Observe one physics segment and emit a gate event if it passes next.

        Supplying ``previous_time_s`` preserves exact interpolated gate timing.
        When it is omitted, the event uses ``time_s`` (the end of the segment).
        At most one ordered pass is accepted per call, even when a very long
        segment happens to intersect several physical gate planes.
        """
        current_time = _time_s(time_s)
        if previous_time_s is None:
            crossing_time = current_time
        else:
            prior_time = _time_s(previous_time_s, "previous_time_s")
            if prior_time > current_time:
                raise ValueError("previous_time_s must not exceed time_s")
            crossing_time = prior_time

        gate = self.next_gate
        direction = self.next_gate_direction
        if gate is None or direction is None:
            return ()

        # Consecutive identical order entries must not be satisfied twice by
        # replaying the segment that caused the first pass.  Rearm only after
        # the vehicle has returned to the required pre-crossing side.
        if self._repeat_gate_rearm_required:
            current_local = gate.local_coordinates(current_position_world_m)
            if float(current_local[0]) * direction < 0.0:
                self._repeat_gate_rearm_required = False
            return ()

        crossing = swept_gate_intersection(
            previous_position_world_m,
            current_position_world_m,
            gate,
            direction=direction,
        )
        if crossing is None:
            return ()
        if previous_time_s is not None:
            prior_time = _time_s(previous_time_s, "previous_time_s")
            crossing_time = prior_time + crossing.segment_fraction * (current_time - prior_time)
        event = self._gate_event_from_crossing(crossing, self.next_gate_index, crossing_time)
        return self._apply_gate_passed(event)

    def update(
        self,
        previous_position_world_m: ArrayLike,
        current_position_world_m: ArrayLike,
        time_s: float,
        *,
        previous_time_s: float | None = None,
    ) -> tuple[RaceEvent, ...]:
        """Alias for :meth:`update_position` used by compact simulation loops."""
        return self.update_position(
            previous_position_world_m,
            current_position_world_m,
            time_s,
            previous_time_s=previous_time_s,
        )

    def step(
        self,
        previous_position_world_m: ArrayLike,
        current_position_world_m: ArrayLike,
        time_s: float,
        *,
        previous_time_s: float | None = None,
    ) -> tuple[RaceEvent, ...]:
        """Simulation-loop alias for :meth:`update_position`."""
        return self.update_position(
            previous_position_world_m,
            current_position_world_m,
            time_s,
            previous_time_s=previous_time_s,
        )

    def record_collision(self, object_id: str, time_s: float) -> tuple[RaceEvent, ...]:
        """Record a collision while a race is running."""
        return self._apply_collision(Collision(object_id, time_s))

    def to_mapping(self) -> dict[str, object]:
        """Return a compact telemetry-safe view of current race progress."""
        return {
            "running": self.running,
            "finished": self.finished,
            "lap": self.lap,
            "total_laps": self.total_laps,
            "next_gate_index": self.next_gate_index,
            "next_gate_order_index": self.next_gate_order_index,
            "next_gate_direction": self.next_gate_direction,
            "lap_started_at_s": self.lap_started_at_s,
            "last_gate_at_s": self.last_gate_at_s,
            "best_lap_s": self.best_lap_s,
            "completed_lap_times_s": list(self.completed_lap_times_s),
            "collisions": self.collisions,
            "gates_passed": self.gates_passed,
        }

    def _apply_gate_passed(self, event: GatePassed) -> tuple[RaceEvent, ...]:
        if not self.running or self.finished:
            return ()
        expected_index = self.next_gate_index
        expected_direction = self.next_gate_direction
        if expected_index is None or expected_direction is None:
            return ()
        if event.gate_index != expected_index:
            return ()
        if event.direction is not None and event.direction != expected_direction:
            return ()
        self._validate_event_time(event.time_s)
        accepted = (
            event
            if event.direction is not None
            else replace(event, direction=expected_direction)
        )
        self.events.append(accepted)
        self.last_gate_at_s = accepted.time_s
        self._last_event_time_s = accepted.time_s
        self.next_gate_order_index += 1

        recorded: list[RaceEvent] = [accepted]
        if self.next_gate_order_index == self.track.gate_passes_per_lap:
            recorded.extend(self._complete_lap(accepted.time_s))
        else:
            self._set_repeat_gate_rearm_if_needed(accepted)
        return tuple(recorded)

    def _apply_collision(self, event: Collision) -> tuple[RaceEvent, ...]:
        if not self.running or self.finished:
            return ()
        self._validate_event_time(event.time_s)
        self.events.append(event)
        self.collisions += 1
        self._last_event_time_s = event.time_s
        return (event,)

    def _complete_lap(self, time_s: float) -> tuple[RaceEvent, ...]:
        if self.lap_started_at_s is None:
            raise RuntimeError("active race has no lap start timestamp")
        lap_time = time_s - self.lap_started_at_s
        completed = LapCompleted(lap=self.lap, lap_time_s=lap_time, time_s=time_s)
        self.events.append(completed)
        self.completed_lap_times_s.append(lap_time)
        self.best_lap_s = lap_time if self.best_lap_s is None else min(self.best_lap_s, lap_time)
        self._last_event_time_s = time_s

        recorded: list[RaceEvent] = [completed]
        if self.lap >= self.total_laps:
            self.running = False
            self.finished = True
            finished = RaceFinished(time_s)
            self.events.append(finished)
            recorded.append(finished)
            return tuple(recorded)

        self.lap += 1
        self.next_gate_order_index = 0
        self.lap_started_at_s = time_s
        self._repeat_gate_rearm_required = False
        return tuple(recorded)

    def _set_repeat_gate_rearm_if_needed(self, accepted: GatePassed) -> None:
        next_index = self.next_gate_index
        next_direction = self.next_gate_direction
        self._repeat_gate_rearm_required = (
            next_index == accepted.gate_index and next_direction == accepted.direction
        )

    def _gate_event_from_crossing(
        self,
        crossing: GateCrossing,
        gate_index: int | None,
        time_s: float,
    ) -> GatePassed:
        if gate_index is None:
            raise RuntimeError("race has no next gate")
        return GatePassed(
            gate_index=gate_index,
            crossing_position_world_m=crossing.position_world_m,
            time_s=time_s,
            direction=crossing.direction,
            segment_fraction=crossing.segment_fraction,
        )

    def _validate_event_time(self, time_s: float) -> None:
        if self._last_event_time_s is not None and time_s < self._last_event_time_s:
            raise ValueError("race event time cannot move backward")


__all__ = [
    "Collision",
    "GatePassed",
    "LapCompleted",
    "RaceEvent",
    "RaceFinished",
    "RaceState",
    "Reset",
    "Start",
]
