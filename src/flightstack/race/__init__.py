"""Race geometry, data-driven tracks, and ordered lap state."""

from flightstack.race.geometry import (
    Gate,
    GateCrossing,
    gate_frame_collision,
    gate_passed,
    ground_collision,
    is_gate_frame_collision,
    is_ground_collision,
    swept_gate_crossing,
    swept_gate_intersection,
)
from flightstack.race.state import (
    Collision,
    GatePassed,
    LapCompleted,
    RaceEvent,
    RaceFinished,
    RaceState,
    Reset,
    Start,
)
from flightstack.race.track import Track, default_tracks_dir, load_technical_eight, load_track

__all__ = [
    "Collision",
    "Gate",
    "GateCrossing",
    "GatePassed",
    "LapCompleted",
    "RaceEvent",
    "RaceFinished",
    "RaceState",
    "Reset",
    "Start",
    "Track",
    "default_tracks_dir",
    "gate_frame_collision",
    "gate_passed",
    "ground_collision",
    "is_gate_frame_collision",
    "is_ground_collision",
    "load_technical_eight",
    "load_track",
    "swept_gate_crossing",
    "swept_gate_intersection",
]
