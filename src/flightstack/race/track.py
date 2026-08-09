"""Data-driven race-track definitions and JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from flightstack.race.geometry import Gate, Vector


def _positive_int(value: Any, name: str) -> int:
    """Validate a positive integer without accepting booleans or truncation."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result: int = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0 or result != value:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _finite_scalar(value: Any, name: str) -> float:
    """Validate a finite scalar."""
    try:
        result: float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector3(value: ArrayLike, name: str) -> Vector:
    """Validate a world vector stored on a track."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite vector with shape (3,)")
    result = vector.copy()
    result.setflags(write=False)
    return result


def _mapping_field(mapping: dict[str, Any], *names: str) -> Any:
    """Return the first present spelling of a JSON field."""
    for name in names:
        if name in mapping:
            return mapping[name]
    expected = " or ".join(repr(name) for name in names)
    raise ValueError(f"track entry is missing {expected}")


def _gate_from_mapping(mapping: object, index: int) -> Gate:
    """Build one canonical :class:`Gate` from a friendly JSON object."""
    if not isinstance(mapping, dict):
        raise ValueError(f"gates[{index}] must be an object")

    identifier = mapping.get("id", f"gate-{index + 1}")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"gates[{index}].id must be a nonempty string")
    center: Any = _mapping_field(mapping, "center_world_m", "position_world_m", "pos")
    normal: Any = _mapping_field(mapping, "normal_world", "normal")
    up: Any = mapping.get("up_world", mapping.get("up", [0.0, 0.0, 1.0]))
    right: Any = mapping.get("right_world", mapping.get("right"))
    center_vector = _vector3(center, f"gates[{index}].center_world_m")
    normal_vector = _vector3(normal, f"gates[{index}].normal_world")
    up_vector = _vector3(up, f"gates[{index}].up_world")
    if right is None:
        derived_right = np.cross(up_vector, normal_vector)
        if float(np.linalg.norm(derived_right)) <= 1e-12:
            raise ValueError(
                f"gates[{index}] needs right_world because up_world and normal_world are parallel"
        )
        right = derived_right
    right_vector = _vector3(right, f"gates[{index}].right_world")

    if "half_width_m" in mapping:
        half_width = mapping["half_width_m"]
    else:
        half_width = _finite_scalar(
            _mapping_field(mapping, "width_m", "width"), f"gates[{index}].width_m"
        ) / 2.0
    if "half_height_m" in mapping:
        half_height = mapping["half_height_m"]
    else:
        half_height = _finite_scalar(
            _mapping_field(mapping, "height_m", "height"), f"gates[{index}].height_m"
        ) / 2.0

    return Gate(
        center_world_m=center_vector,
        normal_world=normal_vector,
        right_world=right_vector,
        up_world=up_vector,
        half_width_m=half_width,
        half_height_m=half_height,
        gate_id=identifier,
        frame_thickness_m=mapping.get("frame_thickness_m", 0.08),
        frame_depth_m=mapping.get("frame_depth_m", 0.08),
    )


@dataclass(frozen=True)
class Track:
    """An ordered collection of physical gates and signed lap instructions.

    ``gate_order`` follows the LSY convention: it is a nonempty, signed,
    1-based sequence.  Positive values request a ``-normal -> +normal`` pass;
    negative values request the reverse.  The same physical gate may appear
    multiple times, including in consecutive order entries.
    """

    name: str
    gates: tuple[Gate, ...]
    gate_order: tuple[int, ...]
    default_laps: int = 1
    description: str = ""
    start_position_world_m: Vector | None = None
    ground_height_m: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("track name must be a nonempty string")
        gates = tuple(self.gates)
        if not gates:
            raise ValueError("track must contain at least one gate")
        if not all(isinstance(gate, Gate) for gate in gates):
            raise ValueError("gates must contain Gate instances")
        normalized_gates: list[Gate] = []
        identifiers: set[str] = set()
        for index, gate in enumerate(gates):
            current = gate if gate.gate_id else replace(gate, gate_id=f"gate-{index + 1}")
            if current.gate_id in identifiers:
                raise ValueError(f"gate ids must be unique; duplicate {current.gate_id!r}")
            identifiers.add(current.gate_id)
            normalized_gates.append(current)

        order = tuple(self.gate_order)
        if not order:
            raise ValueError("gate_order must be a nonempty signed 1-based sequence")
        parsed_order: list[int] = []
        for entry in order:
            if isinstance(entry, bool):
                raise ValueError("gate_order entries must be signed integers")
            try:
                value = int(entry)
            except (TypeError, ValueError) as exc:
                raise ValueError("gate_order entries must be signed integers") from exc
            if value != entry or value == 0 or abs(value) > len(normalized_gates):
                raise ValueError("gate_order entries must be nonzero valid signed gate ids")
            parsed_order.append(value)

        start = (
            None
            if self.start_position_world_m is None
            else _vector3(self.start_position_world_m, "start_position_world_m")
        )
        object.__setattr__(self, "gates", tuple(normalized_gates))
        object.__setattr__(self, "gate_order", tuple(parsed_order))
        object.__setattr__(self, "default_laps", _positive_int(self.default_laps, "default_laps"))
        object.__setattr__(self, "start_position_world_m", start)
        object.__setattr__(
            self,
            "ground_height_m",
            _finite_scalar(self.ground_height_m, "ground_height_m"),
        )

    @property
    def gate_sequence(self) -> tuple[int, ...]:
        """Configured physical gate indices, zero-based for Python callers."""
        return tuple(abs(entry) - 1 for entry in self.gate_order)

    @property
    def gate_directions(self) -> tuple[int, ...]:
        """Directed pass signs matching :attr:`gate_sequence`."""
        return tuple(1 if entry > 0 else -1 for entry in self.gate_order)

    @property
    def gate_sequence_direction(self) -> tuple[int, ...]:
        """LSY-style singular alias for :attr:`gate_directions`."""
        return self.gate_directions

    @property
    def gate_count(self) -> int:
        """Number of physical gate objects."""
        return len(self.gates)

    @property
    def gate_passes_per_lap(self) -> int:
        """Number of ordered gate passes needed to complete one lap."""
        return len(self.gate_order)

    def gate_for_order_index(self, order_index: int) -> Gate:
        """Return the physical gate required by one zero-based order entry."""
        if isinstance(order_index, bool) or not 0 <= int(order_index) < len(self.gate_order):
            raise IndexError("gate order index is out of range")
        return self.gates[self.gate_sequence[int(order_index)]]

    def direction_for_order_index(self, order_index: int) -> int:
        """Return the required direction for one zero-based order entry."""
        if isinstance(order_index, bool) or not 0 <= int(order_index) < len(self.gate_order):
            raise IndexError("gate order index is out of range")
        return self.gate_directions[int(order_index)]

    def to_mapping(self) -> dict[str, object]:
        """Return a JSON-safe canonical representation."""
        result: dict[str, object] = {
            "schema_version": 1,
            "name": self.name,
            "description": self.description,
            "default_laps": self.default_laps,
            "ground_height_m": self.ground_height_m,
            "gates": [gate.to_mapping() for gate in self.gates],
            "gate_order": list(self.gate_order),
        }
        if self.start_position_world_m is not None:
            result["start_position_world_m"] = self.start_position_world_m.tolist()
        return result

    @classmethod
    def from_mapping(cls, mapping: object) -> Track:
        """Parse a track mapping using the repository's JSON schema."""
        if not isinstance(mapping, dict):
            raise ValueError("track must be a JSON object")
        raw_gates = mapping.get("gates")
        if not isinstance(raw_gates, list):
            raise ValueError("track must contain a gates list")
        gates = tuple(_gate_from_mapping(gate, index) for index, gate in enumerate(raw_gates))
        raw_order: Any = mapping.get("gate_order")
        if not isinstance(raw_order, list):
            raise ValueError("track must contain a gate_order list")
        parsed_order: list[int] = []
        for entry in raw_order:
            if isinstance(entry, bool):
                raise ValueError("gate_order entries must be signed integers")
            try:
                parsed_entry: int = int(entry)
            except (TypeError, ValueError) as exc:
                raise ValueError("gate_order entries must be signed integers") from exc
            if parsed_entry != entry:
                raise ValueError("gate_order entries must be signed integers")
            parsed_order.append(parsed_entry)
        raw_start: Any = mapping.get("start_position_world_m")
        return cls(
            name=str(mapping.get("name", "unnamed-track")),
            gates=gates,
            gate_order=tuple(parsed_order),
            default_laps=_positive_int(
                mapping.get("default_laps", mapping.get("laps", 1)), "default_laps"
            ),
            description=str(mapping.get("description", "")),
            start_position_world_m=(
                None if raw_start is None else _vector3(raw_start, "start_position_world_m")
            ),
            ground_height_m=_finite_scalar(mapping.get("ground_height_m", 0.0), "ground_height_m"),
        )


def default_tracks_dir() -> Path:
    """Return the repository-owned directory containing data-driven tracks."""
    return Path(__file__).resolve().parents[3] / "tracks"


def load_track(path: str | Path) -> Track:
    """Load and validate a FlightStack race track JSON file.

    A bare name such as ``"technical-eight"`` resolves against the tracked
    :func:`default_tracks_dir`; an explicit relative or absolute JSON path is
    used as given.
    """
    candidate = Path(path)
    if candidate.suffix == "" and not candidate.exists():
        candidate = default_tracks_dir() / f"{candidate.name}.json"
    try:
        with candidate.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"track file not found: {candidate}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"track JSON is invalid: {candidate}: {exc.msg}") from exc
    return Track.from_mapping(data)


def load_technical_eight() -> Track:
    """Load FlightStack's included technical-eight reference course."""
    return load_track("technical-eight")


__all__ = ["Track", "default_tracks_dir", "load_technical_eight", "load_track"]
