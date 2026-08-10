"""Headless canonical-physics race episodes and portable artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from flightstack.experiments.scenario import Scenario, ScenarioRealization
from flightstack.math.quaternion import from_euler
from flightstack.race import (
    RaceEvent,
    RaceState,
    Track,
    default_tracks_dir,
    gate_frame_collision,
    ground_collision,
    load_track,
)
from flightstack.runtime.pilots import Pilot, PilotKind
from flightstack.runtime.replay import ReplayRecorder
from flightstack.sim.vehicle import FixedStepRuntime, FlightState, PilotCommand, VehicleConfig

Vector = NDArray[np.float64]
Termination = Literal["finished", "crashed", "timeout"]


class PilotFactory(Protocol):
    """Create a fresh comparable pilot for one independent episode."""

    def __call__(self, vehicle: VehicleConfig) -> Pilot: ...


def _sha256_file(path: Path) -> str:
    """Return a full SHA-256 digest without loading an artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe_mapping(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    """Copy a mapping through canonical JSON so artifacts cannot contain opaque values."""
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded: object = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON-serializable") from exc
    if not isinstance(decoded, dict):  # Defensive: JSON objects always decode to dicts.
        raise ValueError(f"{name} must encode to a JSON object")
    return {str(key): item for key, item in decoded.items()}


def _mapping_sha256(value: Mapping[str, object]) -> str:
    """Hash the same canonical JSON form persisted in episode provenance."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _resolved_track_source_path(track: str) -> Path:
    """Mirror ``load_track`` bare-name resolution while preserving the source file identity."""
    candidate = Path(track)
    if candidate.suffix == "" and not candidate.exists():
        candidate = default_tracks_dir() / f"{candidate.name}.json"
    return candidate.resolve()


def _repository_identity() -> tuple[str | None, bool | None]:
    """Best-effort Git identity, intentionally harmless outside a checkout."""
    repository_root = Path(__file__).resolve().parents[3]
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None
    revision = revision_result.stdout.strip()
    if not revision:
        return None, None
    try:
        dirty_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return revision, None
    return revision, bool(dirty_result.stdout.strip())


def checkpoint_model_identity(checkpoint_path: str | Path) -> dict[str, object]:
    """Build an exact, portable provenance record for a learned checkpoint.

    This helper deliberately does not import Stable-Baselines3.  It can be used
    by a CLI or evaluation caller before constructing the learned pilot, and
    records both the model archive and its FlightStack metadata sidecar when
    present.
    """
    checkpoint = Path(checkpoint_path).expanduser()
    if checkpoint.suffix == "" and checkpoint.with_suffix(".zip").is_file():
        checkpoint = checkpoint.with_suffix(".zip")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {checkpoint}")
    result: dict[str, object] = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
    }
    metadata_path = checkpoint.with_suffix(".metadata.json")
    if metadata_path.is_file():
        result["metadata_path"] = str(metadata_path)
        result["metadata_sha256"] = _sha256_file(metadata_path)
    return result


def _automatic_pilot_model_identity(pilot: Pilot) -> dict[str, object] | None:
    """Record learned-policy metadata without making experiments depend on the AI extra."""
    if pilot.kind is not PilotKind.LEARNED:
        return None
    metadata: object = getattr(pilot, "metadata", None)
    serializer: object = getattr(metadata, "to_mapping", None)
    if not callable(serializer):
        return None
    serialized_metadata: object = serializer()
    if not isinstance(serialized_metadata, Mapping):
        return None
    policy: object = getattr(pilot, "policy", pilot)
    return {
        "policy_type": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "policy_metadata": _json_safe_mapping(serialized_metadata, name="policy metadata"),
    }


def _pilot_model_identity(
    pilot: Pilot,
    supplied: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    """Merge caller-supplied checkpoint identity with available policy metadata."""
    automatic = _automatic_pilot_model_identity(pilot)
    if supplied is None:
        return automatic
    if not isinstance(supplied, Mapping):
        raise TypeError("pilot_model_identity must be a mapping when supplied")
    identity = _json_safe_mapping(supplied, name="pilot_model_identity")
    if automatic is not None:
        return {**automatic, **identity}
    return identity


def _event_mapping(event: RaceEvent) -> dict[str, object]:
    result: dict[str, object] = {"type": type(event).__name__}
    for name, value in vars(event).items():
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            result[name] = value.item()
        else:
            result[name] = value
    return result


def _initial_state(config: VehicleConfig, track: Track) -> FlightState:
    """Start at the configured course location facing its first required gate."""
    start = (
        np.array([0.0, 0.0, 1.2], dtype=np.float64)
        if track.start_position_world_m is None
        else np.asarray(track.start_position_world_m, dtype=np.float64)
    )
    target = track.gate_for_order_index(0).center_world_m
    heading = np.asarray(target - start, dtype=np.float64)
    yaw = float(np.arctan2(heading[1], heading[0])) if np.linalg.norm(heading[:2]) > 1e-12 else 0.0
    return FlightState(
        sim_time_s=0.0,
        position_world_m=start,
        velocity_world_m_s=np.zeros(3, dtype=np.float64),
        q_body_to_world_wxyz=from_euler(0.0, 0.0, yaw),
        body_rate_rad_s=np.zeros(3, dtype=np.float64),
        motor_thrust_n=np.full(4, config.hover_thrust_n / 4.0, dtype=np.float64),
    )


def _collision_object(state: FlightState, track: Track, vehicle_radius_m: float) -> str | None:
    if ground_collision(
        state.position_world_m,
        vehicle_radius_m=vehicle_radius_m,
        ground_height_m=track.ground_height_m,
    ):
        return "ground"
    for gate in track.gates:
        if gate_frame_collision(
            state.position_world_m,
            gate,
            vehicle_radius_m=vehicle_radius_m,
        ):
            return f"gate-frame:{gate.gate_id}"
    return None


@dataclass(frozen=True)
class EpisodeProvenance:
    """Inputs and implementation identity needed to interpret an episode."""

    implementation: str
    pilot_name: str
    pilot_kind: PilotKind
    track_name: str
    vehicle_name: str
    vehicle_version: str
    vehicle_config_hash: str
    physics_dt_s: float
    scenario: ScenarioRealization
    scenario_definition: Mapping[str, object] = field(default_factory=dict)
    scenario_definition_sha256: str | None = None
    track_source_path: str | None = None
    track_content_sha256: str | None = None
    git_revision: str | None = None
    git_dirty: bool | None = None
    pilot_model_identity: Mapping[str, object] | None = None

    def to_mapping(self) -> dict[str, object]:
        return {
            "implementation": self.implementation,
            "pilot_name": self.pilot_name,
            "pilot_kind": self.pilot_kind.value,
            "track_name": self.track_name,
            "vehicle_name": self.vehicle_name,
            "vehicle_version": self.vehicle_version,
            "vehicle_config_hash": self.vehicle_config_hash,
            "physics_dt_s": self.physics_dt_s,
            "scenario": self.scenario.to_mapping(),
            "scenario_definition": dict(self.scenario_definition),
            "scenario_definition_sha256": self.scenario_definition_sha256,
            "track_source_path": self.track_source_path,
            "track_content_sha256": self.track_content_sha256,
            "git_revision": self.git_revision,
            "git_dirty": self.git_dirty,
            "pilot_model_identity": (
                None if self.pilot_model_identity is None else dict(self.pilot_model_identity)
            ),
        }


@dataclass(frozen=True)
class EpisodeMetrics:
    """Small, comparable outcome metrics for one bounded run."""

    termination: Termination
    elapsed_time_s: float
    completed: bool
    lap_time_s: float | None
    gates_passed: int
    collisions: int
    distance_travelled_m: float
    mean_speed_m_s: float
    max_speed_m_s: float
    max_body_rate_rad_s: float
    mixer_saturated_steps: int
    final_distance_to_next_gate_m: float | None

    def to_mapping(self) -> dict[str, object]:
        return {
            "termination": self.termination,
            "elapsed_time_s": self.elapsed_time_s,
            "completed": self.completed,
            "lap_time_s": self.lap_time_s,
            "gates_passed": self.gates_passed,
            "collisions": self.collisions,
            "distance_travelled_m": self.distance_travelled_m,
            "mean_speed_m_s": self.mean_speed_m_s,
            "max_speed_m_s": self.max_speed_m_s,
            "max_body_rate_rad_s": self.max_body_rate_rad_s,
            "mixer_saturated_steps": self.mixer_saturated_steps,
            "final_distance_to_next_gate_m": self.final_distance_to_next_gate_m,
        }


@dataclass(frozen=True)
class TelemetrySample:
    """A sampled authoritative state plus its shared CTBR command and race view."""

    state: FlightState
    command: PilotCommand
    race: Mapping[str, object]
    events: tuple[Mapping[str, object], ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "state": self.state.to_mapping(),
            "pilot_command": {
                "collective_thrust_n": self.command.collective_thrust_n,
                "body_rate_rad_s": self.command.body_rate_rad_s.tolist(),
            },
            "race": dict(self.race),
            "events": [dict(event) for event in self.events],
        }


@dataclass(frozen=True)
class EpisodeArtifacts:
    """Paths produced by :meth:`EpisodeResult.write_artifacts`."""

    summary_path: Path
    telemetry_path: Path
    replay_path: Path


@dataclass(frozen=True)
class EpisodeResult:
    """Structured result carrying metrics, samples, events, and replay JSON."""

    provenance: EpisodeProvenance
    metrics: EpisodeMetrics
    telemetry: tuple[TelemetrySample, ...]
    events: tuple[Mapping[str, object], ...]
    replay: Mapping[str, object]

    def to_mapping(self) -> dict[str, object]:
        return {
            "provenance": self.provenance.to_mapping(),
            "metrics": self.metrics.to_mapping(),
            "events": [dict(event) for event in self.events],
        }

    def write_artifacts(self, destination: str | Path) -> EpisodeArtifacts:
        """Write JSON result, sampled telemetry, and replay into ``destination``."""
        directory = Path(destination)
        directory.mkdir(parents=True, exist_ok=True)
        summary_path = directory / "result.json"
        telemetry_path = directory / "telemetry.json"
        replay_path = directory / "replay.json"
        summary_path.write_text(
            json.dumps(self.to_mapping(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        telemetry_path.write_text(
            json.dumps([sample.to_mapping() for sample in self.telemetry], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        replay_path.write_text(
            json.dumps(self.replay, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return EpisodeArtifacts(summary_path, telemetry_path, replay_path)


def run_episode(
    scenario: Scenario,
    pilot_factory: PilotFactory,
    *,
    pilot_name: str | None = None,
    vehicle_config: VehicleConfig | None = None,
    pilot_model_identity: Mapping[str, object] | None = None,
) -> EpisodeResult:
    """Run one bounded fixed-step episode through the canonical CTBR seam.

    The exact same ``PilotCommand -> rate controller -> mixer -> motor -> 6DOF``
    path is used for each pilot.  Gate progress uses swept crossings and the
    same ground/gate-frame collision predicates as the interactive server.
    """
    if not isinstance(scenario, Scenario):
        raise TypeError("scenario must be a Scenario")
    config = VehicleConfig.from_toml() if vehicle_config is None else vehicle_config
    if not isinstance(config, VehicleConfig):
        raise TypeError("vehicle_config must be a VehicleConfig")
    track_source = _resolved_track_source_path(scenario.track)
    track = load_track(track_source)
    scenario_definition = _json_safe_mapping(scenario.to_mapping(), name="scenario definition")
    realization = scenario.realize()
    initial_state = _initial_state(config, track)
    runtime = FixedStepRuntime(config, dt=scenario.physics_dt_s, state=initial_state)
    race = RaceState(track, total_laps=scenario.laps)
    initial_events = race.reset(initial_state.sim_time_s) + race.start(initial_state.sim_time_s)
    pilot = pilot_factory(config)
    pilot.reset(initial_state)
    resolved_name = pilot.kind.value if pilot_name is None else pilot_name
    if not resolved_name:
        raise ValueError("pilot_name must be nonempty")
    git_revision, git_dirty = _repository_identity()
    provenance = EpisodeProvenance(
        implementation="flightstack-python-reference-6dof",
        pilot_name=resolved_name,
        pilot_kind=pilot.kind,
        track_name=track.name,
        vehicle_name=config.name,
        vehicle_version=config.version,
        vehicle_config_hash=config.config_hash,
        physics_dt_s=scenario.physics_dt_s,
        scenario=realization,
        scenario_definition=scenario_definition,
        scenario_definition_sha256=_mapping_sha256(scenario_definition),
        track_source_path=str(track_source),
        track_content_sha256=_sha256_file(track_source),
        git_revision=git_revision,
        git_dirty=git_dirty,
        pilot_model_identity=_pilot_model_identity(pilot, pilot_model_identity),
    )
    recorder = ReplayRecorder({"provenance": provenance.to_mapping()})
    all_events: list[Mapping[str, object]] = [_event_mapping(event) for event in initial_events]
    samples: list[TelemetrySample] = []
    next_telemetry_s = 0.0
    travelled_m = 0.0
    speed_sum = 0.0
    max_speed = 0.0
    max_body_rate = 0.0
    saturated_steps = 0
    crashed = False

    for _ in range(scenario.physics_steps):
        previous = runtime.state
        command = pilot.command(previous, race, scenario.physics_dt_s)
        current, mixed, _terms = runtime.step(command, realization.disturbance)
        travelled_m += float(np.linalg.norm(current.position_world_m - previous.position_world_m))
        speed = float(np.linalg.norm(current.velocity_world_m_s))
        speed_sum += speed
        max_speed = max(max_speed, speed)
        max_body_rate = max(max_body_rate, float(np.linalg.norm(current.body_rate_rad_s)))
        saturated_steps += int(mixed.saturated)

        collision = _collision_object(current, track, scenario.vehicle_radius_m)
        if collision is not None:
            crashed = True
            # Collision has precedence over a same-tick swept gate crossing.
            # Otherwise a final-gate finish marks the race finished first and
            # RaceState correctly rejects the subsequent collision event,
            # leaving a crashed episode with zero recorded collisions.
            new_events = list(race.record_collision(collision, current.sim_time_s))
        else:
            new_events = list(
                race.update(
                    previous.position_world_m,
                    current.position_world_m,
                    current.sim_time_s,
                    previous_time_s=previous.sim_time_s,
                )
            )
        mapped_events = tuple(_event_mapping(event) for event in new_events)
        all_events.extend(mapped_events)

        if current.sim_time_s + 1e-12 >= next_telemetry_s:
            race_snapshot = race.to_mapping()
            sample = TelemetrySample(current, command, race_snapshot, mapped_events)
            samples.append(sample)
            recorder.record(
                current,
                pilot.kind,
                command,
                race=race_snapshot,
                events=mapped_events,
            )
            while next_telemetry_s <= current.sim_time_s + 1e-12:
                next_telemetry_s += scenario.telemetry_period_s
        if crashed or race.finished:
            break

    final_state = runtime.state
    if not samples or samples[-1].state.sim_time_s < final_state.sim_time_s:
        final_command = pilot.command(final_state, race, scenario.physics_dt_s)
        final_sample = TelemetrySample(final_state, final_command, race.to_mapping(), ())
        samples.append(final_sample)
        recorder.record(
            final_state,
            pilot.kind,
            final_command,
            race=final_sample.race,
        )
    termination: Termination = "crashed" if crashed else "finished" if race.finished else "timeout"
    completed = termination == "finished"
    next_gate = race.next_gate
    final_distance = (
        None
        if next_gate is None
        else float(np.linalg.norm(final_state.position_world_m - next_gate.center_world_m))
    )
    steps_taken = max(1, int(round(final_state.sim_time_s / scenario.physics_dt_s)))
    metrics = EpisodeMetrics(
        termination=termination,
        elapsed_time_s=final_state.sim_time_s,
        completed=completed,
        lap_time_s=race.best_lap_s if completed else None,
        gates_passed=race.gates_passed,
        collisions=race.collisions,
        distance_travelled_m=travelled_m,
        mean_speed_m_s=speed_sum / steps_taken,
        max_speed_m_s=max_speed,
        max_body_rate_rad_s=max_body_rate,
        mixer_saturated_steps=saturated_steps,
        final_distance_to_next_gate_m=final_distance,
    )
    return EpisodeResult(
        provenance=provenance,
        metrics=metrics,
        telemetry=tuple(samples),
        events=tuple(all_events),
        replay=recorder.to_mapping(),
    )


__all__ = [
    "EpisodeArtifacts",
    "EpisodeMetrics",
    "EpisodeProvenance",
    "EpisodeResult",
    "PilotFactory",
    "TelemetrySample",
    "checkpoint_model_identity",
    "run_episode",
]
