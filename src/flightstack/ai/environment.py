"""Authoritative fixed-step Gymnasium-compatible FlightStack race environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from flightstack.ai.actions import action_to_command, normalized_action
from flightstack.ai.config import EnvironmentConfig, RacingAIConfig, load_racing_ai_config
from flightstack.ai.errors import TRAIN_EXTRA_COMMAND, OptionalTrainingDependencyError
from flightstack.ai.observation import (
    OBSERVATION_DIMENSION,
    build_observation,
    distance_to_next_gate,
)
from flightstack.ai.reward import RewardTerms, race_reward
from flightstack.ai.spaces import BoxSpace
from flightstack.math.quaternion import from_euler
from flightstack.race import (
    GatePassed,
    RaceEvent,
    RaceState,
    Track,
    gate_frame_collision,
    ground_collision,
)
from flightstack.race.track import load_technical_eight
from flightstack.sim.vehicle import FixedStepRuntime, FlightState, VehicleConfig


def _event_mapping(event: RaceEvent) -> dict[str, object]:
    result: dict[str, object] = {"type": type(event).__name__}
    for name, value in vars(event).items():
        if isinstance(value, np.ndarray):
            result[name] = value.tolist()
        elif isinstance(value, np.floating | np.integer):
            result[name] = value.item()
        else:
            result[name] = value
    return result


@dataclass(frozen=True)
class EpisodeResult:
    """Small state summary useful to non-Gymnasium evaluators."""

    terminated: bool
    truncated: bool
    reason: str | None


class FlightStackRaceEnv:
    """State-based one-vehicle race task using FlightStack's real control seam.

    The core class follows Gymnasium's ``reset``/``step`` return contract while
    depending only on NumPy.  Install ``.[train]`` and call
    :func:`make_gymnasium_env` to obtain a native ``gymnasium.Env`` adapter for
    Stable-Baselines3.  Each action advances the existing chain exactly:

    ``normalized action -> PilotCommand (CTBR) -> rate PID -> mixer -> motors -> 6DOF``.
    """

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(
        self,
        *,
        vehicle: VehicleConfig | None = None,
        track: Track | None = None,
        ai_config: RacingAIConfig | None = None,
    ) -> None:
        self.vehicle = VehicleConfig.from_toml() if vehicle is None else vehicle
        self.track = load_technical_eight() if track is None else track
        self.ai_config = load_racing_ai_config() if ai_config is None else ai_config
        self.environment: EnvironmentConfig = self.ai_config.environment
        self.action_space = BoxSpace(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = BoxSpace(-1.0, 1.0, (OBSERVATION_DIMENSION,), dtype=np.float32)
        self._rng = np.random.default_rng()
        self._seed: int | None = None
        initial = self._initial_state()
        self.runtime = FixedStepRuntime(
            self.vehicle,
            dt=self.environment.physics_dt_s,
            state=initial,
        )
        self.race = RaceState(self.track)
        self._previous_action = np.zeros(4, dtype=np.float64)
        self._previous_distance_m = 0.0
        self._episode_steps = 0
        self._episode_result = EpisodeResult(False, False, None)

    @property
    def state(self) -> FlightState:
        """Return the canonical reference state without exposing mutable internals."""
        return self.runtime.state

    @property
    def episode_result(self) -> EpisodeResult:
        """Return the current end-state flags in a simple typed form."""
        return self._episode_result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, object]]:
        """Reset deterministically; identical seeds produce identical starts."""
        if options not in (None, {}):
            raise ValueError("FlightStackRaceEnv does not support reset options")
        if seed is not None:
            if isinstance(seed, bool):
                raise ValueError("seed must be an integer or None")
            self._seed = int(seed)
            self._rng = np.random.default_rng(self._seed)
        initial = self._initial_state()
        self.runtime.reset(initial)
        events = self.race.reset(0.0) + self.race.start(0.0)
        self._previous_action = np.zeros(4, dtype=np.float64)
        self._previous_distance_m = distance_to_next_gate(self.state, self.race)
        self._episode_steps = 0
        self._episode_result = EpisodeResult(False, False, None)
        observation = self._observation()
        return observation, self._info(events=events, reward_terms=None)

    def step(
        self,
        action: ArrayLike,
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
        """Apply one normalized CTBR action for a fixed number of physics ticks."""
        if self._episode_result.terminated or self._episode_result.truncated:
            raise RuntimeError("episode is complete; call reset() before step()")
        normalized = normalized_action(action)
        command = action_to_command(normalized, self.vehicle)
        events: tuple[RaceEvent, ...] = ()
        out_of_bounds = False
        reason: str | None = None
        for _ in range(self.environment.control_substeps):
            previous = self.state
            current, _mixed, _terms = self.runtime.step(command)
            events += self.race.update_position(
                previous.position_world_m,
                current.position_world_m,
                current.sim_time_s,
                previous_time_s=previous.sim_time_s,
            )
            collision_id = self._collision_id(current)
            if collision_id is not None:
                events += self.race.record_collision(collision_id, current.sim_time_s)
                reason = collision_id
                break
            if self._outside_course(current):
                out_of_bounds = True
                reason = "out_of_bounds"
                break
            if self.race.finished:
                reason = "finished"
                break

        state = self.state
        gate_passed = any(isinstance(event, GatePassed) for event in events)
        current_distance = distance_to_next_gate(state, self.race)
        progress_distance = 0.0 if gate_passed or self.race.finished else current_distance
        reward_terms = race_reward(
            previous_distance_m=self._previous_distance_m,
            current_distance_m=progress_distance,
            previous_action=self._previous_action,
            action=normalized,
            body_rate_rad_s=state.body_rate_rad_s,
            max_body_rate_rad_s=self.vehicle.max_body_rate_rad_s,
            events=events,
            out_of_bounds=out_of_bounds,
            control_dt_s=self.environment.control_dt_s,
            config=self.ai_config.reward,
        )
        self._previous_action = normalized
        self._previous_distance_m = current_distance
        self._episode_steps += 1
        terminated = reason is not None
        truncated = False
        if not terminated and state.sim_time_s >= self.environment.max_episode_s:
            truncated = True
            reason = "time_limit"
        self._episode_result = EpisodeResult(terminated, truncated, reason)
        observation = self._observation()
        return (
            observation,
            reward_terms.total,
            terminated,
            truncated,
            self._info(events=events, reward_terms=reward_terms),
        )

    def close(self) -> None:
        """Provide the standard environment close hook; no external resources exist."""

    def _initial_state(self) -> FlightState:
        start = (
            np.array([0.0, 0.0, 1.2], dtype=np.float64)
            if self.track.start_position_world_m is None
            else np.asarray(self.track.start_position_world_m, dtype=np.float64).copy()
        )
        horizontal_jitter = self._rng.uniform(
            -self.environment.initial_xy_jitter_m,
            self.environment.initial_xy_jitter_m,
            size=2,
        )
        altitude_jitter = self._rng.uniform(
            -self.environment.initial_altitude_jitter_m,
            self.environment.initial_altitude_jitter_m,
        )
        start[:2] += horizontal_jitter
        start[2] += altitude_jitter
        first_gate = self.track.gate_for_order_index(0)
        heading = first_gate.center_world_m - start
        nominal_yaw = float(np.arctan2(heading[1], heading[0]))
        yaw = nominal_yaw + float(
            self._rng.uniform(
                -self.environment.initial_yaw_jitter_rad,
                self.environment.initial_yaw_jitter_rad,
            )
        )
        return FlightState(
            sim_time_s=0.0,
            position_world_m=start,
            velocity_world_m_s=np.zeros(3, dtype=np.float64),
            q_body_to_world_wxyz=from_euler(0.0, 0.0, yaw),
            body_rate_rad_s=np.zeros(3, dtype=np.float64),
            motor_thrust_n=np.full(4, self.vehicle.hover_thrust_n / 4.0, dtype=np.float64),
        )

    def _observation(self) -> NDArray[np.float32]:
        return build_observation(
            self.state,
            self.race,
            self.vehicle,
            self.ai_config.observation,
            self._previous_action,
        )

    def _collision_id(self, state: FlightState) -> str | None:
        if ground_collision(
            state.position_world_m,
            vehicle_radius_m=self.environment.vehicle_radius_m,
            ground_height_m=self.track.ground_height_m,
        ):
            return "ground"
        for gate in self.track.gates:
            if gate_frame_collision(
                state.position_world_m,
                gate,
                vehicle_radius_m=self.environment.vehicle_radius_m,
            ):
                return f"gate-frame:{gate.gate_id}"
        return None

    def _outside_course(self, state: FlightState) -> bool:
        horizontal_distance = float(np.linalg.norm(state.position_world_m[:2]))
        return bool(
            horizontal_distance > self.environment.max_distance_from_course_m
            or state.position_world_m[2] > self.environment.max_altitude_m
        )

    def _info(
        self,
        *,
        events: tuple[RaceEvent, ...],
        reward_terms: RewardTerms | None,
    ) -> dict[str, object]:
        state = self.state
        command = action_to_command(self._previous_action, self.vehicle)
        return {
            "seed": self._seed,
            "episode_steps": self._episode_steps,
            "sim_time_s": state.sim_time_s,
            "state": state.to_mapping(),
            "race": self.race.to_mapping(),
            "events": [_event_mapping(event) for event in events],
            "normalized_action": self._previous_action.astype(np.float32).tolist(),
            "pilot_command": {
                "collective_thrust_n": command.collective_thrust_n,
                "body_rate_rad_s": command.body_rate_rad_s.tolist(),
            },
            "reward_terms": None if reward_terms is None else reward_terms.to_mapping(),
            "termination_reason": self._episode_result.reason,
        }


def make_gymnasium_env(
    *,
    vehicle: VehicleConfig | None = None,
    track: Track | None = None,
    ai_config: RacingAIConfig | None = None,
) -> object:
    """Return a native Gymnasium adapter when the optional training extra exists.

    Keeping this import lazy means simulator/manual users do not pull PyTorch,
    Gymnasium, or Stable-Baselines3 into the authoritative runtime.
    """
    try:
        from gymnasium import Env, spaces  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise OptionalTrainingDependencyError(
            "Gymnasium support is optional; install it with " f"`{TRAIN_EXTRA_COMMAND}`."
        ) from exc

    core = FlightStackRaceEnv(vehicle=vehicle, track=track, ai_config=ai_config)

    class GymnasiumAdapter(Env):  # type: ignore[misc]
        metadata = FlightStackRaceEnv.metadata

        def __init__(self) -> None:
            super().__init__()
            self.core = core
            self.action_space = spaces.Box(
                low=core.action_space.low,
                high=core.action_space.high,
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=core.observation_space.low,
                high=core.observation_space.high,
                dtype=np.float32,
            )

        def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, object] | None = None,
        ) -> tuple[NDArray[np.float32], dict[str, object]]:
            super().reset(seed=seed)
            return self.core.reset(seed=seed, options=options)

        def step(
            self, action: ArrayLike
        ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, object]]:
            return self.core.step(action)

        def close(self) -> None:
            self.core.close()

    return GymnasiumAdapter()


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Normalize a checkpoint path for cross-platform user-facing diagnostics."""
    return Path(path).expanduser().resolve()
