"""Shared pilot contracts and responsive, frame-safe human input shaping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from flightstack.sim.vehicle import FlightState, PilotCommand, VehicleConfig

Vector = NDArray[np.float64]


class PilotKind(StrEnum):
    """The comparable high-level pilots supported by FlightStack."""

    HUMAN = "human"
    CLASSICAL = "classical"
    LEARNED = "learned"


class RaceView(Protocol):
    """Minimal race information pilots may consume without owning race state."""

    @property
    def next_gate_index(self) -> int | None:
        """Return the next physical gate, or ``None`` while the run is idle."""


class Pilot(Protocol):
    """Every pilot emits the same CTBR command at the same control seam."""

    kind: PilotKind

    def reset(self, initial_state: FlightState) -> None: ...

    def command(self, state: FlightState, race: RaceView, dt: float) -> PilotCommand: ...


def deadzone_expo(value: float, *, deadzone: float, expo: float) -> float:
    """Normalize a signed stick with a continuous deadzone and cubic expo.

    ``expo=0`` is linear after the deadzone; ``expo=1`` is fully cubic.  This
    function is deliberately shared by keyboard/gamepad input and unit-tested
    separately from browser event handling.
    """
    scalar = float(value)
    if not np.isfinite(scalar) or not np.isfinite(deadzone) or not np.isfinite(expo):
        raise ValueError("stick value, deadzone, and expo must be finite")
    if not 0.0 <= deadzone < 1.0:
        raise ValueError("deadzone must be in [0, 1)")
    if not 0.0 <= expo <= 1.0:
        raise ValueError("expo must be in [0, 1]")
    clipped = float(np.clip(scalar, -1.0, 1.0))
    magnitude = abs(clipped)
    if magnitude <= deadzone:
        return 0.0
    normalized = (magnitude - deadzone) / (1.0 - deadzone)
    curved = (1.0 - expo) * normalized + expo * normalized**3
    return float(np.copysign(curved, clipped))


def hover_centered_collective(throttle: float, config: VehicleConfig) -> float:
    """Map a human throttle in ``[0, 1]`` with 0.5 at physical hover.

    This avoids coupling manual-stick feel to an arbitrary per-motor maximum:
    zero remains disarmed/minimum thrust, midpoint reliably means level hover,
    and the upper half spans the remaining available collective authority.
    """
    value = float(throttle)
    if not np.isfinite(value):
        raise ValueError("throttle must be finite")
    value = float(np.clip(value, 0.0, 1.0))
    total_max = 4.0 * config.motor_max_thrust_n
    hover = config.hover_thrust_n
    if value <= 0.5:
        return 2.0 * value * hover
    return hover + 2.0 * (value - 0.5) * (total_max - hover)


@dataclass(frozen=True)
class ManualInput:
    """Normalized Mode-2-style human controls after browser/device capture."""

    throttle: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("throttle", self.throttle),
            ("roll", self.roll),
            ("pitch", self.pitch),
            ("yaw", self.yaw),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class ManualControlConfig:
    deadzone: float = 0.07
    expo: float = 0.32
    roll_rate_scale: float = 1.0
    pitch_rate_scale: float = 1.0
    yaw_rate_scale: float = 0.82

    def __post_init__(self) -> None:
        if not 0.0 <= self.deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        if not 0.0 <= self.expo <= 1.0:
            raise ValueError("expo must be in [0, 1]")
        if min(self.roll_rate_scale, self.pitch_rate_scale, self.yaw_rate_scale) <= 0.0:
            raise ValueError("manual rate scales must be positive")


class HumanPilot:
    """Mutable human pilot that maps shaped input to the shared CTBR contract."""

    kind = PilotKind.HUMAN

    def __init__(
        self,
        vehicle: VehicleConfig,
        controls: ManualControlConfig | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.controls = ManualControlConfig() if controls is None else controls
        self.input = ManualInput()

    def set_input(self, input_state: ManualInput) -> None:
        self.input = input_state

    def reset(self, initial_state: FlightState) -> None:
        del initial_state
        self.input = ManualInput()

    def command(self, state: FlightState, race: RaceView, dt: float) -> PilotCommand:
        del state, race, dt
        shaped = np.array(
            [
                deadzone_expo(
                    self.input.roll, deadzone=self.controls.deadzone, expo=self.controls.expo
                ),
                deadzone_expo(
                    self.input.pitch, deadzone=self.controls.deadzone, expo=self.controls.expo
                ),
                deadzone_expo(
                    self.input.yaw, deadzone=self.controls.deadzone, expo=self.controls.expo
                ),
            ],
            dtype=np.float64,
        )
        rates: Vector = np.array(
            [
                shaped[0] * self.vehicle.max_body_rate_rad_s[0] * self.controls.roll_rate_scale,
                shaped[1] * self.vehicle.max_body_rate_rad_s[1] * self.controls.pitch_rate_scale,
                shaped[2] * self.vehicle.max_body_rate_rad_s[2] * self.controls.yaw_rate_scale,
            ],
            dtype=np.float64,
        )
        return PilotCommand(
            collective_thrust_n=hover_centered_collective(self.input.throttle, self.vehicle),
            body_rate_rad_s=rates,
        )
