"""Regression coverage for FlightStack's data-driven race subsystem."""

from __future__ import annotations

import json

import numpy as np
import pytest

from flightstack.race import (
    Collision,
    Gate,
    GatePassed,
    LapCompleted,
    RaceFinished,
    RaceState,
    Track,
    gate_frame_collision,
    gate_passed,
    ground_collision,
    load_technical_eight,
    load_track,
    swept_gate_crossing,
    swept_gate_intersection,
)


def axis_gate(
    *,
    gate_id: str = "gate",
    center: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> Gate:
    """Create a gate whose forward axis is world +X."""
    return Gate(
        center_world_m=center,
        normal_world=[1.0, 0.0, 0.0],
        right_world=[0.0, 1.0, 0.0],
        up_world=[0.0, 0.0, 1.0],
        half_width_m=1.0,
        half_height_m=1.0,
        gate_id=gate_id,
        frame_thickness_m=0.1,
        frame_depth_m=0.1,
    )


def crossing_segment(gate: Gate, *, distance: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return the centerline segment for a forward pass of ``gate``."""
    return (
        np.asarray(gate.center_world_m - distance * gate.normal_world),
        np.asarray(gate.center_world_m + distance * gate.normal_world),
    )


def two_gate_track(*, gate_order: tuple[int, ...] = (1, 2)) -> Track:
    """A compact deterministic track for state-machine tests."""
    return Track(
        name="two-gate",
        gates=(axis_gate(gate_id="one"), axis_gate(gate_id="two", center=(5.0, 0.0, 1.0))),
        gate_order=gate_order,
    )


def test_swept_crossing_returns_exact_centerline_intersection() -> None:
    gate = axis_gate()
    crossing = swept_gate_crossing([-2.0, 0.2, 1.4], [3.0, 0.2, 1.4], gate)

    assert crossing is not None
    assert np.allclose(crossing, [0.0, 0.2, 1.4])
    assert gate_passed([-2.0, 0.2, 1.4], [3.0, 0.2, 1.4], gate)


def test_tilted_gate_crossing_uses_gate_local_coordinates() -> None:
    root_half = np.sqrt(0.5)
    gate = Gate(
        center_world_m=[4.0, -2.0, 1.5],
        normal_world=[root_half, root_half, 0.0],
        right_world=[-root_half, root_half, 0.0],
        up_world=[0.0, 0.0, 1.0],
        half_width_m=0.7,
        half_height_m=0.6,
        gate_id="tilted",
    )
    previous = gate.world_coordinates([-2.0, 0.45, -0.2])
    current = gate.world_coordinates([3.0, 0.45, -0.2])

    crossing = swept_gate_intersection(previous, current, gate)

    assert crossing is not None
    assert crossing.direction == 1
    assert crossing.segment_fraction == pytest.approx(0.4)
    assert np.allclose(gate.local_coordinates(crossing.position_world_m), [0.0, 0.45, -0.2])


def test_high_speed_crossing_does_not_depend_on_endpoint_proximity() -> None:
    gate = axis_gate()
    crossing = swept_gate_intersection([-250.0, 0.95, 1.95], [300.0, 0.95, 1.95], gate)

    assert crossing is not None
    assert crossing.position_world_m[0] == pytest.approx(0.0)
    assert crossing.segment_fraction == pytest.approx(250.0 / 550.0)


@pytest.mark.parametrize(
    ("offset", "should_pass"),
    [
        ((1.0, 1.0), True),
        ((1.0 + 1e-6, 0.0), False),
        ((0.0, 1.0 + 1e-6), False),
    ],
)
def test_aperture_boundary_is_inclusive_but_outside_is_rejected(
    offset: tuple[float, float], should_pass: bool
) -> None:
    gate = axis_gate()
    right, up = offset
    previous = [-2.0, right, 1.0 + up]
    current = [2.0, right, 1.0 + up]

    assert gate_passed(previous, current, gate) is should_pass


def test_wrong_direction_and_non_crossing_segments_are_rejected() -> None:
    gate = axis_gate()

    assert swept_gate_crossing([2.0, 0.0, 1.0], [-2.0, 0.0, 1.0], gate) is None
    reverse = swept_gate_crossing([2.0, 0.0, 1.0], [-2.0, 0.0, 1.0], gate, direction=-1)
    assert reverse is not None
    assert swept_gate_crossing([-2.0, 0.0, 1.0], [-1.0, 0.0, 1.0], gate) is None
    assert swept_gate_crossing([-2.0, 1.2, 1.0], [2.0, 1.2, 1.0], gate) is None


def test_data_driven_technical_eight_track_loads_with_signed_order_contract() -> None:
    track = load_technical_eight()

    assert track.name == "technical-eight"
    assert track.gate_count == 8
    assert track.gate_order == tuple(range(1, 9))
    assert track.gate_sequence == tuple(range(8))
    assert track.gate_directions == (1,) * 8
    assert track.gate_sequence_direction == (1,) * 8
    assert track.gate_for_order_index(4).gate_id == "crossover"


def test_track_loader_accepts_width_height_and_derives_right_basis(tmp_path) -> None:
    source = tmp_path / "derived-basis.json"
    source.write_text(
        json.dumps(
            {
                "name": "derived-basis",
                "gates": [
                    {
                        "id": "one",
                        "pos": [1.0, 2.0, 3.0],
                        "normal": [0.0, 1.0, 0.0],
                        "width": 2.0,
                        "height": 1.0,
                    }
                ],
                "gate_order": [-1, 1],
            }
        ),
        encoding="utf-8",
    )

    track = load_track(source)

    assert track.gate_sequence == (0, 0)
    assert track.gate_directions == (-1, 1)
    assert np.allclose(track.gates[0].right_world, [-1.0, 0.0, 0.0])
    assert track.gates[0].width_m == pytest.approx(2.0)


def test_track_rejects_invalid_signed_gate_order() -> None:
    gate = axis_gate()

    with pytest.raises(ValueError, match="nonzero"):
        Track(name="bad", gates=(gate,), gate_order=(0,))
    with pytest.raises(ValueError, match="valid"):
        Track(name="bad", gates=(gate,), gate_order=(2,))


def test_race_state_ignores_wrong_gate_order_before_advancing() -> None:
    track = two_gate_track()
    race = RaceState(track)
    race.start(0.0)
    gate_one, gate_two = track.gates

    assert race.update(*crossing_segment(gate_two), 1.0, previous_time_s=0.0) == ()
    assert race.next_gate_index == 0
    accepted = race.update(*crossing_segment(gate_one), 2.0, previous_time_s=1.0)

    assert len(accepted) == 1
    assert isinstance(accepted[0], GatePassed)
    assert accepted[0].gate_index == 0
    assert race.next_gate_index == 1


def test_race_state_records_ordered_gate_lap_and_finish_events() -> None:
    track = two_gate_track()
    race = RaceState(track)
    race.start(0.0)
    first = race.update(*crossing_segment(track.gates[0]), 1.0, previous_time_s=0.0)
    second = race.update(*crossing_segment(track.gates[1]), 2.0, previous_time_s=1.0)

    assert len(first) == 1
    assert isinstance(first[0], GatePassed)
    assert [type(event) for event in second] == [GatePassed, LapCompleted, RaceFinished]
    assert race.finished and not race.running
    assert race.lap == 1
    assert race.gates_passed == 2
    assert race.n_gates_passed == 2
    # Each centerline segment crosses its plane halfway through its supplied
    # time interval, so the exact lap end is t=1.5 rather than the t=2 end tick.
    assert race.completed_lap_times_s == [pytest.approx(1.5)]
    assert race.best_lap_s == pytest.approx(1.5)
    assert race.next_gate is None


def test_race_state_supports_multiple_laps_and_best_lap() -> None:
    track = two_gate_track()
    race = RaceState(track, laps=2)
    race.start(0.0)

    race.update(*crossing_segment(track.gates[0]), 1.0, previous_time_s=0.0)
    first_finish = race.update(*crossing_segment(track.gates[1]), 2.0, previous_time_s=1.0)
    race.update(*crossing_segment(track.gates[0]), 2.75, previous_time_s=2.0)
    final_finish = race.update(*crossing_segment(track.gates[1]), 3.5, previous_time_s=2.75)

    assert [type(event) for event in first_finish] == [GatePassed, LapCompleted]
    assert race.completed_lap_times_s == [pytest.approx(1.5), pytest.approx(1.625)]
    assert race.best_lap_s == pytest.approx(1.5)
    assert [type(event) for event in final_finish] == [GatePassed, LapCompleted, RaceFinished]
    assert race.gates_passed == 4


def test_repeated_gate_order_requires_real_recrossing_not_duplicate_observation() -> None:
    gate = axis_gate()
    track = Track(name="repeat", gates=(gate,), gate_order=(1, 1))
    race = RaceState(track)
    race.start(0.0)
    forward = crossing_segment(gate)

    first = race.update(*forward, 1.0, previous_time_s=0.0)
    duplicate = race.update(*forward, 2.0, previous_time_s=1.0)
    rearm = race.update(forward[1], forward[0], 3.0, previous_time_s=2.0)
    second = race.update(*forward, 4.0, previous_time_s=3.0)

    assert len(first) == 1
    assert duplicate == ()
    assert rearm == ()
    assert [type(event) for event in second] == [GatePassed, LapCompleted, RaceFinished]
    assert race.gates_passed == 2


def test_repeated_gate_order_can_request_an_immediate_reverse_pass() -> None:
    gate = axis_gate()
    track = Track(name="backtrack", gates=(gate,), gate_order=(1, -1))
    race = RaceState(track)
    race.start(0.0)
    forward = crossing_segment(gate)

    race.update(*forward, 1.0, previous_time_s=0.0)
    reverse = race.update(forward[1], forward[0], 2.0, previous_time_s=1.0)

    assert [type(event) for event in reverse] == [GatePassed, LapCompleted, RaceFinished]
    assert isinstance(reverse[0], GatePassed)
    assert reverse[0].direction == -1


def test_gate_crossing_time_is_interpolated_when_segment_timestamps_are_known() -> None:
    gate = axis_gate()
    race = RaceState(Track(name="single", gates=(gate,), gate_order=(1,)))
    race.start(0.0)

    events = race.update([-2.0, 0.0, 1.0], [2.0, 0.0, 1.0], 4.0, previous_time_s=2.0)

    assert isinstance(events[0], GatePassed)
    assert events[0].time_s == pytest.approx(3.0)
    assert race.completed_lap_times_s == [pytest.approx(3.0)]


def test_step_is_an_explicit_simulation_loop_alias_for_update() -> None:
    gate = axis_gate()
    race = RaceState(Track(name="single", gates=(gate,), gate_order=(1,)))
    race.start(0.0)

    events = race.step(*crossing_segment(gate), 1.0, previous_time_s=0.0)

    assert isinstance(events[0], GatePassed)


def test_manual_wrong_gate_event_is_ignored_and_collision_is_event_oriented() -> None:
    track = two_gate_track()
    race = RaceState(track)
    race.start(0.0)

    assert race.apply_event(GatePassed(1, [5.0, 0.0, 1.0], 1.0, direction=1)) == ()
    collision = race.record_collision("gate-frame:two", 1.5)

    assert collision == (Collision("gate-frame:two", 1.5),)
    assert race.collisions == 1
    assert race.next_gate_index == 0


def test_ground_and_gate_frame_collision_helpers_keep_aperture_clear() -> None:
    gate = axis_gate()

    assert ground_collision([0.0, 0.0, 0.05], vehicle_radius_m=0.1)
    assert not ground_collision([0.0, 0.0, 0.11], vehicle_radius_m=0.1)
    assert not gate_frame_collision([0.0, 0.0, 1.0], gate, vehicle_radius_m=0.1)
    assert gate_frame_collision([0.0, 1.04, 1.0], gate, vehicle_radius_m=0.05)
    assert not gate_frame_collision([0.2, 0.0, 1.0], gate, vehicle_radius_m=0.01)


def test_gate_basis_and_input_contracts_are_validated() -> None:
    with pytest.raises(ValueError, match="right-handed"):
        Gate(
            center_world_m=[0.0, 0.0, 0.0],
            normal_world=[1.0, 0.0, 0.0],
            right_world=[0.0, -1.0, 0.0],
            up_world=[0.0, 0.0, 1.0],
            half_width_m=1.0,
            half_height_m=1.0,
        )
    with pytest.raises(ValueError, match="finite vector"):
        axis_gate().local_coordinates([np.nan, 0.0, 0.0])
