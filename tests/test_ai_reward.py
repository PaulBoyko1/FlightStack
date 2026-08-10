import numpy as np
import pytest

from flightstack.ai.config import load_racing_ai_config
from flightstack.ai.reward import race_reward
from flightstack.race import Collision, GatePassed, LapCompleted


def test_reward_instruments_progress_events_and_penalties_separately() -> None:
    config = load_racing_ai_config().reward
    events = (
        GatePassed(0, np.zeros(3), time_s=0.1, direction=1),
        LapCompleted(1, lap_time_s=0.1, time_s=0.1),
        Collision("ground", time_s=0.1),
    )
    terms = race_reward(
        previous_distance_m=5.0,
        current_distance_m=2.0,
        previous_action=np.zeros(4),
        action=np.array([1.0, 0.0, 0.0, 0.0]),
        body_rate_rad_s=np.array([1.0, 0.0, 0.0]),
        max_body_rate_rad_s=np.array([2.0, 2.0, 2.0]),
        events=events,
        out_of_bounds=True,
        control_dt_s=0.02,
        config=config,
    )
    assert terms.progress == pytest.approx(3.0)
    assert terms.gate_pass == pytest.approx(5.0)
    assert terms.lap_complete == pytest.approx(20.0)
    assert terms.collision == pytest.approx(-12.0)
    assert terms.out_of_bounds == pytest.approx(-12.0)
    assert terms.action_delta == pytest.approx(-0.03)
    assert terms.angular_rate == pytest.approx(-0.0005)
    assert terms.time == pytest.approx(-0.0004)
    assert terms.to_mapping()["total"] == pytest.approx(terms.total)


def test_reward_rejects_invalid_rate_shape_and_nonfinite_distance() -> None:
    config = load_racing_ai_config().reward
    kwargs = dict(
        previous_distance_m=1.0,
        current_distance_m=1.0,
        previous_action=np.zeros(4),
        action=np.zeros(4),
        body_rate_rad_s=np.zeros(3),
        max_body_rate_rad_s=np.ones(3),
        events=(),
        out_of_bounds=False,
        control_dt_s=0.02,
        config=config,
    )
    with pytest.raises(ValueError, match="distances"):
        race_reward(**{**kwargs, "previous_distance_m": np.nan})
    with pytest.raises(ValueError, match="body rates"):
        race_reward(**{**kwargs, "body_rate_rad_s": np.zeros(2)})
