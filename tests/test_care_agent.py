import pytest

from care_agent import (
    ACCOMPANY,
    EMOTIONS,
    REJOICE,
    SOOTHE,
    STROKE,
    WATCH,
    CarePolicy,
    CareSession,
    care_mapper_config,
    care_response,
    safety_event,
    user_affect,
)
from interfaces import AffectiveState
from state_mapper import Mode, StateMapper


def emotions(**kwargs):
    return {k: kwargs.get(k, 0.0) for k in EMOTIONS}


class FakeEstimator:
    """モデルを読み込まず、固定の感情を返す。"""

    def __init__(self, **kwargs):
        self.result = emotions(**kwargs)

    def estimate(self, text):
        return self.result


@pytest.mark.parametrize(
    ("emo", "low", "high"),
    [
        (dict(joy=3.0), (0.4, 0.7), (0.8, 1.0)),
        (dict(anger=3.0), (0.7, -1.0), (1.0, -0.5)),
        (dict(sadness=3.0), (0.0, -1.0), (0.3, -0.6)),
    ],
)
def test_user_affect_places_emotions_on_the_circumplex(emo, low, high):
    a, v = user_affect(emotions(**emo))
    assert low[0] <= a <= high[0]
    assert low[1] <= v <= high[1]


def test_user_affect_is_neutral_for_flat_text():
    assert user_affect(emotions()) == (0.0, 0.0)


def test_user_affect_clamps_out_of_range_regression_output():
    # 回帰モデルなので 0..3 の外の値が出る
    a, v = user_affect(emotions(joy=9.0, anger=-2.0))
    assert a == pytest.approx(user_affect(emotions(joy=3.0))[0])
    assert v == pytest.approx(user_affect(emotions(joy=3.0))[1])


@pytest.mark.parametrize(
    ("a", "v", "expected"),
    [
        (0.8, -0.7, SOOTHE),  # 怒り → 同調せず、ゆっくり
        (0.2, -0.7, ACCOMPANY),  # 悲しみ → 手を止めて寄り添う
        (0.7, 0.8, REJOICE),
        (0.3, 0.6, STROKE),
        (0.2, 0.0, WATCH),
    ],
)
def test_care_response_quadrants(a, v, expected):
    assert care_response(a, v) is expected


def test_robot_never_mirrors_an_angry_user():
    """怒りの相手には、ロボットは低覚醒・快の側で応答する。"""
    r = care_response(0.9, -0.8)
    assert r.arousal < 0.3
    assert r.valence > 0


@pytest.mark.parametrize("text", ["やめて", "ちょっと痛い", "ストップ！", "もう止めて"])
def test_refusal_words_stop(text):
    assert safety_event(text) == "stop"


def test_refusal_wins_over_resume():
    assert safety_event("やめて、もう大丈夫") == "stop"


def test_resume_words():
    assert safety_event("もう大丈夫、続けて") == "resume"
    assert safety_event("今日は疲れた") is None


def test_stop_does_not_depend_on_the_model():
    """モデルが「喜び」と誤判定しても、拒否の語彙があれば停止する。"""
    decision = CarePolicy(FakeEstimator(joy=3.0)).decide("痛い")
    assert decision.event == "stop"
    assert decision.response is None


def test_resume_strokes_even_if_the_text_reads_negative():
    """「ごめん、続けて」のように文面が不快寄りでも、頼まれたとおりなでる。"""
    decision = CarePolicy(FakeEstimator(sadness=3.0)).decide("ごめん、続けて")
    assert decision.event == "resume"
    assert decision.response is STROKE


def session(**kwargs):
    return CareSession(CarePolicy(FakeEstimator(**kwargs)))


def test_session_keeps_sending_so_mapper_input_does_not_time_out():
    s = session(joy=2.0)
    s.accept(s.policy.decide("うれしい"), 0.0)
    assert s.tick(0.1) is None
    assert isinstance(s.tick(0.51), AffectiveState)


def test_session_is_silent_until_resume_after_stop():
    s = session()
    s.accept(s.policy.decide("やめて"), 0.0)
    assert s.tick(10.0) is None
    state = s.accept(s.policy.decide("続けて"), 11.0)
    assert state.event_trigger == "resume"
    assert s.tick(12.0) is not None


def test_session_sends_heartbeat_only_for_accompany():
    s = session(sadness=3.0)
    s.accept(s.policy.decide("かなしい"), 0.0)
    assert s.decision.response is ACCOMPANY
    events = [e for e in (s.tick(t * 0.1) for t in range(1, 60)) if e and e.event_trigger]
    assert len(events) >= 3
    assert {e.event_trigger for e in events} == {"heartbeat"}

    s = session(joy=3.0)
    s.accept(s.policy.decide("うれしい"), 0.0)
    assert not [e for e in (s.tick(t * 0.1) for t in range(1, 60)) if e and e.event_trigger]


def run_mapper(response, seconds=30.0):
    mapper = StateMapper(care_mapper_config())
    state = AffectiveState(response.arousal, response.valence, received_at=0.0)
    mapper.update(0.0, state)
    positions = []
    dt = 0.01
    for i in range(1, int(seconds / dt)):
        t = i * dt
        state = AffectiveState(response.arousal, response.valence, received_at=t)
        positions.append(mapper.update(t, state).linear_position)
    return mapper, positions, dt


@pytest.mark.parametrize("response", [STROKE, SOOTHE, ACCOMPANY, REJOICE, WATCH])
def test_care_patterns_stay_in_the_pleasant_touch_speed_band(response):
    """前腕 20cm=100% なので 50%/s が 10cm/s。どの振る舞いでも上限を超えない。"""
    _, positions, dt = run_mapper(response)
    peak = max(abs(b - a) / dt for a, b in zip(positions, positions[1:]))
    assert peak <= 50.0 + 1.0  # 丸め誤差分


def test_soothing_is_slower_than_rejoicing():
    def peak(response):
        _, positions, dt = run_mapper(response)
        return max(abs(b - a) / dt for a, b in zip(positions[500:], positions[501:]))

    assert peak(SOOTHE) < peak(REJOICE)


@pytest.mark.parametrize(
    ("response", "mode"),
    [(SOOTHE, Mode.CALM), (STROKE, Mode.CALM), (ACCOMPANY, Mode.GLOOMY), (REJOICE, Mode.EXCITED), (WATCH, Mode.NEUTRAL)],
)
def test_responses_select_expected_modes(response, mode):
    mapper, _, _ = run_mapper(response, seconds=2.0)
    assert mapper.mode is mode
