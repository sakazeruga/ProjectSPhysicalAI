import json
import logging

import pytest

from input_handler import MAX_EVENT_LENGTH, MessageError, parse_affect_message
from interfaces import AffectiveState


def msg(**kw) -> str:
    return json.dumps(kw)


def test_full_message():
    s = parse_affect_message(msg(arousal=0.7, valence=-0.3, event_trigger="surprise"), source="t")
    assert (s.arousal, s.valence, s.event_trigger, s.source) == (0.7, -0.3, "surprise", "t")


def test_missing_keys_inherit_previous_values():
    prev = AffectiveState(0.8, 0.5, "old")
    s = parse_affect_message(msg(event_trigger="stop"), prev)
    assert (s.arousal, s.valence, s.event_trigger) == (0.8, 0.5, "stop")


def test_missing_keys_without_previous_default_to_zero():
    s = parse_affect_message(msg(event_trigger="notify"))
    assert (s.arousal, s.valence) == (0.0, 0.0)


def test_event_is_not_inherited():
    s = parse_affect_message(msg(arousal=0.1), AffectiveState(0.5, 0.0, "surprise"))
    assert s.event_trigger is None


@pytest.mark.parametrize(
    "payload, expected",
    [({"arousal": 1.5}, (1.0, 0.0)), ({"arousal": -0.2}, (0.0, 0.0)), ({"valence": -3}, (0.0, -1.0))],
)
def test_out_of_range_is_clamped_with_warning(payload, expected, caplog):
    with caplog.at_level(logging.WARNING):
        s = parse_affect_message(json.dumps(payload))
    assert (s.arousal, s.valence) == expected
    assert "clamped" in caplog.text


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[1, 2]",
        msg(arousal="0.5"),
        msg(arousal=True),
        '{"arousal": NaN}',
        '{"valence": Infinity}',
        msg(event_trigger=123),
        b"\xff\xfe",
    ],
)
def test_invalid_messages_are_rejected(raw):
    with pytest.raises(MessageError):
        parse_affect_message(raw)


def test_event_is_trimmed_and_truncated():
    assert parse_affect_message(msg(event_trigger="  surprise \n")).event_trigger == "surprise"
    assert parse_affect_message(msg(event_trigger="   ")).event_trigger is None
    assert len(parse_affect_message(msg(event_trigger="x" * 500)).event_trigger) == MAX_EVENT_LENGTH


def test_null_values_are_treated_as_missing():
    s = parse_affect_message(msg(arousal=None, valence=0.2, event_trigger=None), AffectiveState(0.4, 0.0))
    assert (s.arousal, s.valence, s.event_trigger) == (0.4, 0.2, None)
