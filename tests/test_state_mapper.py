import logging

import pytest

from interfaces import ActuatorCommand, AffectiveState
from state_mapper import MapperConfig, Mode, StateMapper

DT = 0.01  # main.py の既定制御周期 (100Hz)


class Sim:
    """仮想時刻で StateMapper を駆動するヘルパ。"""

    def __init__(self, cfg: MapperConfig | None = None):
        self.m = StateMapper(cfg)
        self.t = 1.0
        self.cmds: list[ActuatorCommand] = []

    def send(self, a: float, v: float, event: str | None = None) -> ActuatorCommand:
        cmd = self.m.update(self.t, AffectiveState(a, v, event, received_at=self.t), [event] if event else [])
        self.cmds.append(cmd)
        return cmd

    def run(self, seconds: float) -> list[ActuatorCommand]:
        out = []
        for _ in range(round(seconds / DT)):
            self.t += DT
            out.append(self.m.update(self.t))
        self.cmds += out
        return out

    def hold(self, a: float, v: float, seconds: float) -> list[ActuatorCommand]:
        """10Hz で入力を送り続けながら seconds 秒進める。"""
        out = []
        for _ in range(round(seconds / 0.1)):
            out.append(self.send(a, v))
            out += self.run(0.1)
        return out


@pytest.mark.parametrize(
    "a, v, mode",
    [
        (0.0, 0.0, Mode.IDLE),
        (0.3, 0.0, Mode.NEUTRAL),
        (0.3, 0.6, Mode.CALM),
        (0.9, 0.7, Mode.EXCITED),
        (0.9, -0.7, Mode.TENSE),
        (0.2, -0.6, Mode.GLOOMY),
    ],
)
def test_quadrant_classification(a, v, mode):
    sim = Sim()
    sim.hold(a, v, 1.0)
    assert sim.m.mode is mode


def test_valence_hysteresis():
    sim = Sim()
    sim.hold(0.3, 0.0, 1.0)
    sim.hold(0.3, 0.30, 1.0)  # 0.25 は超えたが 0.25+0.07 未満 → まだ中立
    assert sim.m.mode is Mode.NEUTRAL
    sim.hold(0.3, 0.35, 1.0)
    assert sim.m.mode is Mode.CALM
    sim.hold(0.3, 0.20, 1.0)  # 0.25 を下回ったが 0.25-0.07 以上 → 快のまま
    assert sim.m.mode is Mode.CALM
    sim.hold(0.3, 0.15, 1.0)
    assert sim.m.mode is Mode.NEUTRAL


def test_min_dwell_limits_chattering():
    sim = Sim()
    sim.hold(0.3, 0.0, 1.0)
    modes = []
    for i in range(20):  # 50ms ごとに快/中立を行き来させる (1 秒間)
        sim.send(0.3, 0.6 if i % 2 == 0 else 0.0)
        modes += [c.mode for c in sim.run(0.05)]
    transitions = sum(1 for p, n in zip(modes, modes[1:]) if p != n)
    assert transitions <= 1.0 / MapperConfig().min_dwell + 1


def test_vibration_scales_with_arousal_and_respects_deadzone():
    levels = []
    for a in (0.6, 0.8, 1.0):
        sim = Sim()
        levels.append(sim.hold(a, 0.7, 1.0)[-1].vibration_intensity)
    assert levels == sorted(levels) and levels[-1] == 255

    sim = Sim()
    assert sim.hold(0.03, 0.0, 1.0)[-1].vibration_intensity == 0


def test_vibration_is_slew_rate_limited():
    sim = Sim()
    sim.hold(0.0, 0.0, 1.0)
    sim.send(1.0, 0.7)
    first = sim.run(DT)[0]
    assert first.vibration_intensity <= MapperConfig().pwm_slew_rate * DT * 2


def test_outputs_stay_within_limits_and_motion_is_continuous():
    cfg = MapperConfig()
    sim = Sim(cfg)
    cmds = []
    for a, v in [(0.3, 0.0), (1.0, 0.9), (1.0, -0.9), (0.1, -0.8), (0.4, 0.6), (1.0, 0.9), (0.0, 0.0)]:
        cmds += sim.hold(a, v, 1.5)

    for c in cmds:
        assert cfg.position_min <= c.linear_position <= cfg.position_max
        assert cfg.min_speed <= c.translation_speed <= 100.0
        assert 0 <= c.vibration_intensity <= 255

    # ファームは translation_speed を速度上限として目標を追うため、
    # 実際の目標移動速度が申告速度とデバイス最大速度を超えないことを確認する
    for prev, cur in zip(cmds, cmds[1:]):
        if cur.created_at == prev.created_at:
            continue
        actual = abs(cur.linear_position - prev.linear_position) / DT  # [%/s]
        # 入力受信時は同じ時刻で 2 回 update されるので 0 割を避けて比較
        assert actual <= cfg.max_stroke_speed * 1.05, (prev, cur)
        assert actual <= cur.translation_speed / 100 * cfg.max_stroke_speed * 1.1 + 1.0, (prev, cur)


def test_surprise_triggers_impulse_then_returns():
    sim = Sim()
    sim.hold(0.9, 0.7, 1.0)
    sim.send(0.9, 0.7, "surprise")
    during = sim.run(0.3)
    assert all(c.mode == "IMPULSE" and c.vibration_intensity == 255 for c in during)
    assert during[-1].linear_position == pytest.approx(MapperConfig().impulse_position)
    after = sim.run(0.2)
    assert after[-1].mode == "EXCITED"


def test_heartbeat_pulses_vibration():
    sim = Sim()
    sim.hold(0.2, -0.6, 1.0)
    base = sim.cmds[-1].vibration_intensity
    sim.send(0.2, -0.6, "heartbeat")
    pwm = [c.vibration_intensity for c in sim.run(0.6)]
    assert max(pwm) == 255 and pwm[-1] < 255 and base < 255


def test_unknown_event_warns_once(caplog):
    sim = Sim()
    with caplog.at_level(logging.WARNING):
        sim.send(0.5, 0.0, "dance")
        sim.send(0.5, 0.0, "DANCE ")
    assert caplog.text.count("not mapped") == 1


def test_stop_latches_until_resume():
    cfg = MapperConfig()
    sim = Sim(cfg)
    sim.hold(0.9, 0.7, 1.0)
    sim.send(0.9, 0.7, "emergency_stop")
    sim.send(1.0, 1.0, "surprise")  # 停止中のイベントは無視
    held = sim.hold(1.0, 1.0, 1.0)
    assert all(c.mode == "SAFE_STOP" and c.vibration_intensity == 0 for c in held)
    assert all(c.linear_position == cfg.home_position for c in held)

    sim.send(1.0, 1.0, "resume")
    sim.hold(1.0, 1.0, 1.0)
    assert sim.m.mode is Mode.EXCITED


def test_input_timeout_decays_to_idle(caplog):
    sim = Sim()
    sim.hold(0.9, 0.7, 1.0)
    with caplog.at_level(logging.WARNING):
        tail = sim.run(MapperConfig().input_timeout + 4.0)
    assert "No input" in caplog.text
    assert sim.m.mode is Mode.IDLE
    assert tail[-1].vibration_intensity == 0


def test_new_input_command_carries_receive_timestamp():
    sim = Sim()
    cmd = sim.send(0.5, 0.0)
    assert cmd.source_received_at == sim.t
    assert sim.run(DT)[0].source_received_at is None
