"""MuJoCo の物理シミュレーションで接触制御と安全機構を検証する。"""
import asyncio
import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from care_agent import EMOTIONS, CarePolicy  # noqa: E402
from interfaces import ActuatorCommand  # noqa: E402
from sim.mujoco_output import MujocoOutput  # noqa: E402
from sim.scenario import Cue, ScenarioRunner, evaluate, release_time  # noqa: E402
from sim.touch_care import Phase, TouchCareSim  # noqa: E402


def stroke(t, vib=60):
    return ActuatorCommand(vib, 50 + 30 * math.sin(2 * math.pi * 0.12 * t), 20, "CALM")


def run(sim, seconds, fn=stroke, dt=0.01):
    samples = []
    start = sim.time
    while sim.time - start < seconds:
        sim.command(fn(sim.time))
        sim.step(dt)
        samples.append(sim.sample())
    return samples


@pytest.fixture
def stroking():
    """接触して 1 ストローク以上なでている状態のシミュレーション。"""
    sim = TouchCareSim()
    run(sim, 15.0)
    assert sim.phase is Phase.CONTACT
    return sim


def test_robot_hovers_until_commanded():
    sim = TouchCareSim()
    samples = run(sim, 2.0, lambda t: ActuatorCommand(0, 50, 10, "IDLE"))
    assert max(s.force_raw for s in samples) == 0.0
    assert samples[-1].z == pytest.approx(sim.cfg.hover_height, abs=0.02)


def test_contact_force_tracks_target_while_stroking(stroking):
    sim = stroking
    forces = [s.force for s in run(sim, 12.0) if s.phase is Phase.CONTACT]
    assert np.mean(forces) == pytest.approx(sim.cfg.contact_force, abs=0.1)
    assert np.std(forces) < 0.2
    assert max(forces) < sim.cfg.force_limit


def test_stroking_never_trips_the_reflex(stroking):
    run(stroking, 12.0)
    assert stroking.reflex_count == 0


def test_pad_follows_commanded_position(stroking):
    sim = stroking
    samples = run(sim, 9.0)
    assert min(s.x for s in samples) < sim._position_to_x(30)
    assert max(s.x for s in samples) > sim._position_to_x(70)


def test_stroke_speed_is_limited_even_if_commanded_faster(stroking):
    sim = stroking
    samples = run(sim, 4.0, lambda t: ActuatorCommand(0, 90 if int(t) % 2 else 10, 100, "EXCITED"))
    contact = [abs(s.speed) for s in samples if s.phase is Phase.CONTACT]
    assert max(contact) < 0.105


def test_safe_stop_releases_immediately(stroking):
    sim = stroking
    t0 = sim.time
    samples = run(sim, 1.0, lambda t: ActuatorCommand.safe())
    assert release_time(samples, t0) - t0 <= 0.05
    assert samples[-1].force_raw == 0.0
    assert samples[-1].z > sim.cfg.hover_height - 0.03


def test_watchdog_releases_when_commands_stop(stroking):
    sim = stroking
    t0 = sim.time
    samples = []
    while sim.time - t0 < 1.0:
        sim.step(0.01)  # command() を呼ばない = 指令途絶
        samples.append(sim.sample())
    assert release_time(samples, t0) - t0 == pytest.approx(sim.cfg.watchdog, abs=0.1)


def test_flinch_triggers_reflex_and_recovers(stroking):
    sim = stroking
    sim.elbow_target = 0.55
    samples = run(sim, 1.0)
    assert sim.reflex_count == 1
    assert samples[-1].z > 0.2
    over = sum(s.force_raw > sim.cfg.force_limit for s in samples) * 0.01
    assert over < 0.1  # 過大な力が続かない

    sim.elbow_target = 0.0
    run(sim, sim.cfg.reflex_hold + 6.0)
    assert sim.phase is Phase.CONTACT


@pytest.mark.parametrize("position", [5, 50, 95])
def test_flinch_force_stays_near_the_limit_anywhere_on_the_arm(position):
    """手首側ほど腕の持ち上がりが速く、力の条件が厳しい。"""
    sim = TouchCareSim()
    hold = lambda t: ActuatorCommand(0, position, 20, "CALM")  # noqa: E731
    run(sim, 8.0, hold)
    assert sim.phase is Phase.CONTACT
    sim.elbow_target = 0.55
    samples = run(sim, 1.0, hold, dt=0.002)
    assert sim.reflex_count == 1
    assert max(s.force_raw for s in samples) < 6.0
    assert sum(s.force_raw > sim.cfg.force_limit for s in samples) * 0.002 < 0.05


class FixedEstimator:
    """台本の文に固定の感情を割り当てる。"""

    TABLE = {
        "怒": dict(anger=3.0, disgust=2.0),
        "悲": dict(sadness=2.8),
        "嬉": dict(joy=2.9, anticipation=1.0),
    }

    def estimate(self, text):
        for key, value in self.TABLE.items():
            if key in text:
                return {k: value.get(k, 0.0) for k in EMOTIONS}
        return {k: 0.0 for k in EMOTIONS}


def test_scenario_end_to_end_with_refusal_and_flinch():
    """発話 → 感情 → 応答 → StateMapper → 物理 までを通し、評価値が要件を満たすこと。"""
    script = (
        Cue(1.0, "悲しい"),
        Cue(12.0, "怒った"),
        Cue(30.0, "やめて"),
        Cue(33.0, "続けて"),
        Cue(46.0, elbow=0.55),
        Cue(47.0, elbow=0.0),
    )
    runner = ScenarioRunner(CarePolicy(FixedEstimator()), script=script, duration=54.0)
    ticks = list(runner.step_all())
    m = evaluate(ticks, runner.sim.cfg, [30.0], runner.sim.reflex_count)

    assert m.contact_time > 25.0
    assert m.force_mean == pytest.approx(1.5, abs=0.15)
    assert m.speed_peak < 0.13
    assert m.speed_ct_ratio > 0.9
    assert m.stop_latency and m.stop_latency[0] < 0.1
    assert m.reflex_count == 1  # 腕を引いた 1 回だけ

    modes = {k.sample.mode for k in ticks}
    assert {"GLOOMY", "PULSE", "CALM", "SAFE_STOP"} <= modes


def test_mujoco_output_implements_actuator_interface():
    async def scenario():
        output = MujocoOutput(viewer=False)
        await output.start()
        assert output.is_connected
        for _ in range(100):
            output.submit(ActuatorCommand(0, 50, 20, "CALM"))
            await asyncio.sleep(0.01)
        stats = output.stats()
        await output.stop()
        return output, stats

    output, stats = asyncio.run(scenario())
    assert stats["frames"] == 100
    assert output.sim.time > 0.5
    assert not output.is_connected
