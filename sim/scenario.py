"""
決められた台本でタッチケアを通し実行し、評価値を計算する。

リアルタイムの main.py と同じ CareSession / StateMapper / TouchCareSim をシミュレーション時刻で
駆動するので、実行時間に依存せず同じ結果が得られる。動画生成と統合テストはここを使う。
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from care_agent import CareDecision, CarePolicy, CareSession, care_mapper_config
from interfaces import ActuatorCommand
from sim.touch_care import Phase, TouchCareConfig, TouchCareSim, TouchSample
from state_mapper import StateMapper


@dataclass(frozen=True, slots=True)
class Cue:
    """台本の 1 項目。t 秒に text を発話する、または elbow へ腕を動かす。"""

    t: float
    text: str | None = None
    elbow: float | None = None
    caption: str | None = None  # 発話以外の出来事を動画に表示する ("" で字幕を消す)


DEMO_SCRIPT: tuple[Cue, ...] = (
    Cue(4.0, "今日はちょっと疲れたな"),
    Cue(22.0, "仕事でミスして落ち込んでる"),
    Cue(38.0, "なんであんなこと言われなきゃいけないんだ！腹が立つ！"),
    Cue(58.0, "少し楽になってきた、ありがとう"),
    Cue(76.0, "やったー！明日はお休みだ！"),
    Cue(90.0, "ちょっと痛い、やめて"),
    Cue(97.0, "ごめん、もう大丈夫。続けて"),
    Cue(114.0, elbow=0.55, caption="（びくっと腕を引く）"),
    Cue(116.0, elbow=0.0, caption=""),
    Cue(122.0, "おかげで落ち着いた、ありがとう"),
)
DEMO_DURATION = 136.0


@dataclass(slots=True)
class Tick:
    """制御周期ごとの記録。"""

    sample: TouchSample
    command: ActuatorCommand
    decision: CareDecision | None
    caption: str | None


@dataclass(slots=True)
class Metrics:
    contact_time: float = 0.0  # 接触していた時間 [s]
    force_mean: float = 0.0  # 接触中の押し付け力の平均 [N]
    force_std: float = 0.0
    force_peak_filtered: float = 0.0  # 接触力のピーク (30ms 平滑) [N]
    force_peak_raw: float = 0.0  # 接触力のピーク (生値) [N]
    time_over_limit: float = 0.0  # 生値が上限を超えていた時間 [s]
    speed_peak: float = 0.0  # 接触中の前腕方向の最大速度 [m/s]
    speed_ct_ratio: float = 0.0  # 接触中のうち 10cm/s 以下だった割合
    stop_latency: list[float] = field(default_factory=list)  # 拒否の発話 → 接触解除 [s]
    reflex_count: int = 0


class ScenarioRunner:
    """台本に沿って 1 制御周期ずつ進める。step_all() は各周期の Tick を順に返す。"""

    def __init__(
        self,
        policy: CarePolicy,
        script: tuple[Cue, ...] = DEMO_SCRIPT,
        duration: float = DEMO_DURATION,
        config: TouchCareConfig | None = None,
        rate: float = 100.0,
    ) -> None:
        self.sim = TouchCareSim(config)
        self.mapper = StateMapper(care_mapper_config())
        self.session = CareSession(policy)
        self.policy = policy
        self.script = sorted(script, key=lambda c: c.t)
        self.duration = duration
        self.dt = 1.0 / rate
        self._decisions = [(c, policy.decide(c.text) if c.text else None) for c in self.script]

    def step_all(self) -> Iterator[Tick]:
        pending = list(self._decisions)
        caption = None
        while self.sim.time < self.duration:
            t = self.sim.time
            state = None
            events: list[str] = []
            while pending and pending[0][0].t <= t:
                cue, decision = pending.pop(0)
                if cue.elbow is not None:
                    self.sim.elbow_target = cue.elbow
                if cue.caption is not None:
                    caption = cue.caption
                if decision is not None:
                    caption = None
                    state = self.session.accept(decision, t)
            if state is None:
                state = self.session.tick(t)
            if state is not None and state.event_trigger:
                events.append(state.event_trigger)
            cmd = self.mapper.update(t, state, events)
            self.sim.command(cmd)
            self.sim.step(self.dt)
            yield Tick(self.sim.sample(), cmd, self.session.decision, caption)


def evaluate(ticks: list[Tick], config: TouchCareConfig, dt_stop_events: list[float], reflex_count: int) -> Metrics:
    """通し実行の記録から評価値を計算する。dt_stop_events は拒否を発話した時刻の列。"""
    samples = [k.sample for k in ticks]
    dt = samples[1].t - samples[0].t
    contact = [s for s in samples if s.phase is Phase.CONTACT]
    m = Metrics(reflex_count=reflex_count)
    if contact:
        forces = [s.force for s in contact]
        mean = sum(forces) / len(forces)
        m.contact_time = len(contact) * dt
        m.force_mean = mean
        m.force_std = (sum((f - mean) ** 2 for f in forces) / len(forces)) ** 0.5
        m.speed_peak = max(abs(s.speed) for s in contact)
        m.speed_ct_ratio = sum(abs(s.speed) <= 0.10 for s in contact) / len(contact)
    m.force_peak_filtered = max(s.force for s in samples)
    m.force_peak_raw = max(s.force_raw for s in samples)
    m.time_over_limit = sum(s.force_raw > config.force_limit for s in samples) * dt
    for t_stop in dt_stop_events:
        released = release_time(samples, t_stop)
        if released is not None:
            m.stop_latency.append(released - t_stop)
    return m


def release_time(samples: list[TouchSample], since: float, hold: float = 0.05) -> float | None:
    """since 以降で接触力が hold 秒続けて 0 になった最初の時刻。

    パッドは軽いので接触中も瞬間的に力が 0 になることがあり、1 サンプルでは離脱と判定できない。
    """
    start = None
    for s in samples:
        if s.t < since:
            continue
        if s.force_raw > 0.0:
            start = None
        elif start is None:
            start = s.t
        elif s.t - start >= hold:
            return start
    return None
