"""
状態変換層: 感情パラメータ (arousal / valence / event_trigger) → アクチュエータ制御値。

- vibration_intensity : arousal に単調増加 (デッドゾーン + 最小有効PWM + ガンマ補正 + スルーレート制限)
- linear_position     : モードごとのパターン (中心・振幅・周波数) を位相連続な発振器で生成
- translation_speed   : 発振器のピーク速度をデバイス最大速度で正規化した値

モードは Russell の円環モデルの4象限 + 中立/待機で決まり、ヒステリシスと最小滞在時間で
境界付近のチャタリングを防ぐ。event_trigger は一過性の割り込みパターン (IMPULSE/PULSE) や
ラッチ式の安全停止 (SAFE_STOP) を起動する。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import StrEnum

from interfaces import ActuatorCommand, AffectiveState

logger = logging.getLogger(__name__)


class Mode(StrEnum):
    IDLE = "IDLE"  # 入力なし / ほぼ無覚醒
    NEUTRAL = "NEUTRAL"  # 感情価が中立
    CALM = "CALM"  # 快 × 低覚醒 (リラックス)
    EXCITED = "EXCITED"  # 快 × 高覚醒 (高揚)
    TENSE = "TENSE"  # 不快 × 高覚醒 (緊張・怒り)
    GLOOMY = "GLOOMY"  # 不快 × 低覚醒 (沈静・悲しみ)
    # 以下はイベント起因
    IMPULSE = "IMPULSE"  # 驚き等: 瞬間的な最大突き出し + 振動
    PULSE = "PULSE"  # 通知等: 心拍のような二連振動
    SAFE_STOP = "SAFE_STOP"  # ラッチ停止 (resume まで解除しない)


@dataclass(frozen=True, slots=True)
class PatternSpec:
    center: float  # ストローク中心 [%]
    amplitude: float  # 振幅 [%] (arousal=1 時。低覚醒では縮小)
    freq_lo: float  # arousal=0 時の周波数 [Hz]
    freq_hi: float  # arousal=1 時の周波数 [Hz]
    vib_gain: float  # 振動強度の倍率 0..1


DEFAULT_PATTERNS: dict[Mode, PatternSpec] = {
    Mode.IDLE: PatternSpec(center=10.0, amplitude=0.0, freq_lo=0.0, freq_hi=0.0, vib_gain=0.0),
    Mode.NEUTRAL: PatternSpec(center=50.0, amplitude=10.0, freq_lo=0.2, freq_hi=0.6, vib_gain=0.7),
    Mode.CALM: PatternSpec(center=45.0, amplitude=25.0, freq_lo=0.15, freq_hi=0.4, vib_gain=0.5),
    Mode.EXCITED: PatternSpec(center=50.0, amplitude=40.0, freq_lo=0.8, freq_hi=2.0, vib_gain=1.0),
    Mode.TENSE: PatternSpec(center=75.0, amplitude=8.0, freq_lo=2.0, freq_hi=4.0, vib_gain=1.0),
    Mode.GLOOMY: PatternSpec(center=15.0, amplitude=5.0, freq_lo=0.1, freq_hi=0.25, vib_gain=0.3),
}

# event_trigger (小文字化して照合) → 起動するモード
DEFAULT_EVENT_TABLE: dict[str, Mode] = {
    "surprise": Mode.IMPULSE,
    "startle": Mode.IMPULSE,
    "impact": Mode.IMPULSE,
    "heartbeat": Mode.PULSE,
    "notify": Mode.PULSE,
    "stop": Mode.SAFE_STOP,
    "emergency_stop": Mode.SAFE_STOP,
}
RESUME_EVENTS = frozenset({"resume", "reset"})


@dataclass(slots=True)
class MapperConfig:
    # --- モード判定 ---
    valence_threshold: float = 0.25  # |valence| がこれを超えると快/不快
    arousal_threshold: float = 0.55  # arousal がこれを超えると高覚醒
    idle_arousal: float = 0.05  # これ未満かつ中立なら IDLE
    hysteresis: float = 0.07  # しきい値の上下に設けるヒステリシス幅
    min_dwell: float = 0.25  # モード最小滞在時間 [s]
    # --- 振動 (PWM) ---
    pwm_deadzone: float = 0.05  # arousal がこれ未満なら PWM=0
    pwm_min: int = 60  # モータが回り始める最小PWM
    pwm_gamma: float = 1.3  # arousal → 強度のガンマ補正
    pwm_slew_rate: float = 1500.0  # PWM 変化率上限 [/s]
    # --- 直動 ---
    position_min: float = 2.0  # ソフトリミット [%]
    position_max: float = 98.0
    home_position: float = 0.0  # SAFE_STOP 時の退避位置 [%]
    max_stroke_speed: float = 400.0  # デバイス最大速度 [%/s] (translation_speed=100 に相当)
    min_speed: float = 5.0  # translation_speed の下限 [%]
    param_tau: float = 0.3  # パターン切替時の中心・振幅・周波数の追従時定数 [s]
    # --- イベント ---
    impulse_position: float = 95.0
    impulse_duration: float = 0.35
    pulse_duration: float = 0.6
    # --- 入力監視 ---
    input_timeout: float = 3.0  # これ以上入力が途絶えたら IDLE へ減衰
    decay_tau: float = 1.0  # 減衰時定数 [s]

    patterns: dict[Mode, PatternSpec] = field(default_factory=lambda: dict(DEFAULT_PATTERNS))
    event_table: dict[str, Mode] = field(default_factory=lambda: dict(DEFAULT_EVENT_TABLE))


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class StateMapper:
    """感情状態 → ActuatorCommand の変換器 (状態機械 + パターン発振器)。

    update() を制御周期ごと、および新着入力のたびに呼ぶ。I/O を持たない純粋な計算器なので
    単体テストやシミュレータからも同じように使える。
    """

    def __init__(self, config: MapperConfig | None = None) -> None:
        self.cfg = config or MapperConfig()
        self._mode = Mode.IDLE
        self._mode_since = 0.0
        self._valence_level = 0  # -1 / 0 / +1
        self._arousal_high = False
        self._idle = True

        self._affect: AffectiveState | None = None
        self._last_input_at: float | None = None
        self._eff_arousal = 0.0
        self._eff_valence = 0.0
        self._stale = False

        self._transient: Mode | None = None
        self._transient_until = 0.0
        self._transient_started = 0.0
        self._safe_latched = False

        idle = self.cfg.patterns[Mode.IDLE]
        self._center = idle.center
        self._amplitude = 0.0
        self._freq = 0.0
        self._phase = 0.0
        self._param_vel = 0.0
        self._last_position: float | None = None
        self._pwm = 0.0
        self._last_t: float | None = None
        self._unknown_events: set[str] = set()

    # ------------------------------------------------------------------ public
    @property
    def mode(self) -> Mode:
        if self._safe_latched:
            return Mode.SAFE_STOP
        return self._transient or self._mode

    @property
    def effective_affect(self) -> tuple[float, float]:
        """直近 update() で使った (arousal, valence)。"""
        return self._eff_arousal, self._eff_valence

    def update(
        self, t: float, state: AffectiveState | None = None, events: list[str] | None = None
    ) -> ActuatorCommand:
        dt = 0.0 if self._last_t is None else _clamp(t - self._last_t, 0.0, 0.1)
        self._last_t = t

        if state is not None:
            self._affect = state
            self._last_input_at = state.received_at
            if self._stale:
                logger.info("Input resumed")
                self._stale = False
        for ev in events or ():
            self._handle_event(ev, t)

        arousal, valence = self._effective(t)
        self._eff_arousal, self._eff_valence = arousal, valence
        source_ts = state.received_at if state is not None else None

        if self._safe_latched:
            self._reset_motion(self.cfg.home_position)
            self._last_position = self.cfg.home_position
            return ActuatorCommand.safe(self.cfg.home_position)

        self._classify(arousal, valence, t)
        position, speed = self._run_oscillator(arousal, dt)
        pwm_target = self._pwm_target(arousal)

        # 一過性イベントは定常パターンに上書きする
        mode = self._mode
        if self._transient is not None:
            if t >= self._transient_until:
                logger.info("EVENT %s finished -> %s", self._transient, self._mode)
                self._transient = None
            else:
                mode = self._transient
                elapsed = t - self._transient_started
                if mode is Mode.IMPULSE:
                    position, speed, pwm_target = self.cfg.impulse_position, 100.0, 255.0
                    self._pwm = 255.0
                elif mode is Mode.PULSE:
                    # 心拍状: 0-80ms と 220-300ms にフル振動
                    if elapsed < 0.08 or 0.22 <= elapsed < 0.30:
                        pwm_target = self._pwm = 255.0

        # スルーレート制限 (イベントで直接設定した場合はそのまま)
        max_step = self.cfg.pwm_slew_rate * dt
        self._pwm += _clamp(pwm_target - self._pwm, -max_step, max_step)

        position = _clamp(position, self.cfg.position_min, self.cfg.position_max)
        # 最終安全弁: 出力位置の変化をデバイス最大速度以内に制限 (IMPULSE の跳躍も最大速度の傾斜になる)
        if self._last_position is not None:
            max_step = self.cfg.max_stroke_speed * dt
            position = self._last_position + _clamp(position - self._last_position, -max_step, max_step)
        self._last_position = position
        return ActuatorCommand(
            vibration_intensity=int(round(_clamp(self._pwm, 0.0, 255.0))),
            linear_position=round(position, 2),
            translation_speed=round(_clamp(speed, self.cfg.min_speed, 100.0), 1),
            mode=mode.value,
            source_received_at=source_ts,
        )

    # ----------------------------------------------------------------- events
    def _handle_event(self, raw: str, t: float) -> None:
        name = raw.strip().lower()
        if name in RESUME_EVENTS:
            if self._safe_latched:
                logger.warning("EVENT %r: SAFE_STOP released", name)
                self._safe_latched = False
                self._set_mode(Mode.IDLE, t, "resume")
            return

        target = self.cfg.event_table.get(name)
        if target is None:
            if name not in self._unknown_events:
                self._unknown_events.add(name)
                logger.warning("EVENT %r is not mapped; ignored", name)
            return

        if target is Mode.SAFE_STOP:
            if not self._safe_latched:
                logger.warning("EVENT %r: SAFE_STOP latched (send 'resume' to release)", name)
            self._safe_latched = True
            self._transient = None
            return
        if self._safe_latched:
            logger.info("EVENT %r ignored while SAFE_STOP", name)
            return

        duration = (
            self.cfg.impulse_duration if target is Mode.IMPULSE else self.cfg.pulse_duration
        )
        self._transient = target
        self._transient_started = t
        self._transient_until = t + duration
        logger.info("EVENT %r -> %s (%.0f ms)", name, target, duration * 1000)

    # ---------------------------------------------------------- classification
    def _effective(self, t: float) -> tuple[float, float]:
        """入力途絶時は (0, 0) へ指数減衰させた値を返す (ウォッチドッグ)。"""
        if self._affect is None or self._last_input_at is None:
            return 0.0, 0.0
        silent = t - self._last_input_at
        a, v = self._affect.arousal, self._affect.valence
        if silent <= self.cfg.input_timeout:
            return a, v
        if not self._stale:
            logger.warning("No input for %.1f s; decaying to IDLE", silent)
            self._stale = True
        k = math.exp(-(silent - self.cfg.input_timeout) / self.cfg.decay_tau)
        return a * k, v * k

    def _classify(self, a: float, v: float, t: float) -> None:
        c, h = self.cfg, self.cfg.hysteresis

        # 感情価: 現在のレベルに留まる側へしきい値をずらす
        thr = c.valence_threshold
        pos = v > (thr - h if self._valence_level == 1 else thr + h)
        neg = v < (-thr + h if self._valence_level == -1 else -thr - h)
        self._valence_level = 1 if pos else -1 if neg else 0

        thr = c.arousal_threshold
        self._arousal_high = a > (thr - h if self._arousal_high else thr + h)

        thr = c.idle_arousal
        self._idle = self._valence_level == 0 and a < (thr + h if self._idle else thr)

        if self._idle:
            candidate = Mode.IDLE
        elif self._valence_level == 0:
            candidate = Mode.NEUTRAL
        elif self._valence_level > 0:
            candidate = Mode.EXCITED if self._arousal_high else Mode.CALM
        else:
            candidate = Mode.TENSE if self._arousal_high else Mode.GLOOMY

        if candidate is not self._mode and t - self._mode_since >= c.min_dwell:
            self._set_mode(candidate, t, f"arousal={a:.2f} valence={v:+.2f}")

    def _set_mode(self, mode: Mode, t: float, reason: str) -> None:
        if mode is self._mode:
            return
        logger.info("MODE %s -> %s (%s)", self._mode, mode, reason)
        self._mode = mode
        self._mode_since = t

    # ------------------------------------------------------------- generation
    def _run_oscillator(self, a: float, dt: float) -> tuple[float, float]:
        """位相連続のサイン発振器。パラメータは一次遅れで目標へ追従させ、
        パターン切替時に位置が跳ばないようにする。戻り値は (位置%, 速度%)。"""
        c = self.cfg
        p = c.patterns[self._mode]
        amp_t = p.amplitude * (0.6 + 0.4 * a)
        freq_t = p.freq_lo + (p.freq_hi - p.freq_lo) * a
        if amp_t > 0:
            # デバイス最大速度を超えないよう周波数を制限
            freq_t = min(freq_t, c.max_stroke_speed / (2 * math.pi * amp_t))

        if dt > 0:
            alpha = 1.0 - math.exp(-dt / c.param_tau)
            prev_center, prev_amp = self._center, self._amplitude
            self._center += (p.center - self._center) * alpha
            self._amplitude += (amp_t - self._amplitude) * alpha
            self._freq += (freq_t - self._freq) * alpha
            # パラメータ遷移中の位置変化: d/dt(c + A·sinφ) = ċ + Ȧ·sinφ + A·ω·cosφ
            self._param_vel = (abs(self._center - prev_center) + abs(self._amplitude - prev_amp)) / dt
            # 遷移中の (まだ大きい) 現在振幅に対しても最大速度を超えないよう周波数を抑える
            if self._amplitude > 1e-6:
                headroom = max(0.0, c.max_stroke_speed - self._param_vel)
                self._freq = min(self._freq, headroom / (2 * math.pi * self._amplitude))
            self._phase = (self._phase + 2 * math.pi * self._freq * dt) % (2 * math.pi)
        # dt=0 (入力受信直後の再計算) では直前のパラメータ速度を使う

        position = self._center + self._amplitude * math.sin(self._phase)
        # |ċ| + |Ȧ| + A·ω は位置の変化速度の上界。ファームはこれを速度上限として追従する
        peak_vel = 2 * math.pi * self._freq * self._amplitude + self._param_vel
        return position, 100.0 * peak_vel / c.max_stroke_speed

    def _pwm_target(self, a: float) -> float:
        c = self.cfg
        if a < c.pwm_deadzone:
            return 0.0
        gain = c.patterns[self._mode].vib_gain
        if gain <= 0:
            return 0.0
        return c.pwm_min + (255 - c.pwm_min) * (a**c.pwm_gamma) * gain

    def _reset_motion(self, position: float) -> None:
        self._center, self._amplitude, self._freq, self._pwm = position, 0.0, 0.0, 0.0
