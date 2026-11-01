"""
タッチケア用 AI エージェント: 相手の言葉 → 感情推定 → 応答感情 → AffectiveState。

    テキスト ─▶ EmotionEstimator ─▶ 8感情 ─▶ user_affect() ─▶ care_response() ─▶ StateMapper
    (相手の発話)  (日本語BERT推論)                  (相手の感情)    (ロボットの感情)

相手の感情をそのまま真似ると、怒っている相手に激しく触れてしまう。そこで care_response() で
「相手の感情に対してロボットがどう振る舞うか」に変換してから状態変換層へ渡す。

「やめて」等の拒否はモデルの推論に頼らず、語彙照合で必ず停止する (安全判断を確率モデルに任せない)。
"""
from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace

from interfaces import AffectiveState, AffectSink, InputSource, now
from state_mapper import MapperConfig, Mode, PatternSpec

logger = logging.getLogger(__name__)

MODEL_NAME = "patrickramos/bert-base-japanese-v2-wrime-fine-tune"  # CC BY-SA 3.0, WRIME で学習
EMOTIONS = ("joy", "sadness", "anticipation", "surprise", "anger", "fear", "disgust", "trust")
MAX_INTENSITY = 3.0  # WRIME の感情強度スケール 0..3

# Plutchik の8感情を Russell の円環モデル上に置いた座標 (valence, arousal)
EMOTION_COORDS: dict[str, tuple[float, float]] = {
    "joy": (0.9, 0.6),
    "trust": (0.6, 0.3),
    "anticipation": (0.4, 0.6),
    "surprise": (0.0, 0.9),
    "anger": (-0.7, 0.9),
    "fear": (-0.7, 0.8),
    "disgust": (-0.7, 0.5),
    "sadness": (-0.8, 0.15),
}

STOP_WORDS = ("やめて", "止めて", "とめて", "痛い", "いたい", "ストップ", "離して", "嫌だ")
RESUME_WORDS = ("続けて", "もう大丈夫", "再開")


# ---------------------------------------------------------------------------
# 感情 → 応答
# ---------------------------------------------------------------------------
def user_affect(emotions: dict[str, float]) -> tuple[float, float]:
    """8感情の強度 (0..3) → 相手の (arousal, valence)。強度を重みとする円環上の重心。"""
    weights = {k: max(0.0, min(emotions.get(k, 0.0), MAX_INTENSITY)) / MAX_INTENSITY for k in EMOTIONS}
    total = sum(weights.values())
    if total < 1e-6:
        return 0.0, 0.0
    valence = sum(weights[k] * EMOTION_COORDS[k][0] for k in EMOTIONS) / total
    arousal = sum(weights[k] * EMOTION_COORDS[k][1] for k in EMOTIONS) / total
    # 重心は強度の絶対量を失うので、最も強い感情の強さで縮める (平静な文は中立に寄る)
    strength = max(weights.values())
    return arousal * strength ** 0.5, valence * strength ** 0.5


@dataclass(frozen=True, slots=True)
class CareResponse:
    """ロボット側の振る舞い。arousal / valence は StateMapper のモードを選ぶ値。"""

    name: str  # 表示名
    arousal: float
    valence: float
    heartbeat: bool = False  # 心拍状の振動を周期的に送る


STROKE = CareResponse("ゆっくりなでる", 0.25, 0.6)
SOOTHE = CareResponse("ゆっくり長くなだめる", 0.1, 0.6)
ACCOMPANY = CareResponse("手を止めて寄り添う", 0.15, -0.6, heartbeat=True)
REJOICE = CareResponse("軽やかに一緒に喜ぶ", 0.7, 0.7)
WATCH = CareResponse("そっと触れて見守る", 0.2, 0.0)


def care_response(arousal: float, valence: float) -> CareResponse:
    """相手の感情象限 → ロボットの振る舞い。不快には同調せず、落ち着かせる側へ返す。"""
    if valence <= -0.2:
        return SOOTHE if arousal >= 0.42 else ACCOMPANY
    if valence >= 0.2:
        return REJOICE if arousal >= 0.5 else STROKE
    return WATCH


def safety_event(text: str) -> str | None:
    """拒否・再開の語彙照合。拒否を優先する ("やめて、もう大丈夫" は停止)。"""
    if any(w in text for w in STOP_WORDS):
        return "stop"
    if any(w in text for w in RESUME_WORDS):
        return "resume"
    return None


def care_mapper_config() -> MapperConfig:
    """なでる動作用の StateMapper 設定。

    前腕 20cm を 100% に割り当てる。max_stroke_speed=50%/s が 10cm/s に相当し、心地よいタッチに
    反応する C 触覚線維の至適速度帯 (1〜10cm/s; Löken et al., 2009) の上限を超えない。
    """
    patterns = {
        Mode.IDLE: PatternSpec(center=50.0, amplitude=0.0, freq_lo=0.0, freq_hi=0.0, vib_gain=0.0),
        Mode.NEUTRAL: PatternSpec(center=50.0, amplitude=22.0, freq_lo=0.10, freq_hi=0.20, vib_gain=0.0),
        Mode.CALM: PatternSpec(center=50.0, amplitude=60.0, freq_lo=0.08, freq_hi=0.20, vib_gain=0.35),
        Mode.EXCITED: PatternSpec(center=55.0, amplitude=18.0, freq_lo=0.5, freq_hi=0.9, vib_gain=1.0),
        Mode.TENSE: PatternSpec(center=50.0, amplitude=10.0, freq_lo=0.2, freq_hi=0.4, vib_gain=0.2),
        Mode.GLOOMY: PatternSpec(center=45.0, amplitude=2.0, freq_lo=0.05, freq_hi=0.1, vib_gain=0.0),
    }
    return replace(MapperConfig(), patterns=patterns, max_stroke_speed=50.0, pwm_min=40, home_position=0.0)


# ---------------------------------------------------------------------------
# 感情推定 (推論)
# ---------------------------------------------------------------------------
class EmotionEstimator:
    """日本語 BERT による8感情の強度推定。初回に Hugging Face からモデルを取得する。"""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # 依存が重いので、シミュレータや実機だけ使うときに読み込まないよう遅延 import する
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name, use_safetensors=True).eval()
        labels = self._model.config.id2label
        # 出力 16 次元のうち書き手 (発話者本人) の感情を使う
        self._index = {labels[i].removeprefix("writer_"): i for i in labels if labels[i].startswith("writer_")}

    def estimate(self, text: str) -> dict[str, float]:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with self._torch.inference_mode():
            logits = self._model(**inputs).logits[0]
        return {name: float(logits[i]) for name, i in self._index.items()}


# ---------------------------------------------------------------------------
# 対話 1 発話ぶんの判断
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CareDecision:
    text: str
    emotions: dict[str, float]
    user_arousal: float
    user_valence: float
    response: CareResponse | None  # None は stop / resume のみ
    event: str | None

    def affect(self) -> tuple[float, float]:
        r = self.response
        return (r.arousal, r.valence) if r else (0.0, 0.0)


class CarePolicy:
    """発話を受け取り、感情推定と安全語彙からロボットの振る舞いを決める。"""

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def decide(self, text: str) -> CareDecision:
        event = safety_event(text)
        emotions = self.estimator.estimate(text)
        a, v = user_affect(emotions)
        if event == "stop":
            response = None
        elif event == "resume":
            response = STROKE  # 「続けて」は感情の推定より言葉どおりの依頼を優先する
        else:
            response = care_response(a, v)
        return CareDecision(text, emotions, a, v, response, event)


# ---------------------------------------------------------------------------
# 入力層
# ---------------------------------------------------------------------------
class CareSession:
    """判断を時間軸へ展開する。リアルタイム実行と動画生成の両方が同じ流れを使う。

    StateMapper の入力途絶判定に掛からないよう、最後の判断を period 秒ごとに再送する。
    """

    def __init__(self, policy: CarePolicy, period: float = 0.5, heartbeat_period: float = 1.1) -> None:
        self.policy = policy
        self.period = period
        self.heartbeat_period = heartbeat_period
        self.decision: CareDecision | None = None
        self._stopped = False
        self._next_send = 0.0
        self._next_beat = 0.0

    def accept(self, decision: CareDecision, t: float) -> AffectiveState:
        d = decision
        logger.info(
            "CARE %r user(a=%.2f v=%+.2f) -> %s event=%s",
            d.text, d.user_arousal, d.user_valence, d.response.name if d.response else "-", d.event,
        )
        if d.event == "stop":
            self._stopped = True
        elif d.event == "resume":
            self._stopped = False
        self.decision = d
        self._next_send = t + self.period
        self._next_beat = t + self.heartbeat_period
        a, v = d.affect()
        return AffectiveState(a, v, d.event, t, "care_agent")

    def tick(self, t: float) -> AffectiveState | None:
        """再送時刻になっていれば次の AffectiveState を返す。"""
        d = self.decision
        if d is None or self._stopped or t < self._next_send:
            return None
        self._next_send = t + self.period
        event = None
        if d.response and d.response.heartbeat and t >= self._next_beat:
            event = "heartbeat"
            self._next_beat = t + self.heartbeat_period
        a, v = d.affect()
        return AffectiveState(a, v, event, t, "care_agent")


class CareAgentInput(InputSource):
    """発話 (標準入力の 1 行ごと) を AffectiveState の流れへ変換する入力源。"""

    def __init__(self, session: CareSession, lines: Iterable[str] | None = None) -> None:
        self.session = session
        self.lines = lines

    async def run(self, sink: AffectSink) -> None:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self._read_lines(sink))
            tg.create_task(self._keepalive(sink))

    async def _read_lines(self, sink: AffectSink) -> None:
        source = iter(self.lines) if self.lines is not None else iter(sys.stdin)
        while True:
            line = await asyncio.to_thread(next, source, None)
            if line is None:
                return
            text = line.strip()
            if text:
                decision = await asyncio.to_thread(self.session.policy.decide, text)
                sink(self.session.accept(decision, now()))

    async def _keepalive(self, sink: AffectSink) -> None:
        while True:
            await asyncio.sleep(0.05)
            state = self.session.tick(now())
            if state is not None:
                sink(state)
