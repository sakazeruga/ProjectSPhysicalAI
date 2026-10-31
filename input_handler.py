"""
入力層: AIエージェントから感情パラメータを受信する。

メッセージ形式 (JSON):
    {"arousal": 0.0~1.0, "valence": -1.0~1.0, "event_trigger": "surprise"}

- 部分更新可: 省略したキーは同一接続の直前値を引き継ぐ (イベントだけ送る等)
- 範囲外の値はクランプして警告、数値でない/非有限値はエラー応答して破棄
- 正常時は応答を返さない (往復を増やさず低遅延を優先)
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import random

from interfaces import AffectiveState, AffectSink, InputSource, now

logger = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 4096
MAX_EVENT_LENGTH = 64


class MessageError(ValueError):
    pass


def _number(data: dict, key: str, lo: float, hi: float, default: float) -> float:
    if key not in data or data[key] is None:
        return default
    raw = data[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise MessageError(f"'{key}' must be a number, got {type(raw).__name__}")
    value = float(raw)
    if not math.isfinite(value):
        raise MessageError(f"'{key}' must be finite")
    if not lo <= value <= hi:
        logger.warning("%s=%.3f out of range [%g, %g]; clamped", key, value, lo, hi)
        value = min(max(value, lo), hi)
    return value


def parse_affect_message(
    raw: str | bytes, previous: AffectiveState | None = None, source: str = "unknown"
) -> AffectiveState:
    """JSON メッセージを検証して AffectiveState にする。不正なら MessageError。"""
    received_at = now()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise MessageError(f"invalid JSON: {e}") from None
    if not isinstance(data, dict):
        raise MessageError("message must be a JSON object")

    arousal = _number(data, "arousal", 0.0, 1.0, previous.arousal if previous else 0.0)
    valence = _number(data, "valence", -1.0, 1.0, previous.valence if previous else 0.0)

    event = data.get("event_trigger")
    if event is not None:
        if not isinstance(event, str):
            raise MessageError("'event_trigger' must be a string")
        event = event.strip()[:MAX_EVENT_LENGTH] or None

    return AffectiveState(arousal, valence, event, received_at, source)


class WebSocketInputHandler(InputSource):
    """WebSocket サーバ。複数クライアントの接続を受け付け、最新値を sink へ流す。"""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.host = host
        self.port = port

    async def run(self, sink: AffectSink) -> None:
        try:
            from websockets.asyncio.server import serve
        except ImportError as e:
            raise RuntimeError("websockets>=13 is required: pip install -r requirements.txt") from e

        async def handler(ws) -> None:
            await self._handle_connection(ws, sink)

        # 圧縮は CPU/遅延コストになるだけなので無効化
        async with serve(
            handler, self.host, self.port, compression=None, max_size=MAX_MESSAGE_BYTES
        ) as server:
            logger.info("WebSocket input listening on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    async def _handle_connection(self, ws, sink: AffectSink) -> None:
        from websockets.exceptions import ConnectionClosed

        peer = "%s:%s" % ws.remote_address[:2] if ws.remote_address else "?"
        source = f"ws:{peer}"
        logger.info("Agent connected: %s", peer)
        previous: AffectiveState | None = None
        try:
            async for message in ws:
                try:
                    state = parse_affect_message(message, previous, source)
                except MessageError as e:
                    logger.warning("Rejected message from %s: %s", peer, e)
                    await ws.send(json.dumps({"error": str(e)}))
                    continue
                previous = state
                logger.debug(
                    "IN  %s arousal=%.3f valence=%+.3f event=%s",
                    peer, state.arousal, state.valence, state.event_trigger,
                )
                sink(state)
        except ConnectionClosed as e:
            logger.warning("Agent connection lost: %s (%s)", peer, e)
        else:
            logger.info("Agent disconnected: %s", peer)


class SimulatedInputHandler(InputSource):
    """エージェント無しで動作確認するための疑似入力 (ランダムウォーク + 時々イベント)。"""

    EVENTS = ("surprise", "heartbeat", "notify")

    def __init__(self, rate_hz: float = 10.0, event_probability: float = 0.01, seed: int | None = None):
        self.period = 1.0 / rate_hz
        self.event_probability = event_probability
        self.rng = random.Random(seed)

    async def run(self, sink: AffectSink) -> None:
        logger.info("Simulated input running at %.0f Hz", 1.0 / self.period)
        arousal, valence = 0.3, 0.0
        while True:
            # 平均回帰つきランダムウォーク
            arousal += 0.05 * (0.4 - arousal) + self.rng.gauss(0, 0.04)
            valence += 0.03 * (0.0 - valence) + self.rng.gauss(0, 0.06)
            arousal = min(max(arousal, 0.0), 1.0)
            valence = min(max(valence, -1.0), 1.0)
            event = None
            if self.rng.random() < self.event_probability:
                event = self.rng.choice(self.EVENTS)
            sink(AffectiveState(arousal, valence, event, now(), "sim"))
            await asyncio.sleep(self.period)
