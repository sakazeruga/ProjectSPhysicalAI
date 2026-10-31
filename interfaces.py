"""
層間で共有するデータ型と抽象インターフェース。

    InputSource ──publish──▶ AffectInbox ──▶ StateMapper ──▶ ActuatorOutput
     (WebSocket/Sim)                          (状態機械)      (Serial/Log/Gazebo...)

各層はこのモジュールの型にのみ依存する。ActuatorOutput を Gazebo / ROS 2 ブリッジ等の
実装に差し替えても、入力層・状態変換層は変更不要。
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


def now() -> float:
    """パイプライン共通の単調時計 [s]。

    Windows の time.monotonic() は分解能が約 15.6ms と粗いため perf_counter を使う。
    """
    return time.perf_counter()


# ---------------------------------------------------------------------------
# データ型
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AffectiveState:
    """AIエージェントから受信した感情パラメータ (検証・クランプ済み)。"""

    arousal: float  # 覚醒度 0.0 .. 1.0
    valence: float  # 感情価 -1.0 .. 1.0
    event_trigger: str | None = None
    received_at: float = field(default_factory=now)
    source: str = "unknown"


@dataclass(frozen=True, slots=True)
class ActuatorCommand:
    """物理アクチュエータへの制御値 (ハードウェア非依存の正規化表現)。"""

    vibration_intensity: int  # PWM 0 .. 255
    linear_position: float  # ストローク位置 0.0 .. 100.0 [%]
    translation_speed: float  # 駆動速度 0.0 .. 100.0 [% of device max speed]
    mode: str
    # この指令を生んだ入力の受信時刻。入力→出力の遅延計測に使う (入力直後の指令のみ設定)
    source_received_at: float | None = None
    created_at: float = field(default_factory=now)

    @classmethod
    def safe(cls, home_position: float = 0.0, speed: float = 20.0) -> ActuatorCommand:
        """振動停止 + ホーム位置へ低速復帰する安全指令。"""
        return cls(0, home_position, speed, "SAFE_STOP")


# ---------------------------------------------------------------------------
# 低遅延キュー
# ---------------------------------------------------------------------------
class LatestSlot(Generic[T]):
    """最新値のみを保持するスロット。

    制御系では古い指令をキューに溜めて順に処理すると遅延が蓄積するため、
    未消費の値は新しい値で上書き (drop) する。
    """

    def __init__(self) -> None:
        self._value: T | None = None
        self._event = asyncio.Event()
        self.dropped = 0

    def put(self, value: T) -> None:
        if self._event.is_set():
            self.dropped += 1
        self._value = value
        self._event.set()

    async def get(self, timeout: float | None = None) -> T | None:
        """新しい値を待って取り出す。timeout 経過時は None。"""
        if not self._event.is_set():
            try:
                await asyncio.wait_for(self._event.wait(), timeout)
            except TimeoutError:
                return None
        self._event.clear()
        return self._value


class AffectInbox:
    """入力層 → 状態変換層の受け渡し口。

    連続値 (arousal/valence) は最新値のみ保持し、離散イベント (event_trigger) は
    取りこぼさないよう別キューに積む。
    """

    def __init__(self, max_events: int = 64) -> None:
        self._latest: AffectiveState | None = None
        self._events: deque[str] = deque(maxlen=max_events)
        self._wakeup = asyncio.Event()

    def publish(self, state: AffectiveState) -> None:
        self._latest = state
        if state.event_trigger:
            self._events.append(state.event_trigger)
        self._wakeup.set()

    def tick(self) -> None:
        """データなしで制御ループを起こす (周期実行用)。"""
        self._wakeup.set()

    async def wait(self) -> None:
        """新着入力または tick() まで待つ。"""
        await self._wakeup.wait()

    def take(self) -> tuple[AffectiveState | None, list[str]]:
        """新着の状態 (なければ None) と、溜まったイベント列を取り出す。"""
        self._wakeup.clear()
        state, self._latest = self._latest, None
        events = list(self._events)
        self._events.clear()
        return state, events


AffectSink = Callable[[AffectiveState], None]


# ---------------------------------------------------------------------------
# 抽象インターフェース
# ---------------------------------------------------------------------------
class InputSource(ABC):
    """感情パラメータの入力元 (WebSocket, シミュレータ, ROS トピック等)。"""

    @abstractmethod
    async def run(self, sink: AffectSink) -> None:
        """キャンセルされるまで入力を受け、受信のたびに sink(state) を呼ぶ。"""


class ActuatorOutput(ABC):
    """制御値の出力先 (シリアル接続の実機, Gazebo 等のシミュレータ, ログ)。

    submit() はノンブロッキングで、送信は実装側のタスクが非同期に行う。
    """

    @abstractmethod
    async def start(self) -> None:
        """接続・送信タスクを開始する (接続失敗で例外を投げず、内部で再試行すること)。"""

    @abstractmethod
    def submit(self, cmd: ActuatorCommand) -> None:
        """最新の指令を渡す。未送信の古い指令は破棄してよい。"""

    @abstractmethod
    async def stop(self) -> None:
        """安全指令を送ってから切断する。"""

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    def stats(self) -> dict[str, float]:
        """テレメトリ用の統計値 (任意)。"""
        return {}
