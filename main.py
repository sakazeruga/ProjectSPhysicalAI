"""
Physical AI 制御基盤 エントリポイント。

    python main.py                                   # WebSocket 入力 → ESP32 自動検出
    python main.py --serial-port COM5                # ポート指定
    python main.py --input sim --output log          # 実機・エージェントなしで動作確認
    python main.py --log-level DEBUG                 # 入出力フレームを全件表示
    python main.py --input care --output mujoco      # 文字入力で MuJoCo 上のタッチケアを動かす
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from dataclasses import dataclass

from input_handler import SimulatedInputHandler, WebSocketInputHandler
from interfaces import ActuatorCommand, ActuatorOutput, AffectInbox, InputSource, now
from serial_controller import LoggingOutput, SerialConfig, SerialController
from state_mapper import MapperConfig, StateMapper

logger = logging.getLogger("main")


@dataclass
class ControlState:
    last_cmd: ActuatorCommand | None = None


class Ticker:
    """制御周期を専用スレッドから供給する。

    asyncio のタイマは時計分解能 (Windows では 15.6ms) ぶん早く発火し得るため、10ms 周期を
    イベントループ上で刻むと空回りする。Python 3.11+ の time.sleep は Windows でも
    高分解能タイマを使うので、スレッド側で正確に刻んでループへ通知する。
    """

    def __init__(self, period: float, callback) -> None:
        self.period = period
        self._callback = callback
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="control-ticker", daemon=True)

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        next_tick = now()
        while not self._stop.is_set():
            next_tick += self.period
            delay = next_tick - now()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = now()  # 処理落ち時は周期を詰めず次周期から再開
            try:
                self._loop.call_soon_threadsafe(self._callback)
            except RuntimeError:  # ループ終了済み
                return


async def control_loop(inbox: AffectInbox, mapper: StateMapper, output: ActuatorOutput, ctl: ControlState) -> None:
    """周期 tick でパターンを進めつつ、新着入力があれば周期を待たず即座に再計算する。"""
    while True:
        await inbox.wait()
        state, events = inbox.take()
        ctl.last_cmd = mapper.update(now(), state, events)
        output.submit(ctl.last_cmd)


async def telemetry_loop(mapper: StateMapper, output: ActuatorOutput, ctl: ControlState, interval: float) -> None:
    """パラメータの遷移をターミナルで追えるよう、一定間隔で状態を1行表示する。"""
    prev_frames = 0
    while True:
        await asyncio.sleep(interval)
        last_cmd = ctl.last_cmd
        if last_cmd is None:
            continue
        a, v = mapper.effective_affect
        s = output.stats()
        frames = s.get("frames", 0)
        logger.info(
            "TEL mode=%-9s in(a=%.2f v=%+.2f) out(pwm=%3d pos=%5.1f%% spd=%5.1f%%) "
            "link=%s lat=%.1fms tx=%.0f/s drop=%d",
            mapper.mode, a, v,
            last_cmd.vibration_intensity, last_cmd.linear_position, last_cmd.translation_speed,
            "OK" if output.is_connected else "DOWN", s.get("latency_ms", 0.0),
            (frames - prev_frames) / interval, s.get("dropped", 0),
        )
        prev_frames = frames


def build_input(args) -> InputSource:
    if args.input == "sim":
        return SimulatedInputHandler(seed=args.seed)
    if args.input == "care":
        from care_agent import CareAgentInput, CarePolicy, CareSession, EmotionEstimator

        return CareAgentInput(CareSession(CarePolicy(EmotionEstimator())))
    return WebSocketInputHandler(args.host, args.port)


def build_output(args) -> ActuatorOutput:
    if args.output == "log":
        return LoggingOutput()
    if args.output == "mujoco":
        from sim.mujoco_output import MujocoOutput

        return MujocoOutput(viewer=not args.no_viewer)
    return SerialController(SerialConfig(port=args.serial_port, baudrate=args.baud))


def build_mapper(args) -> StateMapper:
    if args.input == "care":
        from care_agent import care_mapper_config

        return StateMapper(care_mapper_config())
    return StateMapper(MapperConfig())


async def run(args) -> None:
    inbox = AffectInbox()
    mapper = build_mapper(args)
    source = build_input(args)
    output = build_output(args)
    ctl = ControlState()
    ticker = Ticker(1.0 / args.rate, inbox.tick)

    await output.start()
    ticker.start()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(source.run(inbox.publish), name="input")
            tg.create_task(control_loop(inbox, mapper, output, ctl), name="control")
            tg.create_task(telemetry_loop(mapper, output, ctl, args.log_interval), name="telemetry")
    finally:
        ticker.stop()
        await output.stop()
        logger.info("Shutdown complete")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Affective state -> haptics bridge")
    p.add_argument("--input", choices=["ws", "sim", "care"], default="ws")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--output", choices=["serial", "log", "mujoco"], default="serial")
    p.add_argument("--no-viewer", action="store_true", help="run --output mujoco without a window")
    p.add_argument("--serial-port", default="auto", help="COMx / /dev/ttyUSBx or 'auto'")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--rate", type=float, default=100.0, help="control loop rate [Hz]")
    p.add_argument("--log-interval", type=float, default=0.5, help="telemetry interval [s]")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--seed", type=int, default=None, help="seed for --input sim")
    return p.parse_args(argv)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-17s %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv=None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
