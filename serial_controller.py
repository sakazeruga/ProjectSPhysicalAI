"""
出力層: 制御値をシリアル経由でマイコン (ESP32 等) へ非同期送信する。

送信フレーム (ASCII 1行, NMEA 風チェックサム):
    $HAP,<seq>,<pwm>,<pos_x10>,<speed_x10>*<XOR hex>\\n
    例) $HAP,42,180,635,780*69\\n   → seq=42, PWM=180, 位置=63.5%, 速度=78.0%

    - seq       : 0..65535 で循環
    - pos/speed : 0.1% 単位の整数 (0..1000)
    - checksum  : '$' と '*' の間の全バイトの XOR

マイコン側は一定時間 (例: 300ms) フレームが途絶えたら自律停止するウォッチドッグを持つ想定。
本モジュールは新しい指令がなくても keepalive_interval ごとに最終指令を再送する。

pyserial はブロッキング API なので、送信・受信それぞれ専用の 1 スレッド executor で実行し
イベントループを止めない (送信順序は単一ワーカーで保証)。
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from interfaces import ActuatorCommand, ActuatorOutput, LatestSlot, now

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # ログ出力のみで使う場合は pyserial 不要
    serial = None
    list_ports = None

logger = logging.getLogger(__name__)

# ESP32 開発ボードでよく使われる USB-UART の Vendor ID
KNOWN_VIDS = {
    0x10C4: "CP210x",
    0x1A86: "CH34x",
    0x303A: "Espressif USB",
    0x0403: "FTDI",
}


def encode_frame(seq: int, cmd: ActuatorCommand) -> bytes:
    body = "HAP,%d,%d,%d,%d" % (
        seq & 0xFFFF,
        cmd.vibration_intensity,
        round(cmd.linear_position * 10),
        round(cmd.translation_speed * 10),
    )
    checksum = 0
    for b in body.encode("ascii"):
        checksum ^= b
    return f"${body}*{checksum:02X}\n".encode("ascii")


@dataclass(slots=True)
class SerialConfig:
    port: str = "auto"  # "auto" なら KNOWN_VIDS から自動検出
    baudrate: int = 115200
    read_timeout: float = 0.1
    write_timeout: float = 0.05
    suppress_reset: bool = True  # オープン時に DTR/RTS を下げ、ESP32 の自動リセットを避ける
    boot_delay: float = 1.5  # オープン後、ファームウェア起動を待つ時間 [s]
    reconnect_min: float = 0.5  # 再接続バックオフ [s]
    reconnect_max: float = 5.0
    keepalive_interval: float = 0.1  # 新指令がなくても最終指令を再送する間隔 [s]
    home_position: float = 0.0
    extra_vids: dict[int, str] = field(default_factory=dict)


class SerialController(ActuatorOutput):
    def __init__(self, config: SerialConfig | None = None) -> None:
        if serial is None:
            raise RuntimeError("pyserial is required: pip install -r requirements.txt")
        self.cfg = config or SerialConfig()
        self._slot: LatestSlot[ActuatorCommand] = LatestSlot()
        self._tx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="serial-tx")
        self._rx = ThreadPoolExecutor(max_workers=1, thread_name_prefix="serial-rx")
        self._ser: serial.Serial | None = None
        self._port_name: str | None = None
        self._task: asyncio.Task | None = None
        self._closing = False
        self._seq = 0
        self._last_sent: ActuatorCommand | None = None
        self._frames_sent = 0
        self._reconnects = 0
        self._latency_ms = 0.0

    # ------------------------------------------------------------- interface
    async def start(self) -> None:
        self._task = asyncio.create_task(self._supervise(), name="serial-supervisor")

    def submit(self, cmd: ActuatorCommand) -> None:
        self._slot.put(cmd)

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def stats(self) -> dict[str, float]:
        return {
            "frames": self._frames_sent,
            "dropped": self._slot.dropped,
            "reconnects": self._reconnects,
            "latency_ms": self._latency_ms,
        }

    async def stop(self) -> None:
        self._closing = True
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
        loop = asyncio.get_running_loop()
        if self.is_connected:
            # 単一ワーカーなので、送信中のフレームの後に必ず安全指令が送られる
            safe = ActuatorCommand.safe(self.cfg.home_position)
            try:
                await loop.run_in_executor(self._tx, self._write, encode_frame(self._next_seq(), safe))
                logger.info("Sent SAFE_STOP to %s", self._port_name)
            except Exception as e:  # 切断済み等。停止処理は継続する
                logger.warning("Could not send SAFE_STOP: %s", e)
        await loop.run_in_executor(self._tx, self._close_port)
        self._tx.shutdown(wait=False)
        self._rx.shutdown(wait=False)

    # ------------------------------------------------------------ supervisor
    async def _supervise(self) -> None:
        """接続 → 送受信 → 例外で切断 → バックオフして再接続、を繰り返す。"""
        loop = asyncio.get_running_loop()
        backoff = self.cfg.reconnect_min
        while not self._closing:
            try:
                await loop.run_in_executor(self._tx, self._open_port)
                if self.cfg.boot_delay > 0:
                    await asyncio.sleep(self.cfg.boot_delay)
                    await loop.run_in_executor(self._tx, self._discard_boot_output)
                logger.info("Serial connected: %s @ %d bps", self._port_name, self.cfg.baudrate)
                backoff = self.cfg.reconnect_min
                await self._io_loop()
            except asyncio.CancelledError:
                raise
            except (serial.SerialException, OSError) as e:
                if self._port_name:
                    logger.warning("Serial error on %s: %s", self._port_name, e)
                else:
                    logger.warning("Serial unavailable: %s", e)
            except Exception:
                logger.exception("Unexpected error in serial loop")
            finally:
                if not self._closing:
                    await loop.run_in_executor(self._tx, self._close_port)

            if self._closing:
                break
            self._reconnects += 1
            logger.info("Reconnecting in %.1f s ...", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.cfg.reconnect_max)

    async def _io_loop(self) -> None:
        reader = asyncio.create_task(self._read_loop(), name="serial-reader")
        writer = asyncio.create_task(self._write_loop(), name="serial-writer")
        try:
            done, _ = await asyncio.wait({reader, writer}, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                t.result()  # どちらかの例外を supervisor へ伝播
        finally:
            for t in (reader, writer):
                t.cancel()
            await asyncio.gather(reader, writer, return_exceptions=True)

    async def _write_loop(self) -> None:
        loop = asyncio.get_running_loop()
        last_write = 0.0
        while True:
            cmd = await self._slot.get(timeout=self.cfg.keepalive_interval)
            if cmd is None:
                cmd = self._last_sent  # keepalive: 最終指令を再送
                if cmd is None:
                    continue
            elif _same_output(cmd, self._last_sent) and now() - last_write < self.cfg.keepalive_interval:
                continue  # 変化なしなら送らない (帯域節約)

            await loop.run_in_executor(self._tx, self._write, encode_frame(self._next_seq(), cmd))
            last_write = now()
            if cmd.source_received_at is not None and cmd is not self._last_sent:
                self._latency_ms = (last_write - cmd.source_received_at) * 1000
                logger.debug("Input->serial latency %.2f ms", self._latency_ms)
            self._last_sent = cmd
            self._frames_sent += 1

    async def _read_loop(self) -> None:
        """デバイスからの応答/ログ行を読む。切断検知も兼ねる。"""
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(self._rx, self._readline)
            if not line:
                continue
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("ERR"):
                logger.warning("DEVICE %s", text)
            else:
                logger.debug("DEVICE %s", text)

    # ------------------------------------------------------- blocking (threads)
    def _resolve_port(self) -> str:
        if self.cfg.port.lower() != "auto":
            return self.cfg.port
        vids = {**KNOWN_VIDS, **self.cfg.extra_vids}
        for p in list_ports.comports():
            if p.vid in vids:
                logger.info("Auto-detected %s (%s: %s)", p.device, vids[p.vid], p.description)
                return p.device
        raise serial.SerialException("no ESP32-like serial device found (use --serial-port)")

    def _open_port(self) -> None:
        self._port_name = None
        port = self._resolve_port()
        # serial_for_url なので "COM5" 等に加え "socket://esp32.local:3333" (Wi-Fi 経由) や
        # "loop://" (ループバック試験) も指定できる
        ser = serial.serial_for_url(port, do_not_open=True)
        ser.baudrate = self.cfg.baudrate
        ser.timeout = self.cfg.read_timeout
        ser.write_timeout = self.cfg.write_timeout
        if self.cfg.suppress_reset:
            ser.dtr = False
            ser.rts = False
        ser.open()
        self._ser = ser
        self._port_name = port

    def _close_port(self) -> None:
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _discard_boot_output(self) -> None:
        ser = self._ser
        if ser is None:
            raise serial.SerialException("port closed during boot delay")
        ser.reset_input_buffer()

    def _write(self, frame: bytes) -> None:
        ser = self._ser
        if ser is None:
            raise serial.SerialException("port is not open")
        ser.write(frame)

    def _readline(self) -> bytes:
        ser = self._ser
        if ser is None:
            raise serial.SerialException("port is not open")
        return ser.readline()

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq


def _same_output(a: ActuatorCommand, b: ActuatorCommand | None) -> bool:
    return (
        b is not None
        and a.vibration_intensity == b.vibration_intensity
        and a.linear_position == b.linear_position
        and a.translation_speed == b.translation_speed
    )


class LoggingOutput(ActuatorOutput):
    """実機なしで動作確認するためのドライラン出力。送信フレームを DEBUG ログに出す。

    シミュレータ (Gazebo 等) 向け出力も同じ ActuatorOutput を実装すれば差し替えられる。
    """

    def __init__(self) -> None:
        self._seq = 0
        self._frames = 0
        self._latency_ms = 0.0

    async def start(self) -> None:
        logger.info("Dry-run output: frames are logged, not sent")

    def submit(self, cmd: ActuatorCommand) -> None:
        self._seq = (self._seq + 1) & 0xFFFF
        self._frames += 1
        if cmd.source_received_at is not None:
            self._latency_ms = (now() - cmd.source_received_at) * 1000
        logger.debug("OUT %s", encode_frame(self._seq, cmd).decode().strip())

    async def stop(self) -> None:
        safe = ActuatorCommand.safe()
        logger.info("OUT (shutdown) %s", encode_frame(self._seq + 1, safe).decode().strip())

    @property
    def is_connected(self) -> bool:
        return True

    def stats(self) -> dict[str, float]:
        return {"frames": self._frames, "dropped": 0, "reconnects": 0, "latency_ms": self._latency_ms}
