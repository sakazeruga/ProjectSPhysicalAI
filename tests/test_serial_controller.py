import asyncio
import re

from interfaces import ActuatorCommand, LatestSlot
from serial_controller import SerialConfig, SerialController, encode_frame

FRAME_RE = re.compile(rb"^\$(HAP,\d+,\d+,\d+,\d+)\*([0-9A-F]{2})\n$")


def checksum(body: bytes) -> int:
    cs = 0
    for b in body:
        cs ^= b
    return cs


def test_encode_frame_format_and_checksum():
    frame = encode_frame(42, ActuatorCommand(180, 63.5, 78.0, "X"))
    assert frame == b"$HAP,42,180,635,780*69\n"
    m = FRAME_RE.match(frame)
    assert m and checksum(m.group(1)) == int(m.group(2), 16)


def test_encode_frame_wraps_seq_and_rounds():
    assert encode_frame(65536 + 3, ActuatorCommand(0, 12.34, 99.96, "X")).startswith(b"$HAP,3,0,123,1000*")


def test_latest_slot_keeps_only_newest():
    async def go():
        slot = LatestSlot()
        slot.put(1)
        slot.put(2)
        slot.put(3)
        assert await slot.get(0.01) == 3
        assert await slot.get(0.01) is None
        return slot.dropped

    assert asyncio.run(go()) == 2


def run_controller(port: str, seconds: float, **cfg):
    """SerialController を seconds 秒動かし、(stats, connected中だったか) を返す。"""

    async def go():
        ctl = SerialController(SerialConfig(port=port, boot_delay=0.0, **cfg))
        await ctl.start()
        connected = False
        t_end = asyncio.get_running_loop().time() + seconds
        i = 0
        while asyncio.get_running_loop().time() < t_end:
            ctl.submit(ActuatorCommand(i % 256, 50.0, 10.0, "TEST"))
            connected |= ctl.is_connected
            i += 1
            await asyncio.sleep(0.01)
        stats = ctl.stats()
        await asyncio.wait_for(ctl.stop(), 2.0)
        return stats, connected

    return asyncio.run(go())


def test_loopback_sends_frames():
    stats, connected = run_controller("loop://", 0.5)
    assert connected
    assert stats["frames"] >= 20
    assert stats["reconnects"] == 0


def test_missing_port_retries_with_backoff_and_stops_cleanly():
    stats, connected = run_controller("COM250", 0.5, reconnect_min=0.05, reconnect_max=0.1)
    assert not connected
    assert stats["frames"] == 0
    assert stats["reconnects"] >= 3


def test_keepalive_resends_last_command():
    async def go():
        ctl = SerialController(SerialConfig(port="loop://", boot_delay=0.0, keepalive_interval=0.05))
        await ctl.start()
        ctl.submit(ActuatorCommand(10, 20.0, 30.0, "TEST"))
        await asyncio.sleep(0.6)  # 新しい指令なし
        frames = ctl.stats()["frames"]
        await ctl.stop()
        return frames

    assert asyncio.run(go()) >= 8
