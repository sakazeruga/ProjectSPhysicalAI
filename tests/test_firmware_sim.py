"""
ファームウェア (haptics_esp32.ino) を PC 上のシミュレータで動かすエンドツーエンドテスト。

- 生ソケットをホスト役にしてファームの挙動 (追従・ウォッチドッグ・チェックサム・速度制限) を検査
- 実際のホスト (main.py のパイプライン) を socket:// で接続して結合動作を検査

C++ コンパイラ (g++ / clang++ / `pip install ziglang`) が無い環境ではスキップする。
"""
from __future__ import annotations

import asyncio
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SIM_DIR = ROOT / "firmware" / "sim"

from interfaces import ActuatorCommand  # noqa: E402
from serial_controller import encode_frame  # noqa: E402


def _compiler() -> list[str] | None:
    for name in ("g++", "clang++"):
        if path := shutil.which(name):
            return [path]
    try:
        import ziglang  # noqa: F401
    except ImportError:
        return None
    return [sys.executable, "-m", "ziglang", "c++"]


@pytest.fixture(scope="session")
def sim_exe(tmp_path_factory) -> dict[int, Path]:
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler (install g++/clang++ or `pip install ziglang`)")
    out = tmp_path_factory.mktemp("sim")
    exes = {}
    for backend in (1, 2):
        exe = out / f"sim_b{backend}{'.exe' if sys.platform == 'win32' else ''}"
        cmd = cxx + ["-std=gnu++17", "-O2", f"-DACTUATOR_BACKEND={backend}", "-I", str(SIM_DIR),
                     str(SIM_DIR / "sim_main.cpp"), "-o", str(exe)]
        if sys.platform == "win32":
            cmd.append("-lws2_32")
        subprocess.run(cmd, check=True, capture_output=True)
        exes[backend] = exe
    return exes


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Sample:
    at: float  # ホスト側の受信時刻 (time.monotonic)。パイプのバッファリングで揺らぐ
    ms: int  # デバイス側の時刻 (millis)。速度計測はこちらを使う
    vib: int
    pos: float
    tgt: float
    fs: int
    conn: int


class Sim:
    """シミュレータプロセス。stdout のトレース行を Sample として蓄積する。"""

    def __init__(self, exe: Path, port: int):
        self.port = port
        self.samples: list[Sample] = []
        self._ready = threading.Event()
        self.proc = subprocess.Popen([str(exe), str(port), "10"], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()
        assert self._ready.wait(5), "simulator did not start"

    def _read(self):
        for line in self.proc.stdout:
            if line.startswith("READY"):
                self._ready.set()
            elif line.startswith("T "):
                parts = line.split()
                kv = dict(p.split("=") for p in parts[2:])
                self.samples.append(Sample(time.monotonic(), int(parts[1]), int(kv["vib"]), float(kv["pos"]),
                                           float(kv["tgt"]), int(kv["fs"]), int(kv["conn"])))

    def last(self) -> Sample:
        return self.samples[-1]

    def since(self, t: float) -> list[Sample]:
        return [s for s in self.samples if s.at >= t]

    def wait_for(self, pred, timeout: float) -> Sample:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.samples and pred(self.samples[-1]):
                return self.samples[-1]
            time.sleep(0.005)
        raise AssertionError(f"condition not met within {timeout}s; last={self.samples[-1] if self.samples else None}")

    def close(self):
        self.proc.kill()
        self.proc.wait()


class RawHost:
    """ホスト役の生ソケット。デバイスから届いた行を蓄積する。"""

    def __init__(self, port: int):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.lines: list[str] = []
        self.seq = 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        try:
            while chunk := self.sock.recv(1024):
                buf += chunk
                *done, buf = buf.split(b"\n")
                self.lines += [d.decode().strip() for d in done]
        except OSError:
            pass

    def frame(self, pwm: int, pos: float, speed: float) -> bytes:
        self.seq += 1
        return encode_frame(self.seq, ActuatorCommand(pwm, pos, speed, "TEST"))

    def stream(self, pwm: int, pos: float, speed: float, seconds: float, hz: float = 100):
        for _ in range(int(seconds * hz)):
            self.sock.sendall(self.frame(pwm, pos, speed))
            time.sleep(1 / hz)

    def wait_line(self, prefix: str, timeout: float = 2.0) -> str:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for line in self.lines:
                if line.startswith(prefix):
                    return line
            time.sleep(0.01)
        raise AssertionError(f"no line starting with {prefix!r}; got {self.lines[-5:]}")

    def close(self):
        self.sock.close()


@pytest.fixture(params=[1, 2], ids=["servo", "stepper"])
def sim(request, sim_exe):
    s = Sim(sim_exe[request.param], free_port())
    yield s
    s.close()


# --------------------------------------------------------------------------- ファーム単体
def test_starts_in_failsafe_then_follows_frames(sim):
    sim.wait_for(lambda s: True, 1)
    assert sim.last().fs == 1 and sim.last().vib == 0

    host = RawHost(sim.port)
    host.stream(200, 60.0, 50.0, 1.0)
    s = sim.last()
    assert (s.fs, s.vib) == (0, 200)
    assert s.pos == pytest.approx(60.0, abs=0.3)
    assert host.wait_line("INFO first frame")
    host.close()


def test_watchdog_stops_vibration_and_homes(sim):
    host = RawHost(sim.port)
    host.stream(180, 60.0, 50.0, 1.0)
    stopped_at = time.monotonic()

    tripped = sim.wait_for(lambda s: s.fs == 1, 1.0)
    assert 0.25 <= tripped.at - stopped_at <= 0.45  # WATCHDOG_TIMEOUT_MS=300 + トレース周期
    assert tripped.vib == 0
    assert "ERR WATCHDOG" in host.wait_line("ERR WATCHDOG")

    # ホームへ 20% 速度 (80 %/s) で退避: 60% → 0% は 0.75 s 強
    homed = sim.wait_for(lambda s: s.pos <= 0.3, 2.0)
    assert homed.at - tripped.at >= 0.6

    host.stream(100, 30.0, 50.0, 0.5)  # フレーム再開で自動復帰
    assert sim.last().fs == 0 and sim.last().vib == 100
    assert host.wait_line("INFO resumed")
    host.close()


def test_bad_checksum_is_rejected_and_reported(sim):
    host = RawHost(sim.port)
    host.stream(50, 40.0, 50.0, 0.5)
    bad = bytearray(host.frame(255, 90.0, 100.0))
    bad[6] ^= 0x01
    for _ in range(10):
        host.sock.sendall(bytes(bad))
        time.sleep(0.01)
    assert host.wait_line("ERR CHECKSUM")
    assert sim.last().vib == 50  # 破損フレームは適用されない
    stat = host.wait_line("STAT", timeout=1.5)
    assert "cs=" in stat and "cs=0 " not in stat
    host.close()


def test_speed_limit_from_frame(sim):
    host = RawHost(sim.port)
    host.stream(0, 0.0, 100.0, 0.5)
    t0 = time.monotonic()
    host.stream(0, 100.0, 25.0, 1.6)  # 25% = 100 %/s → 0→100% に約 1 秒
    moving = sim.since(t0)
    started = next(s for s in moving if s.tgt >= 99.9)
    reached = next(s for s in moving if s.pos >= 99.5)
    assert 0.95 <= (reached.ms - started.ms) / 1000 <= 1.15
    # 100ms 窓の平均速度で評価 (10ms 間隔だと ms 量子化で ±20% 揺らぐ)
    speeds = []
    for i, a in enumerate(moving):
        b = next((s for s in moving[i:] if s.ms - a.ms >= 100), None)
        if b:
            speeds.append((b.pos - a.pos) / ((b.ms - a.ms) / 1000))
    assert max(speeds) <= 100 * 1.1
    host.close()


def test_stepper_never_exceeds_device_max_speed(sim_exe):
    """loop() が遅れて取り残されても、追いつく際のステップ間隔は最大速度 (400 %/s) を守る。"""
    if sys.platform != "win32":
        pytest.skip("relies on Windows scheduling to starve the loop")
    sim = Sim(sim_exe[2], free_port())
    try:
        host = RawHost(sim.port)
        host.stream(0, 0.0, 100.0, 0.5)
        t0 = time.monotonic()
        host.stream(0, 100.0, 100.0, 1.0)
        moving = sim.since(t0)
        speeds = [(b.pos - a.pos) / ((b.ms - a.ms) / 1000) for a, b in zip(moving, moving[1:]) if b.ms - a.ms >= 8]
        assert max(speeds) <= 400 * 1.3  # ms 量子化ぶんの余裕
        host.close()
    finally:
        sim.close()


# --------------------------------------------------------------------------- ホストとの結合
def run_host(port: int, seconds: float, driver=None, input_mode="ws"):
    """main.py のパイプラインを seconds 秒動かす。driver(ws_url) は並行実行するコルーチン。"""
    import main

    ws_port = free_port()
    args = main.parse_args(["--input", input_mode, "--port", str(ws_port), "--serial-port",
                            f"socket://127.0.0.1:{port}", "--log-level", "WARNING"])

    async def go():
        task = asyncio.create_task(main.run(args))
        try:
            if driver:
                await asyncio.sleep(0.3)
                await asyncio.wait_for(driver(f"ws://127.0.0.1:{ws_port}"), seconds)
            else:
                await asyncio.sleep(seconds)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())


def test_host_pipeline_end_to_end(sim_exe):
    from websockets.asyncio.client import connect

    sim = Sim(sim_exe[1], free_port())
    marks = {}
    try:
        async def agent(url):
            async with connect(url) as ws:
                await asyncio.sleep(1.5)  # SerialController の boot_delay
                marks["excited"] = time.monotonic()
                for _ in range(25):
                    await ws.send(json.dumps({"arousal": 0.9, "valence": 0.7}))
                    await asyncio.sleep(0.1)
                marks["stop"] = time.monotonic()
                await ws.send(json.dumps({"event_trigger": "stop"}))
                await asyncio.sleep(1.5)
                marks["end"] = time.monotonic()

        run_host(sim.port, 10, agent)
        host_stopped = time.monotonic()

        excited = [s for s in sim.since(marks["excited"] + 1.0) if s.at < marks["stop"]]
        assert all(s.fs == 0 for s in excited)
        assert {s.vib for s in excited} == {230}  # 60 + 195 * 0.9^1.3
        positions = [s.pos for s in excited]
        assert max(positions) - min(positions) > 40  # EXCITED の大振幅ストローク

        stopped = [s for s in sim.since(marks["stop"] + 0.1) if s.at < marks["end"]]
        assert all(s.vib == 0 for s in stopped)
        assert stopped[-1].pos <= 1.0  # SAFE_STOP でホームへ

        # ホスト終了 → SAFE_STOP 送信 → 切断 → ウォッチドッグ
        s = sim.wait_for(lambda s: s.fs == 1, 1.0)
        assert s.at - host_stopped < 1.0 and s.vib == 0
    finally:
        sim.close()


def test_host_reconnects_after_device_restart(sim_exe):
    port = free_port()
    sims = [Sim(sim_exe[1], port)]
    restarted = {}

    def restart_device():
        sims[0].wait_for(lambda s: s.fs == 0, 4.0)  # 1 台目が動き出すのを待つ
        sims[0].close()
        time.sleep(1.0)
        sims.append(Sim(sim_exe[1], port))
        restarted["at"] = time.monotonic()

    t = threading.Thread(target=restart_device)
    t.start()
    try:
        run_host(port, 14, input_mode="sim")
        t.join()
        second = sims[-1]
        assert len(sims) == 2
        live = [s for s in second.samples if s.fs == 0]
        assert live, "host never reconnected to the restarted device"
        # 再接続はバックオフ (最大 5s) + boot_delay (1.5s) 以内
        assert live[0].at - restarted["at"] < 8.0
    finally:
        for s in sims:
            s.close()
