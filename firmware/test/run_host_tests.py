"""
ファームウェアのパーサ / 軌道生成をホスト PC 上でテストする。

パーサには serial_controller.encode_frame() が実際に生成したフレームと、
それを壊したフレーム (チェックサム不一致・値域外・欠損・ゴミ混入) を流し込み、
Python 側とファーム側のプロトコル解釈が一致することを確認する。

    python firmware/test/run_host_tests.py

C++ コンパイラ (g++ / clang++) が無い場合は `pip install ziglang` で入る zig c++ を使う。
"""
from __future__ import annotations

import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from interfaces import ActuatorCommand  # noqa: E402
from serial_controller import encode_frame  # noqa: E402


def compiler() -> list[str]:
    for name in ("g++", "clang++"):
        if path := shutil.which(name):
            return [path]
    try:
        import ziglang  # noqa: F401
    except ImportError:
        sys.exit("No C++ compiler found. Install g++/clang++ or run: pip install ziglang")
    return [sys.executable, "-m", "ziglang", "c++"]


def build(out_dir: Path) -> Path:
    exe = out_dir / ("host_test.exe" if sys.platform == "win32" else "host_test")
    cmd = compiler() + ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(exe), str(HERE / "host_test.cpp")]
    subprocess.run(cmd, check=True)
    return exe


def checksum_of(body: str) -> str:
    cs = 0
    for b in body.encode():
        cs ^= b
    return f"{cs:02X}"


def parse_cases() -> list[tuple[bytes, str]]:
    """(入力バイト列, 期待される出力行) の組。"""
    rng = random.Random(0)
    cases: list[tuple[bytes, str]] = []

    # 1) 正常系: Python エンコーダの出力をそのまま (境界値 + ランダム)
    commands = [(0, 0, 0.0, 0.0), (65535, 255, 100.0, 100.0), (65536 + 7, 128, 63.5, 78.0)]
    commands += [
        (rng.randrange(70000), rng.randrange(256), rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(500)
    ]
    for seq, pwm, pos, spd in commands:
        frame = encode_frame(seq, ActuatorCommand(pwm, pos, spd, "TEST"))
        cases.append((frame, f"OK {seq & 0xFFFF} {pwm} {round(pos * 10)} {round(spd * 10)}"))

    # 2) 異常系
    good = encode_frame(1, ActuatorCommand(100, 50.0, 50.0, "TEST"))  # $HAP,1,100,500,500*XX\n
    flipped = bytearray(good)
    flipped[8] ^= 0x01  # 本文 1 ビット反転
    cases += [
        (bytes(flipped), "ERR CHECKSUM"),
        (good.replace(b"\n", b"\r\n"), "OK 1 100 500 500"),  # CRLF 許容
        (b"garbage\x00\xff" + good, "OK 1 100 500 500"),  # 行頭ゴミから再同期
        (b"$HAP,1,100,5" + good, "OK 1 100 500 500"),  # 途中で切れた行の後に再同期
        (b"$HAP,1,100,500*%s\n" % checksum_of("HAP,1,100,500").encode(), "ERR FORMAT"),  # フィールド不足
        (b"$HAP,1,256,500,500*%s\n" % checksum_of("HAP,1,256,500,500").encode(), "ERR RANGE"),
        (b"$HAP,1,100,1001,500*%s\n" % checksum_of("HAP,1,100,1001,500").encode(), "ERR RANGE"),
        (b"$HAP,1,100,-5,500*%s\n" % checksum_of("HAP,1,100,-5,500").encode(), "ERR FORMAT"),
        (b"$HAP,99999,1,1,1*%s\n" % checksum_of("HAP,99999,1,1,1").encode(), "ERR FORMAT"),
        (b"$XYZ,1,100,500,500*%s\n" % checksum_of("XYZ,1,100,500,500").encode(), "ERR FORMAT"),
        (b"$HAP,1,100,500,500*ZZ\n", "ERR FORMAT"),
        (b"$HAP," + b"1" * 60 + b"\n", "ERR OVERFLOW"),
    ]
    return cases


def test_parse(exe: Path) -> bool:
    cases = parse_cases()
    stream = b"".join(data for data, _ in cases)
    out = subprocess.run([str(exe), "parse"], input=stream, capture_output=True, check=True).stdout
    got = out.decode().splitlines()
    want = [expected for _, expected in cases]
    if got == want:
        print(f"PASS parse: {len(cases)} frames (Python encoder <-> firmware parser)")
        return True
    for i, (g, w) in enumerate(zip(got, want)):
        if g != w:
            print(f"FAIL parse case {i}: input={cases[i][0]!r} got={g!r} want={w!r}")
            break
    else:
        print(f"FAIL parse: got {len(got)} lines, want {len(want)}")
    return False


def test_motion(exe: Path) -> bool:
    r = subprocess.run([str(exe), "motion"], capture_output=True, text=True)
    print(r.stdout, end="")
    return r.returncode == 0


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        exe = build(Path(tmp))
        ok = test_parse(exe) & test_motion(exe)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
