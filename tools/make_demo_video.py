"""
台本を通し実行し、提出用のデモ動画と評価値を出力する。

    python tools/make_demo_video.py                         # outputs/touch_care_demo.mp4
    python tools/make_demo_video.py --preview 30,45,92      # 指定秒の静止画だけ出力

物理はシミュレーション時刻で進めるので、描画や推論の速度によらず同じ動きになる。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from care_agent import EMOTIONS, MAX_INTENSITY, CarePolicy, EmotionEstimator  # noqa: E402
from sim.scenario import DEMO_DURATION, Metrics, ScenarioRunner, Tick, evaluate  # noqa: E402
from sim.touch_care import Phase  # noqa: E402

FPS = 30
WIDTH, HEIGHT = 1920, 1080
VIEW_W = 1320
PANEL_X = VIEW_W + 28
PANEL_W = WIDTH - PANEL_X - 28
INTRO, OUTRO = 5.0, 7.0

BG = (20, 24, 31)
CARD = (30, 36, 46)
TEXT = (236, 239, 244)
MUTED = (148, 158, 172)
USER = (240, 150, 70)
ROBOT = (86, 176, 240)
GOOD = (112, 200, 150)
WARN = (232, 92, 92)

EMOTION_JA = {
    "joy": "喜び", "sadness": "悲しみ", "anticipation": "期待", "surprise": "驚き",
    "anger": "怒り", "fear": "恐れ", "disgust": "嫌悪", "trust": "信頼",
}
MODE_NOTE = {
    "stop": "拒否の言葉は推論に頼らず語彙照合で即停止",
    "reflex": "接触力の急上昇を検知して自律的に離脱",
}
# (開始時刻, カメラ)
CAMERA_CUTS = ((0.0, "side"), (10.0, "close"), (38.0, "side"), (52.0, "close"),
               (88.0, "side"), (104.0, "close"), (112.0, "side"), (124.0, "close"))


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ("YuGothB.ttc", "meiryob.ttc") if bold else ("YuGothM.ttc", "meiryo.ttc")
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
    raise FileNotFoundError("日本語フォントが見つからない: " + ", ".join(names))


F_TITLE, F_H, F_BODY, F_SMALL = font(30, True), font(22, True), font(21), font(17)
F_SUB = font(38, True)
F_LARGE = font(30, True)


def camera_at(t: float) -> str:
    name = CAMERA_CUTS[0][1]
    for start, cam in CAMERA_CUTS:
        if t >= start:
            name = cam
    return name


class Panel:
    """右側の情報パネル。値は表示用に平滑化する。"""

    def __init__(self) -> None:
        self.emotions = {k: 0.0 for k in EMOTIONS}
        self.user = np.zeros(2)  # valence, arousal
        self.robot = np.zeros(2)
        self.force = deque(maxlen=FPS * 12)
        self.speed = deque(maxlen=FPS * 12)
        self._static = self._build_static()

    def _build_static(self) -> Image.Image:
        img = Image.new("RGB", (WIDTH - VIEW_W, HEIGHT), BG)
        d = ImageDraw.Draw(img)
        x0 = PANEL_X - VIEW_W
        d.text((x0, 26), "① AIの感情推定 (日本語BERT)", font=F_H, fill=TEXT)
        for i, name in enumerate(EMOTIONS):
            y = 66 + i * 30
            d.text((x0, y), EMOTION_JA[name], font=F_BODY, fill=MUTED)
            d.rounded_rectangle((x0 + 84, y + 7, x0 + PANEL_W, y + 21), 6, fill=CARD)
        d.text((x0, 322), "② 感情の円環モデル", font=F_H, fill=TEXT)
        d.rounded_rectangle((x0, 358, x0 + 226, 584), 10, fill=CARD)
        cx, cy = x0 + 113, 471
        d.line((x0 + 10, cy, x0 + 216, cy), fill=(70, 78, 92), width=1)
        d.line((cx, 368, cx, 574), fill=(70, 78, 92), width=1)
        d.text((x0 + 244, 372), "● 相手", font=F_BODY, fill=USER)
        d.text((x0 + 244, 404), "● ロボット", font=F_BODY, fill=ROBOT)
        d.text((x0 + 244, 448), "縦: 覚醒度", font=F_SMALL, fill=MUTED)
        d.text((x0 + 244, 472), "横: 感情価", font=F_SMALL, fill=MUTED)
        d.text((x0 + 244, 506), "相手の感情に同調せず", font=F_SMALL, fill=MUTED)
        d.text((x0 + 244, 530), "落ち着かせる側へ返す", font=F_SMALL, fill=MUTED)
        d.text((x0, 604), "③ ロボットの振る舞い", font=F_H, fill=TEXT)
        d.text((x0, 746), "④ 接触力 [N]", font=F_H, fill=TEXT)
        d.text((x0, 906), "⑤ なでる速度 [cm/s]", font=F_H, fill=TEXT)
        return img

    def update(self, tick: Tick, reflex_active: bool) -> None:
        d = tick.decision
        k = 0.12
        if d is not None:
            for name in EMOTIONS:
                target = max(0.0, min(d.emotions.get(name, 0.0), MAX_INTENSITY)) / MAX_INTENSITY
                self.emotions[name] += (target - self.emotions[name]) * k
            a, v = d.affect()
            self.user += (np.array([d.user_valence, d.user_arousal]) - self.user) * k
            self.robot += (np.array([v, a]) - self.robot) * k
        s = tick.sample
        self.force.append(s.force if s.phase is Phase.CONTACT or s.force > 0.05 else 0.0)
        self.speed.append(abs(s.speed) * 100 if s.phase is Phase.CONTACT else 0.0)

    def draw(self, tick: Tick, note: str | None) -> Image.Image:
        img = self._static.copy()
        d = ImageDraw.Draw(img)
        x0 = PANEL_X - VIEW_W
        for i, name in enumerate(EMOTIONS):
            y = 66 + i * 30
            w = (PANEL_W - 84) * self.emotions[name]
            if w > 3:
                d.rounded_rectangle((x0 + 84, y + 7, x0 + 84 + w, y + 21), 6, fill=USER)

        cx, cy = x0 + 113, 471
        for value, color in ((self.user, USER), (self.robot, ROBOT)):
            px = cx + value[0] * 98
            py = cy + 98 - value[1] * 196
            d.ellipse((px - 9, py - 9, px + 9, py + 9), fill=color)

        s = tick.sample
        dec = tick.decision
        if s.mode == "SAFE_STOP":
            label, color = "停止して離れる", WARN
        elif note == MODE_NOTE["reflex"]:
            label, color = "反射で離脱", WARN
        elif s.mode == "IDLE" or dec is None or dec.response is None:
            label, color = "待機", MUTED
        else:
            label, color = dec.response.name, ROBOT
        d.rounded_rectangle((x0, 640, x0 + PANEL_W, 726), 10, fill=CARD)
        d.text((x0 + 18, 648), label, font=F_LARGE, fill=color)
        d.text((x0 + 18, 692), f"mode={s.mode}  phase={s.phase}  PWM={s.vibration}", font=F_SMALL, fill=MUTED)

        contact = s.phase is Phase.CONTACT
        self._graph(d, (x0, 782, x0 + PANEL_W, 888), self.force, 6.0, [(1.5, GOOD), (5.0, WARN)], f"{s.force:.2f} N")
        self._graph(d, (x0, 942, x0 + PANEL_W, 1048), self.speed, 14.0, [(10.0, GOOD)],
                    f"{abs(s.speed) * 100:.1f} cm/s" if contact else "非接触")
        return img

    @staticmethod
    def _graph(d: ImageDraw.ImageDraw, box, values, top, lines, label) -> None:
        x0, y0, x1, y1 = box
        d.rounded_rectangle(box, 10, fill=CARD)
        h = y1 - y0 - 12
        base = y1 - 6
        for value, color in lines:
            y = base - h * value / top
            d.line((x0 + 8, y, x1 - 8, y), fill=color, width=1)
        if len(values) > 1:
            n = values.maxlen
            step = (x1 - x0 - 16) / n
            start = x1 - 8 - step * len(values)
            pts = [(start + i * step, base - h * min(v, top) / top) for i, v in enumerate(values)]
            d.line(pts, fill=TEXT, width=2)
        d.text((x1 - d.textlength(label, font=F_BODY), y0 - 34), label, font=F_BODY, fill=TEXT)


def wrap_center(d: ImageDraw.ImageDraw, text: str, y: int, f, fill, width: int = WIDTH) -> None:
    w = d.textlength(text, font=f)
    d.text(((width - w) / 2, y), text, font=f, fill=fill)


def title_card() -> np.ndarray:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)
    wrap_center(d, "気持ちに寄り添うタッチケアロボット", 390, font(66, True), TEXT)
    wrap_center(d, "相手の言葉から AI が感情を推定し、触れ方を変える", 500, font(36), MUTED)
    wrap_center(d, "MuJoCo シミュレーション / 等倍速", 590, font(26), MUTED)
    return np.asarray(img)


def outro_card(m: Metrics) -> np.ndarray:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(img)
    wrap_center(d, "この実行の計測結果", 250, font(50, True), TEXT)
    rows = (
        ("接触を保てた時間", f"{m.contact_time:.0f} 秒"),
        ("押し付け力 (目標 1.5 N)", f"{m.force_mean:.2f} ± {m.force_std:.2f} N"),
        ("なでる速度の最大値", f"{m.speed_peak * 100:.1f} cm/s"),
        ("10 cm/s 以下で動いた割合", f"{m.speed_ct_ratio * 100:.0f} %"),
        ("「やめて」→ 離れるまで", f"{m.stop_latency[0] * 1000:.0f} ms" if m.stop_latency else "-"),
    )
    f = font(36)
    for i, (k, v) in enumerate(rows):
        y = 370 + i * 76
        d.text((470, y), k, font=f, fill=MUTED)
        d.text((1120, y), v, font=font(36, True), fill=TEXT)
    return np.asarray(img)


class Compositor:
    def __init__(self, runner: ScenarioRunner) -> None:
        self.sim = runner.sim
        self.renderer = mujoco.Renderer(runner.sim.model, HEIGHT, VIEW_W)
        self.panel = Panel()

    def frame(self, tick: Tick, note: str | None) -> np.ndarray:
        self.sim.sync_visuals()
        self.renderer.update_scene(self.sim.data, camera=camera_at(tick.sample.t))
        view = Image.fromarray(self.renderer.render())
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        canvas.paste(view, (0, 0))
        canvas.paste(self.panel.draw(tick, note), (VIEW_W, 0))
        d = ImageDraw.Draw(canvas, "RGBA")

        text = tick.caption
        if text is None and tick.decision:
            text = f"「{tick.decision.text}」"
        if text:
            w = d.textlength(text, font=F_SUB)
            x = (VIEW_W - w) / 2
            d.rounded_rectangle((x - 28, 954, x + w + 28, 1032), 14, fill=(0, 0, 0, 170))
            d.text((x, 968), text, font=F_SUB, fill=(255, 255, 255, 255))
        if note:
            w = d.textlength(note, font=F_H)
            d.rounded_rectangle((24, 24, 24 + w + 36, 74), 10, fill=(190, 60, 60, 220))
            d.text((42, 34), note, font=F_H, fill=(255, 255, 255, 255))
        return np.asarray(canvas)


def run(out: Path, previews: list[float]) -> Metrics | None:
    estimator = EmotionEstimator()
    runner = ScenarioRunner(CarePolicy(estimator))
    comp = Compositor(runner)
    stops = [c.t for c, d in runner._decisions if d is not None and d.event == "stop"]

    writer = None
    if not previews:
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=1)
        title = title_card()
        for _ in range(int(INTRO * FPS)):
            writer.append_data(title)

    ticks: list[Tick] = []
    next_frame = 0.0
    prev_reflex = 0
    note_until = 0.0
    note = None
    pending = sorted(previews)
    for tick in runner.step_all():
        ticks.append(tick)
        t = tick.sample.t
        if runner.sim.reflex_count != prev_reflex:
            prev_reflex = runner.sim.reflex_count
            note, note_until = MODE_NOTE["reflex"], t + 4.0
        if tick.sample.mode == "SAFE_STOP":
            note, note_until = MODE_NOTE["stop"], t + 0.1
        if t > note_until:
            note = None
        if t + 1e-9 >= next_frame:
            next_frame += 1.0 / FPS
            comp.panel.update(tick, note is not None)
            if writer is not None:
                writer.append_data(comp.frame(tick, note))
            elif pending and t >= pending[0]:
                preview = out.with_name(f"preview_{pending.pop(0):05.1f}.png")
                preview.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(comp.frame(tick, note)).save(preview)
                print("wrote", preview)
                if not pending:
                    return None

    metrics = evaluate(ticks, runner.sim.cfg, stops, runner.sim.reflex_count)
    if writer is not None:
        outro = outro_card(metrics)
        for _ in range(int(OUTRO * FPS)):
            writer.append_data(outro)
        writer.close()
        out.with_suffix(".json").write_text(json.dumps(asdict(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
        print("wrote", out, f"({INTRO + DEMO_DURATION + OUTRO:.0f} s)")
    print(metrics)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "outputs" / "touch_care_demo.mp4")
    p.add_argument("--preview", default="", help="comma separated times [s]; write PNGs instead of the video")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)
    run(args.out, [float(x) for x in args.preview.split(",") if x])


if __name__ == "__main__":
    main()
