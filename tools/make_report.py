"""
提出用レポート (PDF) を生成する。評価値は make_demo_video.py が出力した JSON から読む。

    python tools/make_report.py --account <omnicampusアカウント名>

レポートに氏名・所属・メールアドレスを書かないこと。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import imageio
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

pdfmetrics.registerFont(TTFont("JP", "C:/Windows/Fonts/meiryo.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("JPB", "C:/Windows/Fonts/meiryob.ttc", subfontIndex=0))

INK = colors.HexColor("#1f2733")
MUTED = colors.HexColor("#5b6675")
RULE = colors.HexColor("#d7dce3")
HEAD = colors.HexColor("#eef1f5")

TITLE = ParagraphStyle("title", fontName="JPB", fontSize=20, leading=28, textColor=INK, alignment=TA_CENTER)
SUB = ParagraphStyle("sub", fontName="JP", fontSize=10.5, leading=16, textColor=MUTED, alignment=TA_CENTER)
H1 = ParagraphStyle("h1", fontName="JPB", fontSize=13, leading=19, textColor=INK, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("body", fontName="JP", fontSize=9.5, leading=15.5, textColor=INK, spaceAfter=4, wordWrap="CJK")
CELL = ParagraphStyle("cell", fontName="JP", fontSize=8.8, leading=13.2, textColor=INK, wordWrap="CJK")
CELLB = ParagraphStyle("cellb", fontName="JPB", fontSize=8.8, leading=13.2, textColor=INK, wordWrap="CJK")
CAP = ParagraphStyle("cap", fontName="JP", fontSize=8.2, leading=12, textColor=MUTED, alignment=TA_CENTER)
REF = ParagraphStyle("ref", fontName="JP", fontSize=8.2, leading=12.5, textColor=MUTED)


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def table(rows, widths, header=True) -> Table:
    data = [[p(escape(str(c)), CELLB if header and i == 0 else CELL) for c in row] for i, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="CENTER")
    style = [
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD))
    t.setStyle(TableStyle(style))
    return t


def extract_frames(video: Path, times: list[float]) -> list[Path]:
    """動画から図に使うコマを切り出す。"""
    paths = []
    for t in times:
        path = OUTPUTS / f"report_frame_{t:05.1f}.jpg"
        reader = imageio.get_reader(str(video))
        frame = reader.get_data(int(t * reader.get_meta_data()["fps"]))
        reader.close()
        PILImage.fromarray(frame).resize((1280, 720)).save(path, quality=88)
        paths.append(path)
    return paths


def build(out: Path, metrics: dict, frames: list[Path], tests: int) -> None:
    width = A4[0] - 36 * mm
    latency = metrics["stop_latency"][0] * 1000 if metrics["stop_latency"] else float("nan")
    s = [
        p("気持ちに寄り添うタッチケアロボット", TITLE),
        p("相手の言葉から AI が感情を推定し、触れ方を変える — Physical AI 応用編1 最終課題", SUB),
        Spacer(1, 6),
        p("1. 概要", H1),
        p("ロボットアームが相手の前腕にやわらかく触れ、相手の言葉から推定した感情に応じて触れ方を変えるシステムを"
          "シミュレータ上に実装した。入力は日本語の文で、学習済み BERT による感情推定、応答感情への変換、"
          "力制御までをひとつのパイプラインとして動かす。"),
        p("中心にある考えは<b>「相手の感情を真似しない」</b>ことである。怒っている相手に激しく触れれば逆効果になる。"
          "そこでロボットは、相手の感情の象限に対して、落ち着かせる側の振る舞いを返す。".replace("<b>", '<font name="JPB">').replace("</b>", "</font>")),
    ]
    img_w = width
    s += [Image(str(frames[0]), width=img_w, height=img_w * 9 / 16), p("図1 怒りの発話に対し、ロボットは低覚醒・快の側で応答している (右の円環図で相手と離れている)", CAP)]
    s += [
        p("2. AI の使い方", H1),
        p("学習は行わず、推論のみを使う。日本語 SNS 投稿の感情データセット WRIME で学習された BERT を用い、"
          "発話文から Plutchik の 8 感情の強度を推定する (CPU で 1 文あたり約 50〜170ms)。"
          "8 感情を Russell の円環モデル上の点に対応づけ、強度を重みとする重心から相手の覚醒度と感情価を求める。"),
        table(
            [
                ["相手の状態", "ロボットの振る舞い", "動き"],
                ["怒り・恐れ (不快 × 高覚醒)", "ゆっくり長くなだめる", "長いストロークを低速で往復"],
                ["悲しみ (不快 × 低覚醒)", "手を止めて寄り添う", "ほぼ静止し、心拍状の振動を送る"],
                ["喜び (快 × 高覚醒)", "軽やかに一緒に喜ぶ", "短いストロークを速めに往復"],
                ["安らぎ (快 × 低覚醒)", "ゆっくりなでる", "長いストローク"],
                ["中立", "そっと触れて見守る", "小さなストローク"],
            ],
            [width * 0.34, width * 0.3, width * 0.36],
        ),
        p("3. システム構成", H1),
        p("先に作成していた感情駆動ハプティクスの制御基盤 (入力層・状態変換層・出力層) をそのまま使い、"
          "入力層に感情推定、出力層に MuJoCo を追加した。出力層は抽象インターフェースの実装を差し替えるだけで"
          "切り替えられ、ESP32 実機向けのシリアル出力と同じ指令が流れる。"),
        table(
            [
                ["層", "内容", "実装"],
                ["入力", "発話 → 感情推定 → 応答感情。安全語彙の照合", "care_agent.py"],
                ["状態変換", "感情 → モード → 位置・速度・振動 (100Hz)。ヒステリシスと位相連続の発振器", "state_mapper.py"],
                ["力制御", "トルク指令のインピーダンス制御 + 力フィードバックのアドミタンス制御 (500Hz)", "sim/touch_care.py"],
                ["物理", "Franka Panda、バネ支持のケアパッド、可動する前腕", "sim/model/"],
            ],
            [width * 0.13, width * 0.62, width * 0.25],
        ),
        p("4. 工夫した点", H1),
        p('<font name="JPB">心地よいタッチの速度帯に収める。</font> 心地よい接触に反応する C 触覚線維は 1〜10cm/s のなでる刺激で'
          "最もよく発火することが知られている。この上限を、状態変換層の最大速度と力制御の参照速度の両方で守る設計にした。"),
        p('<font name="JPB">安全判断を確率モデルに任せない。</font> 「やめて」「痛い」等は推論を通さず語彙照合で停止する。'
          "さらに、接触力の急上昇による反射離脱と、指令途絶のウォッチドッグを、上位層に依存しない形で持たせた。"
          "テストでは、モデルが「喜び」と誤判定する状況でも停止することを確認している。"),
        p('<font name="JPB">接触の不安定さを構造で解決した。</font> 当初は腕を引いていない通常のストローク中にも反射離脱が作動し (通しで 12 回、正しいのは 1 回)、'
          "接触時間は 45 秒にとどまった。原因を切り分けたところ、滑らせる際のスティックスリップで接触が高頻度に"
          "途切れていた。パラメータの調整では解決せず、パッドをバネで支持する直列弾性構造に変更し、力をバネのたわみから"
          "測るようにしたところ、誤作動はなくなった。"),
        p('<font name="JPB">テストが見つけた問題を設計に戻した。</font> 階段状の指令で速度上限を 5 割超えることを'
          f"自動テストが検出し、参照軌道に加速度制限を加えた。テストは全 {tests} 件で、通し実行を含めて成功している。"),
        Image(str(frames[1]), width=img_w * 0.76, height=img_w * 0.76 * 9 / 16),
        p("図2 腕を引く動きを接触力の上昇として検知し、指令を待たずに離脱する場面", CAP),
        KeepTogether([
            p("5. 結果", H1),
            p("台本 136 秒を通し実行した結果を示す。数値はデモ動画と同じ実行の計測値である。"),
            table(
                [
                    ["項目", "結果", "目標・補足"],
                    ["接触を保った時間", f"{metrics['contact_time']:.0f} 秒", "停止・離脱・再接近の時間を除く"],
                    ["押し付け力", f"{metrics['force_mean']:.2f} ± {metrics['force_std']:.2f} N", "目標 1.5 N"],
                    ["なでる速度の最大値", f"{metrics['speed_peak'] * 100:.1f} cm/s", "上限 10 cm/s"],
                    ["速度が 10 cm/s 以下の割合", f"{metrics['speed_ct_ratio'] * 100:.1f} %", "接触中の全サンプル"],
                    ["拒否の発話 → 離脱", f"{latency:.0f} ms", "推論時間を含まない (下記)"],
                    ["反射離脱の回数", f"{metrics['reflex_count']} 回", "腕を引いた 1 回のみが正解"],
                    ["接触力のピーク", f"{metrics['force_peak_filtered']:.1f} N (平滑後) / {metrics['force_peak_raw']:.1f} N (生値)", "腕を引いた瞬間"],
                    ["生値が 5 N を超えた時間", f"{metrics['time_over_limit'] * 1000:.0f} ms", "通し実行の全体で通算"],
                ],
                [width * 0.34, width * 0.34, width * 0.32],
            ),
        ]),
        p("6. 限界と今後", H1),
        p("実機では動かしていない。肌の柔らかさ、パッドの摩擦 (係数 0.15 と実際の布より低い)、人物の動きはいずれも"
          "簡略化したモデルであり、数値は実環境の性能を保証しない。別途、前腕の 7 か所で腕を引く動作を試すと、"
          "接触力のピークは最大 5.1 N で、手首端では 5 N を 32ms 超えた。人に触れる用途には、より低慣性の機構と"
          "別途の安全検証が必要である。"),
        p("拒否から離脱までの時間は、シミュレーション上で発話を受けた時点からの値で、実際の推論時間 (約 50〜170ms) を含まない。"
          "感情モデルは SNS 投稿で学習されているため、「お茶を飲みました」のような平叙文にも喜びを出すなどの偏りがあり、"
          "語彙照合は「やめてほしくない」のような否定表現でも停止する (安全側の誤り)。"
          "今後は音声・表情の入力、相手の反応を見て触れ方を調整する閉ループ化、ESP32 実機との接続に取り組みたい。"),
        p("参考文献", H1),
        p("Löken et al., Coding of pleasant touch by unmyelinated afferents in humans, Nature Neuroscience, 2009. / "
          "Russell, A circumplex model of affect, Journal of Personality and Social Psychology, 1980. / "
          "Plutchik, A general psychoevolutionary theory of emotion, 1980. / "
          "Kajiwara et al., WRIME: A New Dataset for Emotional Intensity Estimation with Subjective and Objective Annotations, ACL 2021. / "
          "Ramos et al., Emotion Analysis of Writers and Readers of Japanese Tweets on Vaccinations, WASSA 2022. / "
          "Todorov et al., MuJoCo: A physics engine for model-based control, IROS 2012. / MuJoCo Menagerie (Franka Emika Panda, Apache-2.0).", REF),
    ]
    doc = SimpleDocTemplate(
        str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=15 * mm,
        title="気持ちに寄り添うタッチケアロボット", author="", subject="Physical AI 応用編1 最終課題",
    )
    doc.build(s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account", required=True, help="omnicampus account name (or account id)")
    parser.add_argument("--tests", type=int, required=True, help="number of passing tests")
    args = parser.parse_args()
    video = OUTPUTS / "touch_care_demo.mp4"
    metrics = json.loads(video.with_suffix(".json").read_text(encoding="utf-8"))
    frames = extract_frames(video, [50.0, 120.0])
    out = OUTPUTS / f"説明資料_PAI最終課題_{args.account}.pdf"
    build(out, metrics, frames, args.tests)
    print("wrote", out)


if __name__ == "__main__":
    main()
