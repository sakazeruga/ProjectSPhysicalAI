# ProjectS Physical AI — Affective State → Haptics Bridge

AIエージェントの感情パラメータ (arousal / valence / event_trigger) を受信し、
ハプティクスデバイス (ESP32 等) の制御値へリアルタイム変換して送信する制御基盤。

```
InputSource ──▶ AffectInbox ──▶ StateMapper ──▶ ActuatorOutput
(WebSocket/Sim)  (最新値+イベント)   (状態機械)     (Serial/Log/差替可)
                       ▲
                Ticker (100Hz, 専用スレッド) / 新着入力で即時起床
```

| ファイル | 役割 |
|---|---|
| `interfaces.py` | 共通データ型・抽象インターフェース (`InputSource`, `ActuatorOutput`)・低遅延キュー |
| `input_handler.py` | WebSocket サーバ (検証・クランプ・部分更新)、疑似入力 |
| `state_mapper.py` | 感情 → PWM / ストローク位置 / 速度。モード状態機械とパターン発振器 |
| `serial_controller.py` | pyserial 非同期送信、自動検出、指数バックオフ再接続、keepalive、ドライラン出力 |
| `main.py` | 組み立て・制御ループ・テレメトリログ |
| `tools/ws_test_client.py` | 全モードを巡るシナリオ送信クライアント |

## セットアップと実行

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python main.py --input sim --output log
```

```bash
.venv/Scripts/python main.py --serial-port COM5
```

```bash
.venv/Scripts/python tools/ws_test_client.py
```

## テスト

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt
```

```bash
.venv/Scripts/python -m pytest tests
```

| ファイル | 内容 |
|---|---|
| `tests/test_input_handler.py` | メッセージ検証・クランプ・部分更新・不正値の拒否 |
| `tests/test_state_mapper.py` | モード判定・ヒステリシス・滞在時間・PWM・イベント・ラッチ停止・入力途絶・**出力位置の変化が申告速度とデバイス最大速度を超えないこと** |
| `tests/test_serial_controller.py` | フレーム形式・チェックサム・ループバック送信・ポート不在時の再試行・keepalive |
| `tests/test_firmware_sim.py` | 実ファームをシミュレータで動かすエンドツーエンド試験 (下記) |

### 実機なしでの結合動作 (ファームウェアシミュレータ)

`firmware/sim/` は `haptics_esp32.ino` をそのまま PC 上でビルドし、シリアルを TCP に置き換えて
動かすシミュレータ。ホストは pyserial の `socket://` でそのまま接続できる:

```bash
.venv/Scripts/python -m ziglang c++ -std=gnu++17 -O2 -I firmware/sim firmware/sim/sim_main.cpp -lws2_32 -o firmware/sim/build/sim.exe
```

```bash
firmware/sim/build/sim.exe 5555
```

```bash
.venv/Scripts/python main.py --input sim --serial-port socket://127.0.0.1:5555
```

シミュレータは stdout に `T <ms> vib=<PWM> pos=<実ピン出力から逆算した位置%> tgt= fs=<フェイルセーフ> conn=`
を出力する。`tests/test_firmware_sim.py` はこれを使い、追従・ウォッチドッグ (300ms で停止 + ホーム退避 +
自動復帰)・チェックサム異常の拒否・フレームの速度制限・ホストのパイプライン全体・デバイス再起動後の
ホスト自動再接続を、サーボ / ステッパ両バックエンドで検証する。

`--serial-port` は `auto` (CP210x / CH34x / Espressif / FTDI を自動検出)、`COM5`、
`socket://esp32.local:3333` (Wi-Fi 経由)、`loop://` (ループバック試験) を受け付ける。
`--log-level DEBUG` で全入力メッセージと送信フレームを表示する。

## 入力メッセージ

```json
{"arousal": 0.8, "valence": -0.4, "event_trigger": "surprise"}
```

- 省略したキーは同一接続の直前値を引き継ぐ (`{"event_trigger": "stop"}` だけ送れる)
- 範囲外はクランプ + 警告、不正値は `{"error": "..."}` を返して破棄

## モード (state_mapper.py)

| モード | 条件 | 位置パターン | 振動 |
|---|---|---|---|
| IDLE | arousal≈0 かつ中立 / 入力途絶 | 10% で静止 | 0 |
| NEUTRAL | \|valence\| < 0.25 | 50±10%, 0.2–0.6Hz | ×0.7 |
| CALM | 快 × 低覚醒 | 45±25%, ゆっくり | ×0.5 |
| EXCITED | 快 × 高覚醒 | 50±40%, 0.8–2.0Hz | ×1.0 |
| TENSE | 不快 × 高覚醒 | 75±8%, 2–4Hz の細かい往復 | ×1.0 |
| GLOOMY | 不快 × 低覚醒 | 15±5%, ごく遅い | ×0.3 |
| IMPULSE | `surprise` `startle` `impact` | 95% へ最大速度 (350ms) | 255 |
| PULSE | `heartbeat` `notify` | 継続 | 心拍状の二連パルス |
| SAFE_STOP | `stop` `emergency_stop` | ホームへ退避、`resume` までラッチ | 0 |

境界はヒステリシス (±0.07) と最小滞在時間 (250ms) でチャタリングを抑制。
パターン切替時は中心・振幅・周波数を一次遅れで追従させ、位相を連続に保つため位置が跳ばない。
周波数はデバイス最大速度 (`max_stroke_speed`) を超えないよう自動で制限される。
入力が 3 秒途絶えると感情値を 0 へ減衰させ IDLE に戻る。
しきい値・パターンはすべて `MapperConfig` で変更できる。

## シリアルプロトコル (ESP32 側で実装する)

```
$HAP,<seq>,<pwm>,<pos_x10>,<speed_x10>*<XOR>\n
例: $HAP,42,180,635,780*69   → PWM=180, 位置=63.5%, 速度=78.0%
```

- `XOR` は `$` と `*` の間の全バイトの XOR (16進2桁)。不一致フレームは破棄すること
- 115200bps、約 100 フレーム/秒。値が変わらない間も 100ms ごとに再送 (keepalive)
- ファームウェアは有効フレームが 300ms 途絶えると振動停止 + ホーム退避する (下記)
- `ERR` で始まる行を返すとホスト側で WARNING ログになる (その他の行は DEBUG)

## ESP32 ファームウェア (`firmware/haptics_esp32/`)

| ファイル | 内容 |
|---|---|
| `haptics_esp32.ino` | 受信・ウォッチドッグ・振動 PWM・アクチュエータ駆動 (Arduino-ESP32 2.x / 3.x 両対応) |
| `config.h` | ピン配置・速度/加速度・バックエンド選択 (`BACKEND_SERVO` / `BACKEND_STEPPER`) |
| `hap_protocol.h` | `$HAP` パーサ。チェックサム・値域検査、`$` による自動再同期 |
| `motion.h` | 速度・加速度制限つき追従。連続フレームから目標速度を推定してフィードフォワード |

- **振動**: LEDC 20kHz / 8bit で PWM 値をそのまま出力 (MOSFET 等のドライバ経由で接続)
- **直動**: RC サーボ信号 (1000–2000µs) か STEP/DIR。`MAX_SPEED_PCT_S` はホストの
  `MapperConfig.max_stroke_speed` (400 %/s) と一致させる
- **ウォッチドッグ**: 有効フレームが 300ms 途絶えると振動 0、20% 速度でホームへ退避し
  `ERR WATCHDOG` を送信。次の有効フレームで自動復帰
- **状態通知**: 1 秒ごとに `STAT rx= lost= cs= fmt= wdt= pos= tgt= pwm= fs=` (ホストでは DEBUG)
- ステッパ駆動は loop() でのソフトウェアパルス生成によるスタブ (ステップ間隔の下限で最大速度を保証)。高速域が必要なら
  FastAccelStepper 等のタイマ駆動に置き換えること。起動時の原点は 0% とみなす (ホーミング未実装)

ビルドは Arduino IDE か arduino-cli (esp32:esp32 3.3.12 と 2.0.17 で ESP32 / S3 / C3 × 両バックエンドの
ビルドを確認済み)。バックエンドは `config.h` を書き換えるか、ビルドフラグで切り替える:

```bash
arduino-cli compile --fqbn esp32:esp32:esp32 --build-property "compiler.cpp.extra_flags=-DACTUATOR_BACKEND=2" firmware/haptics_esp32
```

Arduino 非依存部 (パーサ・軌道生成) はホスト PC でテストできる。パーサには Python の
`encode_frame()` が生成した実フレームと破損フレームを流し込み、両者の解釈一致を確認する:

```bash
.venv/Scripts/python -m pip install ziglang
```

```bash
.venv/Scripts/python firmware/test/run_host_tests.py
```

## シミュレータへの差し替え

`ActuatorOutput` (start / submit / stop / is_connected) を実装したクラスを作り、
`main.build_output()` に分岐を足すだけで Gazebo / ROS 2 ブリッジ等へ出力先を切り替えられる。
`StateMapper` は I/O を持たない純粋な計算器なので、単体テストやオフライン再生にもそのまま使える。
