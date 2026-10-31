// ハードウェア構成。使うボード・アクチュエータに合わせて変更する。
#pragma once

// ---------------------------------------------------------------- 通信
#define HAP_BAUDRATE 115200
#define WATCHDOG_TIMEOUT_MS 300  // 有効フレームがこれだけ途絶えたらフェイルセーフ
#define STATUS_INTERVAL_MS 1000  // STAT 行の送信間隔 (0 で無効)

// ---------------------------------------------------------------- ピン配置
// チップごとの既定値。実際の配線に合わせて変更すること。
// 振動モータは MOSFET / モータドライバ経由で接続する (GPIO から直接駆動しないこと)
#if defined(CONFIG_IDF_TARGET_ESP32C3) || defined(CONFIG_IDF_TARGET_ESP32S3)
#define VIB_PIN 4
#define SERVO_PIN 5
#define STEP_PIN 6
#define DIR_PIN 7
#define ENABLE_PIN 10
#else  // ESP32 (無印)
#define VIB_PIN 25
#define SERVO_PIN 26
#define STEP_PIN 18
#define DIR_PIN 19
#define ENABLE_PIN 23
#endif

// ---------------------------------------------------------------- 振動モータ
#define VIB_PWM_FREQ 20000  // 20kHz: 可聴域外
#define VIB_PWM_BITS 8      // ホストの PWM 0..255 をそのまま使う
#define VIB_LEDC_CHANNEL 0  // Arduino-ESP32 2.x 用 (3.x では自動割当)

// ---------------------------------------------------------------- 直動軸
#define BACKEND_SERVO 1    // RC サーボ信号のリニアアクチュエータ (例: Actuonix L16-R)
#define BACKEND_STEPPER 2  // STEP/DIR 入力のステッピングモータドライバ (例: TMC2209)
#ifndef ACTUATOR_BACKEND   // ビルドフラグ -DACTUATOR_BACKEND=2 でも切り替え可
#define ACTUATOR_BACKEND BACKEND_SERVO
#endif

// 速度・加速度。MAX_SPEED_PCT_S はホストの MapperConfig.max_stroke_speed と一致させること
#define MAX_SPEED_PCT_S 400.0f
#define ACCEL_PCT_S2 4000.0f
#define POS_MIN_PCT 0.0f
#define POS_MAX_PCT 100.0f
#define HOME_PCT 0.0f            // フェイルセーフ時の退避位置
#define FAILSAFE_SPEED_PCT 20.0f // 退避速度 [% of MAX_SPEED_PCT_S]
#define MOTION_PERIOD_US 1000    // 軌道更新周期 (1kHz)

// BACKEND_SERVO
#define SERVO_FREQ 50
#define SERVO_BITS 14          // ESP32-S3/C3 の LEDC 上限に合わせる
#define SERVO_LEDC_CHANNEL 2   // 2.x 用。振動 (ch0) と別タイマになるよう ch2
#define SERVO_MIN_US 1000      // 位置 0% のパルス幅
#define SERVO_MAX_US 2000      // 位置 100% のパルス幅

// BACKEND_STEPPER
#define STROKE_STEPS 4000       // 0% → 100% のステップ数
#define STEP_PULSE_US 3
// 起動時、軸は 0% 位置にあるものとみなす。実機ではリミットスイッチ等でホーミングすること
