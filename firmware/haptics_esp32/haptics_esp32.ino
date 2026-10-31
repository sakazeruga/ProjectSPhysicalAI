// ProjectS Physical AI — ハプティクスデバイス用 ESP32 ファームウェア (スタブ)
//
// ホスト (main.py / serial_controller.py) から $HAP フレームを受け取り、
//   - 振動モータ PWM
//   - 直動軸の位置・速度制御 (RC サーボ or ステッピングモータ)
// を駆動する。有効フレームが WATCHDOG_TIMEOUT_MS 途絶えると振動を止め、ホームへ低速退避する。
//
// デバイス → ホストの行:
//   ERR ...  : ホストで WARNING ログになる (チェックサム異常・ウォッチドッグ発動等)
//   その他   : ホストで DEBUG ログ (INFO / STAT)
//
// 対応: Arduino-ESP32 コア 2.x / 3.x
#include <Arduino.h>

#include "config.h"
#include "hap_protocol.h"
#include "motion.h"

#ifndef ESP_ARDUINO_VERSION_MAJOR
#define ESP_ARDUINO_VERSION_MAJOR 0
#endif

// ------------------------------------------------------------------ LEDC 互換層
static void pwmAttach(uint8_t pin, uint8_t channel, uint32_t freq, uint8_t bits) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)channel;
  ledcAttach(pin, freq, bits);
#else
  ledcSetup(channel, freq, bits);
  ledcAttachPin(pin, channel);
#endif
}

static void pwmWrite(uint8_t pin, uint8_t channel, uint32_t duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  (void)channel;
  ledcWrite(pin, duty);
#else
  (void)pin;
  ledcWrite(channel, duty);
#endif
}

// ------------------------------------------------------------------ 状態
static hap::FrameReader reader;
static hap::Motion motion({MAX_SPEED_PCT_S, ACCEL_PCT_S2, POS_MIN_PCT, POS_MAX_PCT}, HOME_PCT);

static uint32_t lastFrameMs = 0;
static bool failsafe = true;  // 最初の有効フレームまでは停止状態
static bool everReceived = false;
static uint8_t vibDuty = 0;
static uint32_t lastMotionUs = 0;

static struct {
  uint32_t rx = 0;
  uint32_t badChecksum = 0;
  uint32_t badFormat = 0;
  uint32_t lost = 0;  // seq の欠番数
  uint32_t watchdog = 0;
} stats;
static int32_t lastSeq = -1;
static uint32_t lastErrMs = 0;
static uint32_t lastStatusMs = 0;

// ------------------------------------------------------------------ 出力
static void setVibration(uint8_t duty) {
  if (duty == vibDuty) return;
  vibDuty = duty;
  pwmWrite(VIB_PIN, VIB_LEDC_CHANNEL, duty);
}

#if ACTUATOR_BACKEND == BACKEND_SERVO

static void actuatorBegin() { pwmAttach(SERVO_PIN, SERVO_LEDC_CHANNEL, SERVO_FREQ, SERVO_BITS); }

static void actuatorApply(float posPct) {
  const float periodUs = 1000000.0f / SERVO_FREQ;
  float us = SERVO_MIN_US + (SERVO_MAX_US - SERVO_MIN_US) * posPct * 0.01f;
  pwmWrite(SERVO_PIN, SERVO_LEDC_CHANNEL, (uint32_t)(us / periodUs * ((1 << SERVO_BITS) - 1)));
}

static void actuatorService() {}

#elif ACTUATOR_BACKEND == BACKEND_STEPPER

// スタブ実装: loop() 1 周につき最大 1 ステップ。速度は Motion 側で制限済み。
// 高速・低ジッタが必要になったら FastAccelStepper 等のハードウェアタイマ駆動へ置き換える。
static int32_t currentStep = 0;
static int32_t targetStep = 0;
static int8_t currentDir = 0;
// ステップ間隔の下限 = デバイス最大速度。loop() が一時的に遅れて目標から取り残されても、
// 追いつく際に最大速度を超えるバーストを出さない (脱調防止)
static const uint32_t kMinStepIntervalUs = (uint32_t)(1e6f / (MAX_SPEED_PCT_S * 0.01f * STROKE_STEPS));
static uint32_t lastStepUs = 0;

static void actuatorBegin() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(ENABLE_PIN, OUTPUT);
  digitalWrite(ENABLE_PIN, LOW);
  currentStep = targetStep = (int32_t)lroundf(HOME_PCT * 0.01f * STROKE_STEPS);
}

static void actuatorApply(float posPct) { targetStep = (int32_t)lroundf(posPct * 0.01f * STROKE_STEPS); }

static void actuatorService() {
  if (currentStep == targetStep) return;
  uint32_t now = micros();
  if (now - lastStepUs < kMinStepIntervalUs) return;
  lastStepUs = now;
  int8_t dir = targetStep > currentStep ? 1 : -1;
  if (dir != currentDir) {
    digitalWrite(DIR_PIN, dir > 0 ? HIGH : LOW);
    currentDir = dir;
    delayMicroseconds(1);  // DIR セットアップ時間
  }
  digitalWrite(STEP_PIN, HIGH);
  delayMicroseconds(STEP_PULSE_US);
  digitalWrite(STEP_PIN, LOW);
  currentStep += dir;
}

#else
#error "Unknown ACTUATOR_BACKEND"
#endif

// ------------------------------------------------------------------ 受信処理
static void reportError(const char* what) {
  uint32_t now = millis();
  if (now - lastErrMs < 200) return;  // ログ洪水を防ぐ (件数は STAT で確認)
  lastErrMs = now;
  Serial.printf("ERR %s (checksum=%lu format=%lu)\n", what, (unsigned long)stats.badChecksum,
                (unsigned long)stats.badFormat);
}

static void onFrame(const hap::Frame& f) {
  stats.rx++;
  if (lastSeq >= 0) {
    uint16_t gap = (uint16_t)(f.seq - (uint16_t)lastSeq - 1);
    if (gap < 1000) stats.lost += gap;  // それ以上はホスト再起動による seq リセットとみなす
  }
  lastSeq = f.seq;
  lastFrameMs = millis();

  if (failsafe) {
    failsafe = false;
    Serial.println(everReceived ? "INFO resumed" : "INFO first frame received");
    everReceived = true;
  }
  setVibration(f.pwm);
  motion.track(f.pos_x10 * 0.1f, f.speed_x10 * 0.1f);
}

static void pollSerial() {
  hap::Result result;
  hap::Frame frame;
  // 1 周で読む量を制限し、ステップ生成を長く止めない
  for (int n = 0; n < 64 && Serial.available() > 0; n++) {
    if (!reader.push((char)Serial.read(), result, frame)) continue;
    switch (result) {
      case hap::Result::Ok:
        onFrame(frame);
        break;
      case hap::Result::BadChecksum:
        stats.badChecksum++;
        reportError("CHECKSUM");
        break;
      default:
        stats.badFormat++;
        reportError(hap::resultName(result));
        break;
    }
  }
}

static void checkWatchdog() {
  if (failsafe || millis() - lastFrameMs < WATCHDOG_TIMEOUT_MS) return;
  failsafe = true;
  stats.watchdog++;
  setVibration(0);
  motion.moveTo(HOME_PCT, FAILSAFE_SPEED_PCT);
  Serial.printf("ERR WATCHDOG no frame for %d ms: vibration off, homing\n", WATCHDOG_TIMEOUT_MS);
}

static void reportStatus() {
  if (STATUS_INTERVAL_MS == 0 || millis() - lastStatusMs < STATUS_INTERVAL_MS) return;
  lastStatusMs = millis();
  Serial.printf("STAT rx=%lu lost=%lu cs=%lu fmt=%lu wdt=%lu pos=%.1f tgt=%.1f pwm=%u fs=%d\n",
                (unsigned long)stats.rx, (unsigned long)stats.lost, (unsigned long)stats.badChecksum,
                (unsigned long)stats.badFormat, (unsigned long)stats.watchdog, motion.position(),
                motion.target(), vibDuty, failsafe ? 1 : 0);
}

// ------------------------------------------------------------------ Arduino
void setup() {
  Serial.setRxBufferSize(512);
  Serial.begin(HAP_BAUDRATE);

  pwmAttach(VIB_PIN, VIB_LEDC_CHANNEL, VIB_PWM_FREQ, VIB_PWM_BITS);
  pwmWrite(VIB_PIN, VIB_LEDC_CHANNEL, 0);
  actuatorBegin();
  motion.moveTo(HOME_PCT, FAILSAFE_SPEED_PCT);
  actuatorApply(motion.position());

  lastMotionUs = micros();
  Serial.println("INFO HAP firmware ready");
}

void loop() {
  pollSerial();
  checkWatchdog();

  uint32_t nowUs = micros();
  if (nowUs - lastMotionUs >= MOTION_PERIOD_US) {
    float dt = (nowUs - lastMotionUs) * 1e-6f;
    lastMotionUs = nowUs;
    motion.update(dt > 0.01f ? 0.01f : dt);  // 長い停止後に大きく跳ばない
    actuatorApply(motion.position());
  }
  actuatorService();
  reportStatus();
}
