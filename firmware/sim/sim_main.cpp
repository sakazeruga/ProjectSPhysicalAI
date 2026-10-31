// haptics_esp32.ino をそのまま PC 上で動かすシミュレータ。
//
//   sim <tcp-port> [trace-interval-ms]
//
// ホストは pyserial の socket:// で接続する:
//   python main.py --serial-port socket://127.0.0.1:<tcp-port>
//
// stdout に物理出力のトレースを 1 行ずつ出す (テストや可視化用):
//   T <ms> vib=<duty> pos=<actuator %> tgt=<motion target %> fs=<failsafe> conn=<0/1>
// pos は実際のピン出力 (サーボのパルス幅 / ステップ数) から逆算した値。
#include <stdlib.h>

#include <thread>

#include "Arduino.h"

#include "../haptics_esp32/haptics_esp32.ino"

static float actuatorPositionPct() {
#if ACTUATOR_BACKEND == BACKEND_SERVO
  SimPins& p = simPins();
  uint32_t max = (1u << p.ledcBits[SERVO_PIN]) - 1;
  if (max == 0 || p.ledcFreq[SERVO_PIN] == 0) return NAN;
  float us = (float)p.ledcDuty[SERVO_PIN] / max * (1e6f / p.ledcFreq[SERVO_PIN]);
  return (us - SERVO_MIN_US) / (SERVO_MAX_US - SERVO_MIN_US) * 100.0f;
#else
  return (float)simPins().stepCount / STROKE_STEPS * 100.0f;
#endif
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <tcp-port> [trace-interval-ms]\n", argv[0]);
    return 2;
  }
  uint16_t port = (uint16_t)atoi(argv[1]);
  uint32_t traceMs = argc > 2 ? (uint32_t)atoi(argv[2]) : 20;
  setvbuf(stdout, nullptr, _IOLBF, 1024);

#if ACTUATOR_BACKEND == BACKEND_STEPPER
  simPins().stepPin = STEP_PIN;
  simPins().dirPin = DIR_PIN;
#endif
  if (!Serial.listen(port)) {
    fprintf(stderr, "[sim] cannot listen on 127.0.0.1:%u\n", port);
    return 1;
  }
  setup();
#if ACTUATOR_BACKEND == BACKEND_STEPPER
  simPins().stepCount = (long)lroundf(HOME_PCT * 0.01f * STROKE_STEPS);
#endif
  printf("READY port=%u backend=%d\n", port, ACTUATOR_BACKEND);
  fflush(stdout);

  uint32_t lastTrace = 0;
  for (;;) {
    loop();
    uint32_t now = millis();
    if (traceMs && now - lastTrace >= traceMs) {
      lastTrace = now;
      printf("T %lu vib=%u pos=%.2f tgt=%.2f fs=%d conn=%d\n", (unsigned long)now,
             simPins().ledcDuty[VIB_PIN], actuatorPositionPct(), motion.target(), failsafe ? 1 : 0,
             Serial.connected() ? 1 : 0);
      fflush(stdout);
    }
    std::this_thread::yield();
  }
}
