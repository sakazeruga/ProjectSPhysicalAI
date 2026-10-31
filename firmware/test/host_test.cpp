// ファームウェアの Arduino 非依存部 (hap_protocol.h / motion.h) をホスト PC で検証する。
//   host_test parse   : stdin のバイト列を FrameReader に流し、1 行ごとに結果を出力
//   host_test motion  : Motion の速度・加速度制限と追従性を検査 (失敗で終了コード 1)
// run_host_tests.py から呼ばれる。
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../haptics_esp32/hap_protocol.h"
#include "../haptics_esp32/motion.h"

static int runParse() {
  hap::FrameReader reader;
  hap::Result r;
  hap::Frame f;
  int c;
  while ((c = getchar()) != EOF) {
    if (!reader.push((char)c, r, f)) continue;
    if (r == hap::Result::Ok) {
      printf("OK %u %u %u %u\n", f.seq, f.pwm, f.pos_x10, f.speed_x10);
    } else {
      printf("ERR %s\n", hap::resultName(r));
    }
  }
  return 0;
}

static const float kPi = 3.14159265f;
static int failures = 0;

static void check(bool ok, const char* name, double value) {
  printf("%-4s %-44s %.3f\n", ok ? "PASS" : "FAIL", name, value);
  if (!ok) failures++;
}

static int runMotion() {
  const hap::MotionLimits lim{400.0f, 4000.0f, 0.0f, 100.0f};
  const float dt = 0.001f;

  {  // 0 → 100% を最大速度で: 速度・加速度上限を守り、行き過ぎず、所要時間が理論値付近
    hap::Motion m(lim, 0.0f);
    m.moveTo(100.0f, 100.0f);
    float maxV = 0, maxA = 0, maxPos = 0, prevV = 0, t = 0;
    while (t < 2.0f) {
      m.update(dt);
      t += dt;
      maxV = fmaxf(maxV, fabsf(m.velocity()));
      maxPos = fmaxf(maxPos, m.position());
      if (m.velocity() == 0.0f) break;  // 到着 (この 1 tick の停止は下で別に検査)
      maxA = fmaxf(maxA, fabsf(m.velocity() - prevV) / dt);
      prevV = m.velocity();
    }
    check(maxV <= 400.0f + 1e-3f, "step: peak speed <= 400 %/s", maxV);
    check(maxA <= 4000.0f + 1.0f, "step: peak accel <= 4000 %/s^2", maxA);
    check(prevV <= 2 * 4000.0f * dt + 1e-3f, "step: final stop from <= 2*a*dt (8 %/s)", prevV);
    check(maxPos <= 100.0f, "step: no overshoot", maxPos);
    check(t > 0.33f && t < 0.40f, "step: settle time ~0.35 s", t);
  }
  {  // 速度 10% → 40 %/s で頭打ち
    hap::Motion m(lim, 0.0f);
    m.moveTo(100.0f, 10.0f);
    float maxV = 0;
    for (int i = 0; i < 1000; i++) {
      m.update(dt);
      maxV = fmaxf(maxV, fabsf(m.velocity()));
    }
    check(fabsf(maxV - 40.0f) < 0.01f, "speed 10%: cruise at 40 %/s", maxV);
  }
  // ホストと同じく 1Hz・振幅40% のサイン目標を送り、追従誤差を見る。
  // burst=2 は USB 経由で 2 フレームずつまとめて届く (20ms ごとに 2 つ) 状況を模擬する。
  for (int burst = 1; burst <= 2; burst++) {
    hap::Motion m(lim, 50.0f);
    const float peakV = 2 * kPi * 1.0f * 40.0f;  // 251 %/s
    float worst = 0, maxV = 0;
    for (int k = 0; k < 300; k += burst) {  // 3 秒
      for (int b = 0; b < burst; b++) {
        float t = (k + b) * 0.01f;
        m.track(50.0f + 40.0f * sinf(2 * kPi * t), 100.0f * peakV / 400.0f);
      }
      for (int i = 0; i < 10 * burst; i++) {
        m.update(dt);
        maxV = fmaxf(maxV, fabsf(m.velocity()));
        // 最後のフレームの時刻 (k+burst-1)*10ms から i+1 ms 経過した時点の真の軌道と比較
        float now = (k + burst - 1) * 0.01f + (i + 1) * dt;
        if (k >= 100) worst = fmaxf(worst, fabsf(m.position() - (50.0f + 40.0f * sinf(2 * kPi * now))));
      }
    }
    // バースト時は先頭フレームが 10ms 古く、速度上限 = 波形のピーク速度で余裕がないため遅れを取り戻せない
    check(worst < (burst == 1 ? 1.0f : 3.0f),
          burst == 1 ? "sine 100Hz frames: error vs true path < 1 %" : "sine bursty frames: error vs true path < 3 %",
          worst);
    check(maxV <= peakV + 0.5f, burst == 1 ? "sine 100Hz frames: speed <= frame limit" : "sine bursty frames: speed <= frame limit", maxV);

    // ストリームが途切れたら暴走せず、最後の目標で静止する
    float last = m.target();
    for (int i = 0; i < 500; i++) m.update(dt);
    if (burst == 1) {
      check(m.velocity() == 0.0f && fabsf(m.position() - last) < 1e-3f, "stream stop: settles on last target",
            fabsf(m.position() - last));
    }
  }
  {  // フェイルセーフ相当: ストリーム中に moveTo で退避しても目標速度が残らない
    hap::Motion m(lim, 50.0f);
    for (int k = 0; k < 50; k++) {
      m.track(50.0f + 0.5f * k, 50.0f);
      for (int i = 0; i < 10; i++) m.update(dt);
    }
    m.moveTo(0.0f, 20.0f);
    float maxV = 0;
    for (int i = 0; i < 3000; i++) {
      m.update(dt);
      if (i > 200) maxV = fmaxf(maxV, fabsf(m.velocity()));
    }
    check(m.position() == 0.0f && maxV <= 80.0f + 1e-3f, "moveTo home after stream: 80 %/s, reaches 0", m.position());
  }
  {  // ソフトリミット
    hap::MotionLimits narrow = lim;
    narrow.minPos = 2.0f;
    narrow.maxPos = 98.0f;
    hap::Motion m(narrow, 50.0f);
    m.moveTo(150.0f, 100.0f);
    for (int i = 0; i < 1000; i++) m.update(dt);
    check(m.position() == 98.0f, "soft limit clamps target", m.position());
  }
  return failures ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc > 1 && strcmp(argv[1], "parse") == 0) return runParse();
  if (argc > 1 && strcmp(argv[1], "motion") == 0) return runMotion();
  fprintf(stderr, "usage: host_test parse|motion\n");
  return 2;
}
