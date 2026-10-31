// 直動軸の目標追従 (速度・加速度制限つき)。Arduino 非依存 (ホストでテスト可能)。
//
// 2 つの使い方がある:
//   track()  : ホストから 100Hz で流れてくる目標列を追従する。連続する目標から目標速度を推定し
//              フィードフォワードするので、目標ごとに減速・停止を繰り返すカクつきが出ない。
//   moveTo() : 単発の位置決め (フェイルセーフのホーム退避等)。目標で静止する。
// どちらも速度は speedPct (% of maxSpeed) で、加速度は accel で制限する。
#pragma once

#include <math.h>

namespace hap {

struct MotionLimits {
  float maxSpeed;  // translation_speed=100% に相当する速度 [%/s]
  float accel;     // 加速度上限 [%/s^2]
  float minPos;    // ソフトリミット [%]
  float maxPos;
};

class Motion {
 public:
  // 目標速度の推定に使う目標更新間隔の範囲 [s]。ホストは 10ms 周期。
  static constexpr float kMinTargetInterval = 0.002f;  // これより短い = USB でまとめて届いた
  static constexpr float kMaxTargetInterval = 0.05f;   // これより長い = ストリームが途切れた
  // 最後の目標からこの時間を過ぎたらフィードフォワードを打ち切り、目標で静止する
  static constexpr float kFeedForwardHold = 0.03f;

  explicit Motion(const MotionLimits& lim, float initialPos = 0.0f)
      : lim_(lim), pos_(initialPos), target_(initialPos) {}

  void track(float target, float speedPct) {
    setLimits(target, speedPct);
    // 目標速度は「アンカー (前回の非バースト到着)」からの変化量で推定する。
    // まとめて届いたフレームはアンカーを動かさないので、バースト全体の時間幅で正しく割れる。
    if (sinceAnchor_ < kMinTargetInterval) {
      // バースト中: 推定値を据え置く
    } else if (sinceAnchor_ <= kMaxTargetInterval) {
      float v = clamp((target_ - anchorTarget_) / sinceAnchor_, -vmax_, vmax_);
      targetVel_ = 0.5f * (targetVel_ + v);  // 到着時刻の揺らぎを平滑化
      anchorTarget_ = target_;
      sinceAnchor_ = 0.0f;
    } else {
      targetVel_ = 0.0f;  // ストリーム再開直後
      anchorTarget_ = target_;
      sinceAnchor_ = 0.0f;
    }
    sinceTarget_ = 0.0f;
  }

  void moveTo(float target, float speedPct) {
    setLimits(target, speedPct);
    targetVel_ = 0.0f;
    sinceAnchor_ = kMaxTargetInterval * 2;  // 次の track() で速度推定しない
  }

  void update(float dt) {
    if (dt <= 0.0f) return;
    sinceTarget_ += dt;
    sinceAnchor_ += dt;

    // フィードフォワード中は目標を目標速度で外挿した参照点を追う
    bool feedForward = targetVel_ != 0.0f && sinceTarget_ < kFeedForwardHold;
    float ff = feedForward ? targetVel_ : 0.0f;
    float ref = clamp(target_ + ff * sinceTarget_, lim_.minPos, lim_.maxPos);

    float err = ref - pos_;
    float dir = err > 0.0f ? 1.0f : -1.0f;
    float vDesired = clamp(ff + dir * brakingSpeed(fabsf(err), dt), -vmax_, vmax_);

    float dvMax = lim_.accel * dt;
    vel_ += clamp(vDesired - vel_, -dvMax, dvMax);
    float next = pos_ + vel_ * dt;

    if (!feedForward && (ref - next) * err <= 0.0f) {
      // 静止目標に到達 (離散化による行き過ぎ・微振動を防ぐ)
      pos_ = ref;
      vel_ = 0.0f;
    } else {
      pos_ = clamp(next, lim_.minPos, lim_.maxPos);
    }
  }

  float position() const { return pos_; }
  float velocity() const { return vel_; }
  float target() const { return target_; }

 private:
  static float clamp(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }

  void setLimits(float target, float speedPct) {
    target_ = clamp(target, lim_.minPos, lim_.maxPos);
    vmax_ = clamp(speedPct, 0.0f, 100.0f) * 0.01f * lim_.maxSpeed;
  }

  // 1 周期あたり accel·dt ずつ減速して距離 d 以内で止まれる最大速度 (離散時間版の sqrt(2ad))
  float brakingSpeed(float d, float dt) const {
    float a = lim_.accel;
    return a * (sqrtf(0.25f * dt * dt + 2.0f * d / a) - 0.5f * dt);
  }

  MotionLimits lim_;
  float pos_;
  float vel_ = 0.0f;
  float target_;
  float vmax_ = 0.0f;
  float targetVel_ = 0.0f;
  float sinceTarget_ = 0.0f;  // 最新目標からの経過 (外挿用)
  float anchorTarget_ = 0.0f;
  float sinceAnchor_ = kMaxTargetInterval * 2;  // アンカーからの経過 (速度推定用)
};

}  // namespace hap
