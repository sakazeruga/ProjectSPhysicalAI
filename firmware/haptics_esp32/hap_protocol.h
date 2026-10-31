// $HAP フレームのパーサ。Arduino に依存しない純粋な C++ (ホストでテスト可能)。
//
//   $HAP,<seq>,<pwm>,<pos_x10>,<speed_x10>*<XOR hex>\n
//   例) $HAP,42,180,635,780*69   → seq=42, PWM=180, 位置=63.5%, 速度=78.0%
//
// ホスト側エンコーダは serial_controller.py の encode_frame()。
#pragma once

#include <stddef.h>
#include <stdint.h>

namespace hap {

struct Frame {
  uint16_t seq;
  uint8_t pwm;         // 0..255
  uint16_t pos_x10;    // 0..1000 (0.1% 単位)
  uint16_t speed_x10;  // 0..1000 (0.1% 単位)
};

enum class Result : uint8_t {
  Ok,
  BadFormat,    // '$' / '*' / フィールド構成が不正
  BadChecksum,  // XOR 不一致
  OutOfRange,   // 値域外
  Overflow,     // 行が長すぎる (改行が欠落した等)
};

inline const char* resultName(Result r) {
  switch (r) {
    case Result::Ok: return "OK";
    case Result::BadFormat: return "FORMAT";
    case Result::BadChecksum: return "CHECKSUM";
    case Result::OutOfRange: return "RANGE";
    case Result::Overflow: return "OVERFLOW";
  }
  return "?";
}

namespace detail {

inline int hexValue(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  return -1;
}

// line[i] から 10 進数を読み、区切り文字 sep の直後まで i を進める
inline bool readField(const char* line, size_t end, size_t& i, char sep, uint32_t& out) {
  size_t start = i;
  uint32_t v = 0;
  while (i < end && line[i] >= '0' && line[i] <= '9') {
    v = v * 10 + (uint32_t)(line[i] - '0');
    if (v > 65535) return false;
    i++;
  }
  if (i == start || i > 5 + start) return false;
  if (sep != '\0') {
    if (i >= end || line[i] != sep) return false;
    i++;
  }
  out = v;
  return true;
}

}  // namespace detail

// 改行・CR を除いた 1 行を解析する
inline Result parseLine(const char* line, size_t len, Frame& out) {
  // 最短: "$HAP,0,0,0,0*XX" = 15 文字
  if (len < 15 || line[0] != '$' || line[len - 3] != '*') return Result::BadFormat;
  const size_t star = len - 3;

  uint8_t cs = 0;
  for (size_t i = 1; i < star; i++) cs ^= (uint8_t)line[i];
  int hi = detail::hexValue(line[star + 1]);
  int lo = detail::hexValue(line[star + 2]);
  if (hi < 0 || lo < 0) return Result::BadFormat;
  if (cs != (uint8_t)((hi << 4) | lo)) return Result::BadChecksum;

  if (line[1] != 'H' || line[2] != 'A' || line[3] != 'P' || line[4] != ',') return Result::BadFormat;
  size_t i = 5;
  uint32_t seq, pwm, pos, spd;
  if (!detail::readField(line, star, i, ',', seq) || !detail::readField(line, star, i, ',', pwm) ||
      !detail::readField(line, star, i, ',', pos) || !detail::readField(line, star, i, '\0', spd) ||
      i != star) {
    return Result::BadFormat;
  }
  if (pwm > 255 || pos > 1000 || spd > 1000) return Result::OutOfRange;

  out.seq = (uint16_t)seq;
  out.pwm = (uint8_t)pwm;
  out.pos_x10 = (uint16_t)pos;
  out.speed_x10 = (uint16_t)spd;
  return Result::Ok;
}

// バイトストリームを行に組み立てて解析する。
// '$' で常に新しい行を開始するので、ゴミバイトや途中で切れた行から自動で再同期する。
class FrameReader {
 public:
  static constexpr size_t kMaxLine = 40;

  // 1 バイト投入。行が完成したら true を返し、result / frame に結果を入れる。
  bool push(char c, Result& result, Frame& frame) {
    if (c == '$') {
      len_ = 0;
      inLine_ = true;
      overflow_ = false;
    }
    if (!inLine_) return false;  // 行頭 '$' 以前のバイトは無視
    if (c == '\r') return false;
    if (c == '\n') {
      inLine_ = false;
      result = overflow_ ? Result::Overflow : parseLine(buf_, len_, frame);
      return true;
    }
    if (len_ >= kMaxLine) {
      overflow_ = true;
      return false;
    }
    buf_[len_++] = c;
    return false;
  }

 private:
  char buf_[kMaxLine];
  size_t len_ = 0;
  bool inLine_ = false;
  bool overflow_ = false;
};

}  // namespace hap
