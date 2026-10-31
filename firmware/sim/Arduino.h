// ファームウェアを PC 上で動かすための Arduino-ESP32 API の最小実装 (シミュレータ用)。
//
// - Serial  : TCP サーバ。ホストは pyserial の "socket://127.0.0.1:<port>" で接続する。
//             切断されてもプロセスは動き続け (ウォッチドッグが働く)、次の接続を待つ。
// - LEDC    : ピンごとのデューティを記録する (sim_main.cpp がサーボ位置等に換算して出力)
// - GPIO    : STEP ピンの立ち上がりを数えてステッパ位置を再現する
#pragma once

#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <chrono>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
typedef SOCKET sim_socket_t;
#define SIM_INVALID_SOCKET INVALID_SOCKET
#define sim_close closesocket
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int sim_socket_t;
#define SIM_INVALID_SOCKET (-1)
#define sim_close close
#endif

#define ESP_ARDUINO_VERSION_MAJOR 3
#define HIGH 1
#define LOW 0
#define OUTPUT 1

// ------------------------------------------------------------------ 時刻
inline uint64_t sim_micros64() {
  using namespace std::chrono;
  static const auto t0 = steady_clock::now();
  return (uint64_t)duration_cast<microseconds>(steady_clock::now() - t0).count();
}
inline uint32_t micros() { return (uint32_t)sim_micros64(); }
inline uint32_t millis() { return (uint32_t)(sim_micros64() / 1000); }
inline void delayMicroseconds(uint32_t) {}

// ------------------------------------------------------------------ GPIO / LEDC
struct SimPins {
  uint32_t ledcDuty[64] = {};
  uint32_t ledcFreq[64] = {};
  uint8_t ledcBits[64] = {};
  uint8_t level[64] = {};
  int stepPin = -1, dirPin = -1;
  long stepCount = 0;
};
inline SimPins& simPins() {
  static SimPins p;
  return p;
}

inline void pinMode(uint8_t, uint8_t) {}
inline void digitalWrite(uint8_t pin, uint8_t v) {
  SimPins& p = simPins();
  if ((int)pin == p.stepPin && v && !p.level[pin]) p.stepCount += p.level[p.dirPin] ? 1 : -1;
  p.level[pin] = v;
}
inline bool ledcAttach(uint8_t pin, uint32_t freq, uint8_t bits) {
  simPins().ledcFreq[pin] = freq;
  simPins().ledcBits[pin] = bits;
  return true;
}
inline bool ledcWrite(uint8_t pin, uint32_t duty) {
  simPins().ledcDuty[pin] = duty;
  return true;
}

// ------------------------------------------------------------------ Serial (TCP)
class SimSerial {
 public:
  bool listen(uint16_t port) {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
#endif
    server_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_ == SIM_INVALID_SOCKET) return false;
    int one = 1;
    setsockopt(server_, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(server_, (sockaddr*)&addr, sizeof(addr)) != 0 || ::listen(server_, 1) != 0) return false;
    setNonBlocking(server_);
    return true;
  }

  bool connected() const { return client_ != SIM_INVALID_SOCKET; }

  void setRxBufferSize(size_t) {}
  void begin(unsigned long) {}

  int available() {
    poll();
    return (int)(len_ - pos_);
  }
  int read() {
    poll();
    return pos_ < len_ ? (uint8_t)buf_[pos_++] : -1;
  }
  size_t println(const char* s) {
    send(s, strlen(s));
    send("\n", 1);
    return strlen(s) + 1;
  }
  size_t printf(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
    char line[256];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(line, sizeof(line), fmt, ap);
    va_end(ap);
    if (n > 0) send(line, (size_t)n < sizeof(line) ? (size_t)n : sizeof(line) - 1);
    return n > 0 ? (size_t)n : 0;
  }

 private:
  static void setNonBlocking(sim_socket_t s) {
#ifdef _WIN32
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL) | O_NONBLOCK);
#endif
  }
  static bool wouldBlock() {
#ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
  }

  void drop() {
    sim_close(client_);
    client_ = SIM_INVALID_SOCKET;
    fprintf(stderr, "[sim] host disconnected\n");
  }

  void poll() {
    if (client_ == SIM_INVALID_SOCKET) {
      client_ = accept(server_, nullptr, nullptr);
      if (client_ == SIM_INVALID_SOCKET) return;
      setNonBlocking(client_);
      int one = 1;
      setsockopt(client_, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one));
      fprintf(stderr, "[sim] host connected\n");
    }
    if (pos_ == len_) pos_ = len_ = 0;
    if (len_ == sizeof(buf_)) return;
    int n = recv(client_, buf_ + len_, (int)(sizeof(buf_) - len_), 0);
    if (n > 0) {
      len_ += (size_t)n;
    } else if (n == 0 || !wouldBlock()) {
      drop();
    }
  }

  void send(const char* data, size_t n) {
    if (client_ == SIM_INVALID_SOCKET) return;  // 実機の USB-CDC 同様、未接続時の出力は捨てる
    if (::send(client_, data, (int)n, 0) < 0 && !wouldBlock()) drop();
  }

  sim_socket_t server_ = SIM_INVALID_SOCKET;
  sim_socket_t client_ = SIM_INVALID_SOCKET;
  char buf_[1024];
  size_t len_ = 0, pos_ = 0;
};

static SimSerial Serial;
