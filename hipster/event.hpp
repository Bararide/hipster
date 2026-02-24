#ifndef HIPSTER_EVENT_HPP
#define HIPSTER_EVENT_HPP

#include "alies.hpp"

namespace hipster {

constexpr unsigned int HipEventDefault = 0;
constexpr unsigned int HipEventBlockingSync = 1;
constexpr unsigned int HipEventDisableTiming = 2;
constexpr unsigned int HipEventInterprocess = 4;
constexpr unsigned int HipEventReleaseToDevice = 8;

class HipEvent {
public:
  explicit HipEvent(unsigned int flags = HipEventDisableTiming) {
    create(flags);
  }

  ~HipEvent() { destroy(); }

  HipEvent(const HipEvent &) = delete;
  HipEvent &operator=(const HipEvent &) = delete;

  HipEvent(HipEvent &&other) noexcept { moveFrom(std::move(other)); }

  HipEvent &operator=(HipEvent &&other) noexcept {
    if (this != &other) {
      destroy();
      moveFrom(std::move(other));
    }
    return *this;
  }

  void create(unsigned int flags) {
    destroy();
    hipError_t err = hipEventCreateWithFlags(&event_, flags);
    if (err != hipSuccess) {
      event_ = nullptr;
      valid_ = false;
      return;
    }
    valid_ = true;
  }

  void destroy() {
    if (valid_ && event_) {
      hipEventDestroy(event_);
      event_ = nullptr;
      valid_ = false;
    }
  }

  hipEvent_t get() const { return event_; }
  operator hipEvent_t() const { return event_; }

  void record(hipStream_t stream = 0) {
    if (valid_ && event_) {
      hipEventRecord(event_, stream);
    }
  }

  void synchronize() const {
    if (valid_ && event_) {
      hipEventSynchronize(event_);
    }
  }

  bool ready() const {
    if (valid_ && event_) {
      return hipEventQuery(event_) == hipSuccess;
    }
    return false;
  }

  bool wait(int timeout_us = 100) const {
    if (!valid_ || !event_)
      return false;

    auto start = Clock::high_resolution_clock::now();
    while (hipEventQuery(event_) == hipErrorNotReady) {
      auto now = Clock::high_resolution_clock::now();
      auto elapsed =
          Clock::duration_cast<Clock::microseconds>(now - start).count();

      if (elapsed > timeout_us) {
        return false;
      }
      std::this_thread::yield();
    }
    return true;
  }

  static float elapsedTime(const HipEvent &start, const HipEvent &end) {
    float ms = 0.0f;
    if (start.valid_ && start.event_ && end.valid_ && end.event_) {
      hipEventElapsedTime(&ms, start.event_, end.event_);
    }
    return ms;
  }

  void reset() {
    unsigned int flags = HipEventDisableTiming;
    create(flags);
  }

private:
  hipEvent_t event_ = nullptr;
  bool valid_ = false;

  void moveFrom(HipEvent &&other) noexcept {
    event_ = other.event_;
    valid_ = other.valid_;
    other.event_ = nullptr;
    other.valid_ = false;
  }
};

inline hipError_t HipEventRecord(hipEvent_t event, hipStream_t stream = 0) {
  return hipEventRecord(event, stream);
}

inline hipError_t HipEventSynchronize(hipEvent_t event) {
  return hipEventSynchronize(event);
}

inline hipError_t HipEventElapsedTime(float *ms, hipEvent_t start,
                                      hipEvent_t end) {
  return hipEventElapsedTime(ms, start, end);
}

} // namespace hipster

#endif // HIPSTER_EVENT_HPP