#ifndef HIPSTER_STREAM_HPP
#define HIPSTER_STREAM_HPP

#include "alies.hpp"

namespace hipster {

constexpr unsigned int HipStreamDefault = 0;
constexpr unsigned int HipStreamNonBlocking = 1;
constexpr unsigned int HipStreamDefaultFlags = HipStreamNonBlocking;

class HipStream {
public:
  explicit HipStream(unsigned int flags = HipStreamNonBlocking,
                     int priority = 0) {
    create(flags, priority);
  }

  ~HipStream() { destroy(); }

  HipStream(const HipStream &) = delete;
  HipStream &operator=(const HipStream &) = delete;

  HipStream(HipStream &&other) noexcept { moveFrom(std::move(other)); }

  HipStream &operator=(HipStream &&other) noexcept {
    if (this != &other) {
      destroy();
      moveFrom(std::move(other));
    }
    return *this;
  }

  void create(unsigned int flags = HipStreamNonBlocking, int priority = 0) {
    destroy();

    hipError_t err;
    if (priority != 0) {
      int priority_low, priority_high;
      hipDeviceGetStreamPriorityRange(&priority_low, &priority_high);

      int actual_priority = priority;
      if (priority < priority_high)
        actual_priority = priority_high;
      if (priority > priority_low)
        actual_priority = priority_low;

      err = hipStreamCreateWithPriority(&stream_, flags, actual_priority);
    } else {
      err = hipStreamCreateWithFlags(&stream_, flags);
    }

    if (err != hipSuccess) {
      stream_ = nullptr;
      valid_ = false;
      return;
    }

    valid_ = true;
  }

  void destroy() noexcept {
    if (valid_ && stream_) {
      hipStreamDestroy(stream_);
      stream_ = nullptr;
      valid_ = false;
    }
  }

  hipStream_t get() const noexcept { return stream_; }
  operator hipStream_t() const noexcept { return stream_; }

  void synchronize() const {
    if (valid_ && stream_) {
      hipStreamSynchronize(stream_);
    }
  }

  bool ready() const {
    if (valid_ && stream_) {
      return hipStreamQuery(stream_) == hipSuccess;
    }
    return false;
  }

  void waitEvent(hipEvent_t event, unsigned int flags = 0) {
    if (valid_ && stream_ && event) {
      hipStreamWaitEvent(stream_, event, flags);
    }
  }

  void addCallback(hipStreamCallback_t callback, void *userData = nullptr) {
    if (valid_ && stream_ && callback) {
      hipStreamAddCallback(stream_, callback, userData, 0);
    }
  }

  static void getPriorityRange(int *least_priority, int *greatest_priority) {
    hipDeviceGetStreamPriorityRange(least_priority, greatest_priority);
  }

  unsigned int getFlags() const {
    unsigned int flags = 0;
    if (valid_ && stream_) {
      hipStreamGetFlags(stream_, &flags);
    }
    return flags;
  }

  int getPriority() const {
    int priority = 0;
    if (valid_ && stream_) {
      hipStreamGetPriority(stream_, &priority);
    }
    return priority;
  }

  void reset() {
    synchronize();
    unsigned int flags = getFlags();
    int priority = getPriority();
    create(flags, priority);
  }

private:
  hipStream_t stream_ = nullptr;
  bool valid_ = false;

  void moveFrom(HipStream &&other) noexcept {
    stream_ = other.stream_;
    valid_ = other.valid_;
    other.stream_ = nullptr;
    other.valid_ = false;
  }
};

inline hipError_t HipStreamSynchronize(hipStream_t stream) {
  return hipStreamSynchronize(stream);
}

inline hipError_t HipStreamQuery(hipStream_t stream) {
  return hipStreamQuery(stream);
}

inline hipError_t HipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                                     unsigned int flags = 0) {
  return hipStreamWaitEvent(stream, event, flags);
}

} // namespace hipster

#endif // HIPSTER_STREAM_HPP