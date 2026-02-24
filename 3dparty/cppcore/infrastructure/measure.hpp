#ifndef CORE_INFRASTRUCTURE_MEASURE_HPP
#define CORE_INFRASTRUCTURE_MEASURE_HPP

#include "concepts.hpp"
#include <chrono>
#include <fmt/format.h>
#include <mutex>
#include <spdlog/spdlog.h>

namespace core::measure {

class Measure {
public:
  static void start() {
    std::lock_guard<std::mutex> lock(mutex_);
    start_ = std::chrono::high_resolution_clock::now();
    is_running_ = true;
  }

  static void end() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_running_) {
      end_ = std::chrono::high_resolution_clock::now();
      is_running_ = false;
    }
  }

  static void cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    is_running_ = false;
  }

  template <IsChronable T>
  static std::chrono::duration<typename T::rep, typename T::period> duration() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_running_) {
      return std::chrono::duration_cast<T>(end_ - start_);
    }
    return T::zero();
  }

  template <IsChronable T>
  static void result(const std::string &message = "duration: {}") {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_running_) {
      spdlog::warn("Measurement is still running. Call end() first.");
      return;
    }
    auto dur = std::chrono::duration_cast<T>(end_ - start_);
    spdlog::info(fmt::runtime(message), dur.count());
  }

  template <IsChronable T>
  static auto result_and_get(const std::string &message = "duration: {}") {
    std::lock_guard<std::mutex> lock(mutex_);
    if (is_running_) {
      spdlog::warn("Measurement is still running. Call end() first.");
      return T::zero();
    }
    auto dur = std::chrono::duration_cast<T>(end_ - start_);
    spdlog::info(fmt::runtime(message), dur.count());
    return dur;
  }

  static bool is_running() {
    std::lock_guard<std::mutex> lock(mutex_);
    return is_running_;
  }

  static void reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    start_ = TimePoint{};
    end_ = TimePoint{};
    is_running_ = false;
  }

private:
  using Clock = std::chrono::high_resolution_clock;
  using TimePoint = std::chrono::time_point<Clock>;

  static inline TimePoint start_;
  static inline TimePoint end_;
  static inline bool is_running_ = false;
  static inline std::mutex mutex_;
};

} // namespace core::measure

#endif // CORE_INFRASTRUCTURE_MEASURE_HPP