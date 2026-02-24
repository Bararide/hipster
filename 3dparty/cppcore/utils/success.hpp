#ifndef CORE_UTILS_SUCCESS_NOTIFICATION_HPP
#define CORE_UTILS_SUCCESS_NOTIFICATION_HPP

#include "infrastructure/concepts.hpp"
#include <spdlog/spdlog.h>
#include <string>

namespace core::utils {

template <typename T> struct Success {
public:
  explicit Success(T value) : value_(std::move(value)) {}

  template <typename OtherType>
    requires std::is_convertible_v<OtherType, T>
  Success(const Success<OtherType> &other) : value_(other.value()) {}

  const T &value() const { return value_; }
  T &value() { return value_; }

  std::string serialize() const {
    if constexpr (std::is_same_v<T, std::string>) {
      return value_;
    } else if constexpr (requires {
                           { value_.serialize() } -> std::same_as<std::string>;
                         }) {
      return value_.serialize();
    } else if constexpr (requires {
                           {
                             std::to_string(value_)
                           } -> std::same_as<std::string>;
                         }) {
      return std::to_string(value_);
    } else {
      return "Success";
    }
  }

  friend std::ostream &operator<<(std::ostream &os, const Success &success) {
    os << "Success: " << success.serialize();
    return os;
  }

private:
  T value_;
};

template <typename T>
auto success_notification =
    [](const Success<T> &success) { spdlog::info("{}", success.serialize()); };

} // namespace core::utils

#endif // CORE_UTILS_SUCCESS_NOTIFICATION_HPP