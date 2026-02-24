#ifndef CORE_INFRASTRUCTURE_RESULT_HPP
#define CORE_INFRASTRUCTURE_RESULT_HPP

#include "concepts.hpp"
#include <concepts>
#include <exception>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace core {

template <typename T, typename Err = std::runtime_error> class Result;

template <typename T, typename Err> class Result<T &, Err> {
private:
  T *value_ptr_;
  std::optional<Err> error_;
  bool ok_flag;

public:
  Result(T &value) noexcept
      : value_ptr_(&value), error_(std::nullopt), ok_flag(true) {}

  template <typename E>
    requires std::is_constructible_v<Err, E &&>
  Result(E &&error) noexcept(std::is_nothrow_constructible_v<Err, E &&>)
      : value_ptr_(nullptr), error_(std::forward<E>(error)), ok_flag(false) {}

  bool is_ok() const { return ok_flag; }

  T &value() & {
    if (!ok_flag) {
      throw error_.value();
    }
    return *value_ptr_;
  }

  const T &value() const & {
    if (!ok_flag) {
      throw error_.value();
    }
    return *value_ptr_;
  }

  Err &error() & {
    if (ok_flag) {
      throw std::runtime_error("No error in Result");
    }
    return error_.value();
  }

  const Err &error() const & {
    if (ok_flag) {
      throw std::runtime_error("No error in Result");
    }
    return error_.value();
  }

  Err &&error() && {
    if (ok_flag) {
      throw std::runtime_error("No error in Result");
    }
    return std::move(error_.value());
  }

  T &unwrap() & { return value(); }
  const T &unwrap() const & { return value(); }

  T &expect(const char *message) {
    if (!ok_flag) {
      throw std::runtime_error(message);
    }
    return *value_ptr_;
  }

  template <typename F>
  auto map(F &&func) -> Result<std::invoke_result_t<F, T &>, Err> {
    if (!ok_flag) {
      return Result<std::invoke_result_t<F, T &>, Err>::Error(error_.value());
    }
    return Result<std::invoke_result_t<F, T &>, Err>::Ok(func(*value_ptr_));
  }

  template <typename F>
  auto map(F &&func) -> Result<std::invoke_result_t<F>, Err> {
    if (!ok_flag) {
      return Result<std::invoke_result_t<F>, Err>::Error(error_.value());
    }
    return Result<std::invoke_result_t<F>, Err>::Ok(func());
  }

  template <typename F>
  auto and_then(F &&func) -> std::invoke_result_t<F, T &> {
    if (!ok_flag) {
      using ResultType = std::invoke_result_t<F, T &>;
      return ResultType::Error(error_.value());
    }
    return func(*value_ptr_);
  }

  template <typename F> auto and_then(F &&func) -> std::invoke_result_t<F> {
    if (!ok_flag) {
      using ResultType = std::invoke_result_t<F>;
      return ResultType::Error(error_.value());
    }
    return func();
  }

  template <typename F1, typename F2>
  void handle(F1 &&on_success, F2 &&on_error) {
    if (ok_flag) {
      on_success(value());
    } else {
      on_error(error());
    }
  }

  template <typename OkHandler, typename ErrHandler>
  auto match(OkHandler &&ok_func, ErrHandler &&err_func)
      -> std::common_type_t<std::invoke_result_t<OkHandler, T &>,
                            std::invoke_result_t<ErrHandler, Err &>> {
    if (ok_flag) {
      return ok_func(*value_ptr_);
    }
    return err_func(error_.value());
  }

  template <typename OkHandler, typename ErrHandler>
  auto match(OkHandler &&ok_func, ErrHandler &&err_handler) const
      -> std::enable_if_t<
          std::is_void_v<std::invoke_result_t<OkHandler, T>> &&
              std::is_void_v<std::invoke_result_t<ErrHandler, Err>>,
          void> {
    if (ok_flag) {
      ok_func(value());
    } else {
      err_handler(error());
    }
  }

  static Result Ok(T &value) { return Result(value); }

  static Result Error(Err &&error) { return Result(std::forward<Err>(error)); }

  static Result Error(const Err &error) { return Result(error); }

  static Result Error(const char *message) { return Result(Err{message}); }

  static Result Error(const std::string &message) {
    return Result(Err{message.c_str()});
  }

  static Result Error(std::string &&message) {
    return Result(Err{message.c_str()});
  }
};

template <typename T, typename Err> class Result {
private:
  std::variant<T, Err> data_;
  bool ok_flag;

public:
  template <typename U>
    requires(!std::is_same_v<std::decay_t<U>, Err>) &&
                std::is_constructible_v<T, U &&>
  Result(U &&value) noexcept(std::is_nothrow_constructible_v<T, U &&>)
      : data_(std::in_place_index<0>, std::forward<U>(value)), ok_flag(true) {}

  template <typename E>
    requires std::is_constructible_v<Err, E &&>
  Result(E &&error) noexcept(std::is_nothrow_constructible_v<Err, E &&>)
      : data_(std::in_place_index<1>, std::forward<E>(error)), ok_flag(false) {}

  bool is_ok() const { return ok_flag; }

  T &value() & { return std::get<0>(data_); }
  const T &value() const & { return std::get<0>(data_); }
  T &&value() && { return std::get<0>(std::move(data_)); }

  Err &error() & { return std::get<1>(data_); }
  const Err &error() const & { return std::get<1>(data_); }
  Err &&error() && { return std::get<1>(std::move(data_)); }

  T &unwrap() & {
    if (!ok_flag) {
      throw error();
    }
    return value();
  }

  const T &unwrap() const & {
    if (!ok_flag) {
      throw error();
    }
    return value();
  }

  T &&unwrap() && {
    if (!ok_flag) {
      throw error();
    }
    return std::move(value());
  }

  T expect(const char *message) {
    if (!ok_flag) {
      throw std::runtime_error(message);
    }
    return std::move(value());
  }

  template <typename F>
  auto map(F &&func) -> Result<std::invoke_result_t<F, T>, Err> {
    if (!ok_flag) {
      return Result<std::invoke_result_t<F, T>, Err>::Error(error());
    }
    return Result<std::invoke_result_t<F, T>, Err>::Ok(func(value()));
  }

  template <typename F>
  auto map(F &&func) -> Result<std::invoke_result_t<F>, Err> {
    if (!ok_flag) {
      return Result<std::invoke_result_t<F>, Err>::Error(error());
    }
    return Result<std::invoke_result_t<F>, Err>::Ok(func());
  }

  template <typename F> auto and_then(F &&func) -> std::invoke_result_t<F, T> {
    if (!ok_flag) {
      using ResultType = std::invoke_result_t<F, T>;
      return ResultType::Error(error());
    }
    return func(value());
  }

  template <typename F> auto and_then(F &&func) -> std::invoke_result_t<F> {
    if (!ok_flag) {
      using ResultType = std::invoke_result_t<F>;
      return ResultType::Error(error());
    }
    return func();
  }

  template <typename F1, typename F2>
  void handle(F1 &&on_success, F2 &&on_error) {
    if (ok_flag) {
      on_success(value());
    } else {
      on_error(error());
    }
  }

  template <typename OkHandler, typename ErrHandler>
  auto match(OkHandler &&ok_func, ErrHandler &&err_func)
      -> std::common_type_t<std::invoke_result_t<OkHandler, T>,
                            std::invoke_result_t<ErrHandler, Err>> {
    if (ok_flag) {
      return ok_func(value());
    }
    return err_func(error());
  }

  template <typename OkHandler, typename ErrHandler>
  auto match(OkHandler &&ok_func, ErrHandler &&err_handler) const
      -> std::enable_if_t<
          std::is_void_v<std::invoke_result_t<OkHandler, T>> &&
              std::is_void_v<std::invoke_result_t<ErrHandler, Err>>,
          void> {
    if (ok_flag) {
      ok_func(value());
    } else {
      err_handler(error());
    }
  }

  static Result Ok(T &&value) { return Result(std::forward<T>(value)); }

  static Result Ok(const T &value) { return Result(value); }

  template <typename U> static Result Ok(U &&value) {
    return Result(std::forward<U>(value));
  }

  static Result Error(Err &&error) { return Result(std::forward<Err>(error)); }

  static Result Error(const Err &error) { return Result(error); }

  static Result Error(const char *message) { return Result(Err{message}); }
};

} // namespace core

#endif // RESULT_HPP