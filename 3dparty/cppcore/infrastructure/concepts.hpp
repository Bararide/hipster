#ifndef CORE_INFRASTRUCTURE_CONCEPTS_HPP
#define CORE_INFRASTRUCTURE_CONCEPTS_HPP

#include <chrono>
#include <concepts>
#include <functional>
#include <iostream>
#include <type_traits>

namespace core {
template <typename T>
concept Showable = requires(T &t, std::ostream &os) {
  { os << t } -> std::same_as<std::ostream &>;
};

template <typename T, typename U>
concept IsConvertable = std::is_convertible_v<U, T>;

template<typename From, typename To>
concept ConvertableTo = std::convertible_to<From, To>;

template <typename T>
concept Orderable = requires(const T &v, const T &u) {
  { v > u } -> std::same_as<bool>;
  { v >= u } -> std::same_as<bool>;
  { v < u } -> std::same_as<bool>;
  { v <= u } -> std::same_as<bool>;
  { v == u } -> std::same_as<bool>;
};

template <typename T>
concept OrderableAndShowable = Orderable<T> && Showable<T>;

template <typename T>
concept Stringify = requires(const T &obj) {
  requires std::same_as<T, std::string> || requires {
    { obj.toString() } -> std::same_as<std::string>;
  } || requires {
    { std::to_string(obj) } -> std::same_as<std::string>;
  };
};

template <typename T>
concept Serializable = requires(const T &t) {
  requires Stringify<T> || requires {
    { t.serialize() } -> std::same_as<std::string>;
  };
};

template <typename T>
concept IsArray = std::is_array_v<T>;

template <typename T>
concept IsPointer = std::is_pointer_v<T>;

template <typename T>
concept IsIterable = requires(T &t) {
  { t.begin() } -> std::same_as<decltype(t.begin())>;
  { t.end() } -> std::same_as<decltype(t.end())>;
};

template <typename T>
concept IsSizable = requires(T &t) {
  { t.size() } -> std::same_as<decltype(t.size())>;
} || requires(T &t) {
  { std::size(t) } -> std::same_as<decltype(std::size(t))>;
};

template <typename T>
concept IsIterableAndSizable = IsIterable<T> && IsSizable<T>;

template <typename F>
concept Functor = requires(F f) {
  requires requires {
    { f() } -> std::same_as<std::invoke_result_t<F>>;
  };
};

template <typename F, typename... Args>
concept FunctorWith = requires(F f, Args... args) {
  { f(args...) } -> std::same_as<std::invoke_result_t<F, Args...>>;
};

template <typename F, typename... Args>
concept Callable = requires(F f, Args... args) { f(args...); };

template <typename F, typename Ret, typename... Args>
concept CallableReturning =
    Callable<F, Args...> && requires(F f, Args... args) {
      { f(args...) } -> std::convertible_to<Ret>;
    };

template <typename F>
concept SimpleCallable = Callable<F>;

template <typename F, typename Ret>
concept SimpleCallableReturning = CallableReturning<F, Ret>;

template <typename T>
concept SimpleAwaitable = requires(T t) {
  { t.await() };
};

template <typename T, typename Ret>
concept AwaitableReturning = requires(T t) {
  { t.await() } -> std::convertible_to<Ret>;
};

template <typename T, typename Ret, typename... Args>
concept AwaitableWith = requires(T t, Args... args) {
  { t.await(args...) } -> std::convertible_to<Ret>;
};

template <typename T>
concept Awaitable = SimpleAwaitable<T>;

template <typename F, typename... Args>
concept Invocable = std::is_invocable_v<F, Args...>;

template <typename T>
concept IsMilliseconds =
    std::is_same_v<std::decay_t<T>(), std::chrono::milliseconds>;

template <typename T>
concept IsNanoseconds =
    std::is_same_v<std::decay_t<T>(), std::chrono::nanoseconds>;

template <typename T>
concept IsChronable = requires {
  typename T::rep;
  typename T::period;
  requires std::is_same_v<
      T, std::chrono::duration<typename T::rep, typename T::period>>;
} && requires(T duration) {
  { duration.count() } -> std::convertible_to<typename T::rep>;
};
} // namespace core

#endif // CONCEPTS_HPP