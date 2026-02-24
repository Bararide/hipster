#ifndef CORE_INFRASTUCTURE_NOTIFICATION_HPP
#define CORE_INFRASTUCTURE_NOTIFICATION_HPP

#include "concepts.hpp"
#include "utils/error.hpp"
#include "utils/success.hpp"
#include "event.hpp"
#include "result.hpp"
#include <functional>
#include <memory>
#include <type_traits>

namespace core {

template <Serializable NotificationType> class Notification {
public:
  using HandlerType = std::function<void(const NotificationType &)>;
  using EventType = NotificationType;

  explicit Notification(HandlerType handler) : handler_(std::move(handler)) {
    static_assert(Serializable<NotificationType>,
                  "NotificationType must be Serializable");
  }

  template <Serializable OtherType>
    requires IsConvertable<OtherType, NotificationType>
  Notification(const Notification<OtherType> &other)
      : handler_([other_handler =
                      other.handler_](const NotificationType &notification) {
          if constexpr (std::is_convertible_v<OtherType, NotificationType>) {
            other_handler(static_cast<OtherType>(notification));
          }
        }) {}

  void operator()(const NotificationType &notification) const {
    if (handler_) {
      handler_(notification);
    }
  }

  template <typename EventBus = Event>
  auto subscribe(EventBus &event_bus) const {
    return event_bus.template Subscribe<NotificationType>(handler_);
  }

  template <Serializable OtherType>
    requires IsConvertable<NotificationType, OtherType>
  Notification<OtherType> as() const {
    return Notification<OtherType>([this](const OtherType &other_notification) {
      if constexpr (std::is_convertible_v<NotificationType, OtherType>) {
        NotificationType notification =
            static_cast<NotificationType>(other_notification);
        (*this)(notification);
      }
    });
  }

  template <Serializable OtherType>
  Notification<std::pair<NotificationType, OtherType>>
  combine(const Notification<OtherType> &other) const {
    return Notification<std::pair<NotificationType, OtherType>>(
        [this, other](const std::pair<NotificationType, OtherType> &pair) {
          (*this)(pair.first);
          other(pair.second);
        });
  }

  template <typename Predicate>
  Notification<NotificationType> filter(Predicate predicate) const {
    return Notification<NotificationType>(
        [this, predicate =
                   std::move(predicate)](const NotificationType &notification) {
          if (predicate(notification)) {
            (*this)(notification);
          }
        });
  }

private:
  HandlerType handler_;
};

template <Serializable T, typename Handler>
auto make_notification(Handler &&handler) {
  return Notification<T>(std::forward<Handler>(handler));
}

template <Serializable T> auto success_notification() {
  return make_notification<utils::Success<T>>(
      [](const utils::Success<T> &success) {
        // Базовая реализация для успеха
      });
}

template <typename Err = std::exception> auto error_notification() {
  return make_notification<utils::Error>([](const utils::Error &error) {
    // Базовая реализация для ошибки
  });
}

template <Serializable T, typename Err = std::exception>
auto result_notification(Notification<utils::Success<T>> success_notif,
                         Notification<utils::Error> error_notif) {
  return make_notification<Result<T, Err>>(
      [success = std::move(success_notif),
       error = std::move(error_notif)](const Result<T, Err> &result) {
        result.match(
            [&success](const T &value) { success(utils::Success<T>{value}); },
            [&error](const Err &err) { error(utils::Error{err.what()}); });
      });
}

} // namespace core

#endif // CORE_NOTIFICATION_HPP