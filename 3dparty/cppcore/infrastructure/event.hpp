#ifndef EVENT_HPP
#define EVENT_HPP

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core {
class Event {
public:
  using eventID = size_t;
  using handlerID = size_t;

  template <typename EventType> struct EventT {
    using CallbackType = std::function<void(const EventType &)>;
    using MutableCallbackType = std::function<void(EventType &)>;
  };

  template <typename EventType>
  handlerID Subscribe(std::function<void(const EventType &)> handler) {
    return SubscribeSingle<EventType>(std::move(handler));
  }

  template <typename EventType>
  handlerID Subscribe(std::function<void(EventType &)> handler) {
    return SubscribeSingleMutable<EventType>(std::move(handler));
  }

  template <typename... EventTypes>
  handlerID Subscribe(std::function<void(const EventTypes &)>... handlers) {
    static_assert(sizeof...(EventTypes) > 0,
                  "At least one handler must be provided");
    static_assert(
        (... && std::is_invocable_v<decltype(handlers), const EventTypes &>),
        "Each handler must be callable with the corresponding event type.");

    handlerID id = nextHandlerID++;
    std::unique_lock lock(handler_mutex);
    (SubscribeSingle<EventTypes>(id, std::move(handlers)), ...);
    return id;
  }

  template <typename... EventTypes>
  handlerID Subscribe(std::function<void(EventTypes &)>... handlers) {
    static_assert(sizeof...(EventTypes) > 0,
                  "At least one handler must be provided");
    static_assert(
        (... && std::is_invocable_v<decltype(handlers), EventTypes &>),
        "Each handler must be callable with the corresponding event type.");

    handlerID id = nextHandlerID++;
    std::unique_lock lock(handler_mutex);
    (SubscribeSingleMutable<EventTypes>(id, std::move(handlers)), ...);
    return id;
  }

  template <typename InputType, typename OutputType>
  handlerID
  SubscribeChain(std::function<OutputType(const InputType &)> handler) {
    handlerID id = nextHandlerID++;
    std::unique_lock lock(handler_mutex);

    auto event_id = GetEventID<InputType>();
    auto &handler_list = chain_event_handlers[event_id];

    if (!handler_list) {
      handler_list =
          std::make_shared<ChainHandlerList<InputType, OutputType>>();
    }

    auto &chain_list =
        static_cast<ChainHandlerList<InputType, OutputType> &>(*handler_list);
    chain_list.handlers.emplace_back(id, std::move(handler));

    return id;
  }

  template <typename... EventTypes> void Unsubscribe(handlerID id) {
    std::unique_lock lock(handler_mutex);
    (UnsubscribeSingle<EventTypes>(id), ...);
  }

  template <typename EventType> void Notify(const EventType &event) {
    std::shared_lock lock(handler_mutex);
    auto it = event_handlers.find(GetEventID<EventType>());

    if (it != event_handlers.end()) {
      auto &handler_list = static_cast<HandlerList<EventType> &>(*it->second);

      for (const auto &[current_id, handler] : handler_list.const_handlers) {
        handler(event);
      }

      for (const auto &[current_id, handler] : handler_list.mutable_handlers) {
        handler(const_cast<EventType &>(event));
      }
    }
  }

  template <typename EventType> void Notify(EventType &event) {
    std::shared_lock lock(handler_mutex);
    auto it = event_handlers.find(GetEventID<EventType>());

    if (it != event_handlers.end()) {
      auto &handler_list = static_cast<HandlerList<EventType> &>(*it->second);

      for (const auto &[current_id, handler] : handler_list.const_handlers) {
        handler(event);
      }

      for (const auto &[current_id, handler] : handler_list.mutable_handlers) {
        handler(event);
      }
    }
  }

  template <typename InputType>
  std::optional<InputType> NotifyChain(const InputType &event) {
    std::shared_lock lock(handler_mutex);
    auto it = chain_event_handlers.find(GetEventID<InputType>());

    if (it != chain_event_handlers.end()) {
      auto &chain_list =
          static_cast<ChainHandlerList<InputType, InputType> &>(*it->second);

      std::optional<InputType> result = event;
      for (const auto &[id, handler] : chain_list.handlers) {
        if (result) {
          result = handler(*result);
        }
      }
      return result;
    }

    return std::nullopt;
  }

  template <typename EventType>
  std::future<void> NotifyAsync(const EventType &event) {
    return std::async(std::launch::async, [this, event]() { Notify(event); });
  }

  template <typename EventType>
  std::future<void> NotifyAsync(EventType &event) {
    return std::async(std::launch::async, [this, &event]() { Notify(event); });
  }

private:
  struct BaseHandler {
    virtual ~BaseHandler() = default;
  };

  template <typename EventType> struct HandlerList : BaseHandler {
    using CallbackType = typename EventT<EventType>::CallbackType;
    using MutableCallbackType = typename EventT<EventType>::MutableCallbackType;
    std::vector<std::pair<handlerID, CallbackType>> const_handlers;
    std::vector<std::pair<handlerID, MutableCallbackType>> mutable_handlers;
  };

  template <typename InputType, typename OutputType>
  struct ChainHandlerList : BaseHandler {
    using HandlerType = std::function<OutputType(const InputType &)>;
    std::vector<std::pair<handlerID, HandlerType>> handlers;
  };

  std::unordered_map<eventID, std::shared_ptr<BaseHandler>> event_handlers;
  std::unordered_map<eventID, std::shared_ptr<BaseHandler>>
      chain_event_handlers;

  std::shared_mutex handler_mutex;
  std::atomic<handlerID> nextHandlerID = 0;

  template <typename EventType> eventID GetEventID() {
    static eventID id = typeid(EventType).hash_code();
    return id;
  }

  template <typename EventType>
  handlerID SubscribeSingle(std::function<void(const EventType &)> handler) {
    handlerID id = nextHandlerID++;
    std::unique_lock lock(handler_mutex);

    auto event_id = GetEventID<EventType>();
    auto &base_handler = event_handlers[event_id];

    if (!base_handler) {
      base_handler = std::make_shared<HandlerList<EventType>>();
    }

    auto &handler_list = static_cast<HandlerList<EventType> &>(*base_handler);
    handler_list.const_handlers.emplace_back(id, std::move(handler));

    return id;
  }

  template <typename EventType>
  handlerID SubscribeSingleMutable(std::function<void(EventType &)> handler) {
    handlerID id = nextHandlerID++;
    std::unique_lock lock(handler_mutex);

    auto event_id = GetEventID<EventType>();
    auto &base_handler = event_handlers[event_id];

    if (!base_handler) {
      base_handler = std::make_shared<HandlerList<EventType>>();
    }

    auto &handler_list = static_cast<HandlerList<EventType> &>(*base_handler);
    handler_list.mutable_handlers.emplace_back(id, std::move(handler));

    return id;
  }

  template <typename EventType>
  void SubscribeSingle(handlerID id,
                       std::function<void(const EventType &)> handler) {
    auto event_id = GetEventID<EventType>();
    auto &base_handler = event_handlers[event_id];

    if (!base_handler) {
      base_handler = std::make_shared<HandlerList<EventType>>();
    }

    auto &handler_list = static_cast<HandlerList<EventType> &>(*base_handler);
    handler_list.const_handlers.emplace_back(id, std::move(handler));
  }

  template <typename EventType>
  void SubscribeSingleMutable(handlerID id,
                              std::function<void(EventType &)> handler) {
    auto event_id = GetEventID<EventType>();
    auto &base_handler = event_handlers[event_id];

    if (!base_handler) {
      base_handler = std::make_shared<HandlerList<EventType>>();
    }

    auto &handler_list = static_cast<HandlerList<EventType> &>(*base_handler);
    handler_list.mutable_handlers.emplace_back(id, std::move(handler));
  }

  template <typename EventType> void UnsubscribeSingle(handlerID id) {
    auto event_id = GetEventID<EventType>();
    auto it = event_handlers.find(event_id);

    if (it != event_handlers.end()) {
      auto &handler_list = static_cast<HandlerList<EventType> &>(*it->second);

      auto &const_handlers = handler_list.const_handlers;
      const_handlers.erase(
          std::remove_if(const_handlers.begin(), const_handlers.end(),
                         [id](const auto &pair) { return pair.first == id; }),
          const_handlers.end());

      auto &mutable_handlers = handler_list.mutable_handlers;
      mutable_handlers.erase(
          std::remove_if(mutable_handlers.begin(), mutable_handlers.end(),
                         [id](const auto &pair) { return pair.first == id; }),
          mutable_handlers.end());
    }
  }
};

} // namespace core

#endif // EVENT_HPP