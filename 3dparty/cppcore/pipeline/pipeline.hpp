#ifndef OWL_VECTORFS_PIPELINE_HPP
#define OWL_VECTORFS_PIPELINE_HPP

#include "event.hpp"
#include "pipeline_handler.hpp"
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <spdlog/spdlog.h>
#include <sstream>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace core::pipeline {

class Pipeline {
public:
  Pipeline() = default;

  Pipeline(Pipeline &&other) noexcept
      : handlers_(std::move(other.handlers_)),
        chain_subscriptions_(std::move(other.chain_subscriptions_)) {}

  Pipeline &operator=(Pipeline &&other) noexcept {
    if (this != &other) {
      handlers_ = std::move(other.handlers_);
      chain_subscriptions_ = std::move(other.chain_subscriptions_);
    }
    return *this;
  }

  Pipeline(const Pipeline &other) = delete;
  Pipeline &operator=(const Pipeline &other) = delete;

  ~Pipeline() { cleanup_all_chains(); }

  template <typename Handler> void add_handler(Handler &handler) {
    handlers_.push_back(std::ref(handler));
  }

  template <typename T> core::Result<T> process(T &data) {
    spdlog::info("Starting pipeline processing with {} handlers",
                 handlers_.size());

    if (handlers_.empty()) {
      return core::Result<T>::Ok(data);
    }

    cleanup_all_chains();

    if (!setup_handler_chain<T>()) {
      return core::Result<T>::Error("Failed to setup handler chain for type");
    }

    auto promise = std::make_shared<std::promise<core::Result<T>>>();
    auto future = promise->get_future();

    auto &last_handler = handlers_.back().get();
    auto final_handler_id = last_handler.get_event_bus().template Subscribe<T>(
        [promise](const T &final_data) {
          promise->set_value(core::Result<T>::Ok(final_data));
        });

    spdlog::info("Starting processing through event chain");

    auto &first_handler = handlers_.front().get();
    first_handler.await();

    auto first_result = dynamic_call_handle<T>(&first_handler, data);
    if (first_result.is_ok()) {
      first_handler.get_event_bus().Notify(first_result.value());
    } else {
      promise->set_value(first_result);
    }

    auto result = future.get();

    last_handler.get_event_bus().Unsubscribe(final_handler_id);
    cleanup_all_chains();

    return result;
  }

  template <typename T>
  std::future<core::Result<T>> process_async(const T &data) {
    return std::async(std::launch::async,
                      [this, data]() { return process(data); });
  }

  [[nodiscard]] std::string describe() const {
    std::string result =
        "Pipeline with " + std::to_string(handlers_.size()) + " handlers:\n";
    for (size_t i = 0; i < handlers_.size(); ++i) {
      result += "  " + std::to_string(i) + ": " +
                handlers_[i].get().get_type_info() + "\n";
    }
    return result;
  }

  [[nodiscard]] bool empty() const { return handlers_.empty(); }
  [[nodiscard]] size_t size() const noexcept { return handlers_.size(); }

private:
  template <typename T>
  core::Result<T> dynamic_call_handle(IHandler *handler, T &data) {
    auto typed_handler = dynamic_cast<ITypedHandler<T, T> *>(handler);
    if (typed_handler) {
      return typed_handler->handle(data);
    }

    spdlog::error("Handler type mismatch for type: {}", typeid(T).name());
    return core::Result<T>::Error("Handler cannot process type");
  }

  template <typename T> bool setup_handler_chain() {
    if (handlers_.size() <= 1)
      return true;

    spdlog::info("Setting up handler chain for type: {}", typeid(T).name());

    for (size_t i = 0; i < handlers_.size() - 1; ++i) {
      auto &current_handler = handlers_[i].get();
      auto &next_handler = handlers_[i + 1].get();

      auto subscription_id =
          current_handler.get_event_bus().template Subscribe<T>(
              [&next_handler, this](T &data) {
                spdlog::debug("Passing data to next handler");
                next_handler.await();

                auto result = this->dynamic_call_handle<T>(&next_handler, data);
                if (result.is_ok()) {
                  next_handler.get_event_bus().Notify(result.value());
                } else {
                  spdlog::error("Handler chain interrupted: {}",
                                result.error().what());
                }
              });

      chain_subscriptions_.push_back({&current_handler, subscription_id});
    }

    return true;
  }

  void cleanup_all_chains() {
    for (const auto &[handler, subscription_id] : chain_subscriptions_) {
      handler->get_event_bus().Unsubscribe(subscription_id);
    }
    chain_subscriptions_.clear();
  }

private:
  std::vector<std::reference_wrapper<IHandler>> handlers_;
  std::vector<std::pair<IHandler *, core::Event::handlerID>>
      chain_subscriptions_;
};

} // namespace core::pipeline

#endif // OWL_VECTORFS_PIPELINE_HPP