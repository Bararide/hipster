#ifndef OWL_VECTORFS_PIPELINE_HANDLER_HPP
#define OWL_VECTORFS_PIPELINE_HANDLER_HPP

#include <chrono>
#include "event.hpp"
#include "result.hpp"
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace core::pipeline {

class IHandler {
public:
  virtual ~IHandler() = default;
  virtual void await() = 0;
  virtual core::Event &get_event_bus() = 0;
  virtual std::string get_type_info() const = 0;
};

template <typename Input, typename Output>
class ITypedHandler : public IHandler {
public:
  virtual core::Result<Output> handle(Input &data) = 0;

  std::string get_type_info() const override {
    return "ITypedHandler<" + std::string(typeid(Input).name()) + ", " +
           std::string(typeid(Output).name()) + ">";
  }
};

template <typename Derived, typename Input, typename Output = Input>
class PipelineHandler : public ITypedHandler<Input, Output> {
public:
  PipelineHandler() = default;

  core::Event &get_event_bus() override { return event_bus_; }

  core::Result<Output> handle(Input &data) override {
    if constexpr (requires(Derived d) { d.handle(data); }) {
      return static_cast<Derived *>(this)->handle(data);
    } else {
      if constexpr (std::is_convertible_v<Input, Output>) {
        return core::Result<Output>::Ok(static_cast<Output>(data));
      } else {
        return core::Result<Output>::Error(
            "No conversion available from Input to Output");
      }
    }
  }

  void await() override {
    if constexpr (requires(Derived d) { d.await(); }) {
      static_cast<Derived *>(this)->await();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  std::string get_type_info() const override {
    return "PipelineHandler<" + std::string(typeid(Derived).name()) + ">";
  }

protected:
  ~PipelineHandler() = default;

private:
  core::Event event_bus_;
};

} // namespace owl::pipeline

#endif // OWL_VECTORFS_PIPELINE_HANDLER_HPP