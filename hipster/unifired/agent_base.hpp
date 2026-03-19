#ifndef HIPSTER_AGENT_BASE_HPP
#define HIPSTER_AGENT_BASE_HPP

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cpuid.h>
#include <cstring>
#include <functional>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <immintrin.h>
#include <iostream>
#include <liburing.h>
#include <linux/dma-buf.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <uring.hpp>
#include <random>
#include <xmmintrin.h>

namespace hipster {

class HSAInitializer {
public:
  HSAInitializer() {
    hsa_status_t status = hsa_init();
    if (status != HSA_STATUS_SUCCESS) {
      const char *err_string;
      hsa_status_string(status, &err_string);
      throw std::runtime_error("Failed to initialize HSA: " +
                               std::string(err_string));
    }
  }

  ~HSAInitializer() { hsa_shut_down(); }
};

enum class AgentType { CPU = HSA_DEVICE_TYPE_CPU, GPU = HSA_DEVICE_TYPE_GPU };

template <typename Derived, AgentType AgentT> class AgentBase {
public:
  virtual ~AgentBase() = default;

  hsa_agent_t agent() const noexcept { return agent_; }

  std::string name() const {
    char name[64] = {};
    hsa_agent_get_info(agent_, HSA_AGENT_INFO_NAME, name);
    return std::string(name);
  }

  bool isValid() const noexcept { return agent_.handle != 0; }

protected:
  AgentBase() : agent_{} {
    static HSAInitializer initializer;

    hsa_agent_t found_agent = {};

    hsa_status_t status = hsa_iterate_agents(
        [](hsa_agent_t agent, void *data) -> hsa_status_t {
          hsa_device_type_t type;
          hsa_status_t status =
              hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);

          if (status != HSA_STATUS_SUCCESS) {
            return status;
          }

          if (type == static_cast<hsa_device_type_t>(AgentT)) {
            *static_cast<hsa_agent_t *>(data) = agent;
            return HSA_STATUS_INFO_BREAK;
          }

          return HSA_STATUS_SUCCESS;
        },
        &found_agent);

    if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
      const char *err_string;
      hsa_status_string(status, &err_string);
      throw std::runtime_error("Failed to iterate HSA agents: " +
                               std::string(err_string));
    }

    if (found_agent.handle == 0) {
      throw std::runtime_error(std::string("No ") +
                               (AgentT == AgentType::CPU ? "CPU" : "GPU") +
                               " agent found");
    }

    agent_ = found_agent;
  }

private:
  hsa_agent_t agent_;
};

} // namespace hipster

#endif // HIPSTER_AGENT_BASE_HPP