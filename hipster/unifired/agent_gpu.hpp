#ifndef HIPSTER_UNIFIRED_AGENT_GPU_HPP
#define HIPSTER_UNIFIRED_AGENT_GPU_HPP

#include "agent_base.hpp"

namespace hipster {

class GpuAgent final : public AgentBase<GpuAgent, AgentType::GPU> {
public:
  using Base = AgentBase<GpuAgent, AgentType::GPU>;
  using Base::Base;

  void info() {
    std::cout << "GPU: " << name() << "\n";
  }
};

} // namespace hipster

#endif // HIPSTER_UNIFIRED_AGENT_GPU_HPP