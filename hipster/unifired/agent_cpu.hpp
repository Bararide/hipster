#ifndef HIPSTER_UNIFIRED_AGENT_CPU_HPP
#define HIPSTER_UNIFIRED_AGENT_CPU_HPP

#include "agent_base.hpp"

namespace hipster {

class CpuAgent final : public AgentBase<CpuAgent, AgentType::CPU> {
public:
  using Base = AgentBase<CpuAgent, AgentType::CPU>;
  using Base::Base;

  void info() {
    std::cout << "CPU: " << name() << "\n";
  }
};

} // namespace hipster

#endif // HIPSTER_UNIFIRED_AGENT_CPU_HPP