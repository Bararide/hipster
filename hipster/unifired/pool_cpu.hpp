#ifndef HIPSTER_POOL_CPU_HPP
#define HIPSTER_POOL_CPU_HPP

#include "agent_cpu.hpp"
#include "pool_base.hpp"

namespace hipster {

class CpuPool final : public PoolBase<CpuPool, PoolType::GLOBAL,
                                      GlobalMemoryProperty::COARSE_GRAINED> {
public:
  using Base =
      PoolBase<CpuPool, PoolType::GLOBAL, GlobalMemoryProperty::COARSE_GRAINED>;

  explicit CpuPool(const CpuAgent &agent) : Base(agent.agent()) {}

  void info() const {
    std::cout << "CPU Pool:\n"
              << "  Size: " << (size() / (1024 * 1024)) << " MB\n"
              << "  Max alloc: " << (allocMaxSize() / (1024 * 1024)) << " MB"
              << std::endl;
  }
};

} // namespace hipster

#endif // HIPSTER_POOL_CPU_HPP