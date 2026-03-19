#ifndef HIPSTER_UNIFIRED_POOL_GPU_HPP
#define HIPSTER_UNIFIRED_POOL_GPU_HPP

#include "agent_gpu.hpp"
#include "pool_base.hpp"

namespace hipster {

class GpuPool final : public PoolBase<GpuPool, PoolType::GLOBAL> {
public:
  using Base = PoolBase<GpuPool, PoolType::GLOBAL>;

  explicit GpuPool(const GpuAgent &agent) : Base(agent.agent()) {
    if (!runtimeAllocAllowed()) {
      std::cout << "⚠ GPU pool: runtime allocation not allowed" << std::endl;
    }
  }

  void info() const {
    std::cout << "GPU Pool:\n"
              << "  Size: " << (size() / (1024 * 1024)) << " MB\n"
              << "  Max alloc: " << (allocMaxSize() / (1024 * 1024)) << " MB\n"
              << "  Alloc allowed: " << (runtimeAllocAllowed() ? "yes" : "no")
              << std::endl;
  }
};

} // namespace hipster

#endif // HIPSTER_UNIFIRED_POOL_GPU_HPP