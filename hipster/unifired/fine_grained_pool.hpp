#ifndef HIPSTER_POOL_GPU_FINE_GRAINED_HPP
#define HIPSTER_POOL_GPU_FINE_GRAINED_HPP

#include "agent_gpu.hpp"
#include "pool_base.hpp"

namespace hipster {

class GpuFineGrainedPool final
    : public PoolBase<GpuFineGrainedPool, PoolType::GLOBAL> {
public:
  using Base = PoolBase<GpuFineGrainedPool, PoolType::GLOBAL>;

  explicit GpuFineGrainedPool(const GpuAgent &agent) : Base(agent.agent()) {
    uint32_t global_flags = 0;
    hsa_amd_memory_pool_get_info(pool(), HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                 &global_flags);
    fine_grained_ =
        (global_flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) != 0;
  }

  bool isFineGrained() const noexcept { return fine_grained_; }

  void *allocateCoherent(size_t size) { return allocate(size); }

private:
  bool fine_grained_ = false;
};

} // namespace hipster

#endif // HIPSTER_POOL_GPU_FINE_GRAINED_HPP