#ifndef HIPSTER_POOL_BASE_HPP
#define HIPSTER_POOL_BASE_HPP

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace hipster {

enum class PoolType {
  GLOBAL = HSA_AMD_SEGMENT_GLOBAL,
  READONLY = HSA_AMD_SEGMENT_READONLY,
  PRIVATE = HSA_AMD_SEGMENT_PRIVATE,
  GROUP = HSA_AMD_SEGMENT_GROUP
};

template <typename Derived, PoolType PoolT> class PoolBase {
public:
  virtual ~PoolBase() = default;

  hsa_amd_memory_pool_t pool() const noexcept { return pool_; }

  size_t size() const {
    size_t pool_size = 0;
    hsa_amd_memory_pool_get_info(pool_, HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                 &pool_size);
    return pool_size;
  }

  size_t allocMaxSize() const {
    size_t max_alloc = 0;
    hsa_amd_memory_pool_get_info(pool_, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                 &max_alloc);
    return max_alloc;
  }

  bool runtimeAllocAllowed() const {
    bool alloc_allowed = false;
    hsa_amd_memory_pool_get_info(
        pool_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc_allowed);
    return alloc_allowed;
  }

  void *allocate(size_t size) {
    if (!runtimeAllocAllowed()) {
      throw std::runtime_error("Runtime allocation not allowed for this pool");
    }

    void *ptr = nullptr;
    hsa_status_t status = hsa_amd_memory_pool_allocate(pool_, size, 0, &ptr);
    if (status != HSA_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to allocate memory from pool");
    }
    return ptr;
  }

  void free(void *ptr) {
    if (ptr) {
      hsa_amd_memory_pool_free(ptr);
    }
  }

protected:
  explicit PoolBase(hsa_agent_t agent) {
    std::vector<hsa_amd_memory_pool_t> found_pools;

    hsa_amd_agent_iterate_memory_pools(
        agent,
        [](hsa_amd_memory_pool_t pool, void *data) -> hsa_status_t {
          auto *pools = static_cast<std::vector<hsa_amd_memory_pool_t> *>(data);
          hsa_amd_segment_t segment;
          hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                       &segment);

          if (segment == static_cast<hsa_amd_segment_t>(PoolT)) {
            pools->push_back(pool);
          }
          return HSA_STATUS_SUCCESS;
        },
        &found_pools);

    if (found_pools.empty()) {
      throw std::runtime_error("No suitable memory pool found");
    }

    pool_ = found_pools[0];
  }

private:
  hsa_amd_memory_pool_t pool_;
};

} // namespace hipster

#endif // HIPSTER_POOL_BASE_HPP