#ifndef HIPSTER_POOL_BASE_HPP
#define HIPSTER_POOL_BASE_HPP

#include "utils.hpp"

namespace hipster {

enum class PoolType {
  GLOBAL = HSA_AMD_SEGMENT_GLOBAL,
  READONLY = HSA_AMD_SEGMENT_READONLY,
  PRIVATE = HSA_AMD_SEGMENT_PRIVATE,
  GROUP = HSA_AMD_SEGMENT_GROUP
};

enum class GlobalMemoryProperty {
  ANY = 0,
  FINE_GRAINED = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED,
  COARSE_GRAINED = HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED
};

template <typename Derived, PoolType PoolT,
          GlobalMemoryProperty PropertyT = GlobalMemoryProperty::ANY>
class PoolBase {
public:
  ~PoolBase() = default;

  hsa_amd_memory_pool_t pool() const noexcept { return pool_; }

  size_t size() const {
    size_t pool_size = 0;
    checkStatus(hsa_amd_memory_pool_get_info(
                    pool_, HSA_AMD_MEMORY_POOL_INFO_SIZE, &pool_size),
                "Failed to get pool size");
    return pool_size;
  }

  size_t allocMaxSize() const {
    size_t max_alloc = 0;
    checkStatus(hsa_amd_memory_pool_get_info(
                    pool_, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE, &max_alloc),
                "Failed to get max alloc size");
    return max_alloc;
  }

  bool runtimeAllocAllowed() const {
    bool alloc_allowed = false;
    checkStatus(hsa_amd_memory_pool_get_info(
                    pool_, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                    &alloc_allowed),
                "Failed to check if runtime alloc allowed");
    return alloc_allowed;
  }

  size_t alignment() const {
    size_t align = 0;
    checkStatus(hsa_amd_memory_pool_get_info(
                    pool_, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &align),
                "Failed to get pool alignment");
    return align;
  }

  void *allocate(size_t size, uint32_t flags = 0) {
    if (!runtimeAllocAllowed()) {
      throw std::runtime_error("Runtime allocation not allowed for this pool");
    }

    void *ptr = nullptr;
    hsa_status_t status =
        hsa_amd_memory_pool_allocate(pool_, size, flags, &ptr);
    if (status != HSA_STATUS_SUCCESS) {
      throw std::runtime_error(
          "Failed to allocate memory from pool. Status code: " +
          std::to_string(status));
    }
    return ptr;
  }

  void free(void *ptr) noexcept {
    if (ptr) {
      hsa_amd_memory_pool_free(ptr);
    }
  }

protected:
  explicit PoolBase(hsa_agent_t agent) {
    struct Context {
      std::vector<hsa_amd_memory_pool_t> pools;
      GlobalMemoryProperty required_property;
    } context;

    context.required_property = PropertyT;

    hsa_amd_agent_iterate_memory_pools(
        agent,
        [](hsa_amd_memory_pool_t pool, void *data) -> hsa_status_t {
          auto *ctx = static_cast<Context *>(data);
          hsa_amd_segment_t segment;
          hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                       &segment);

          if (segment == static_cast<hsa_amd_segment_t>(PoolT)) {
            if (PoolT == PoolType::GLOBAL &&
                ctx->required_property != GlobalMemoryProperty::ANY) {
              uint32_t flags = 0;
              hsa_amd_memory_pool_get_info(
                  pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);

              if (!(flags & static_cast<uint32_t>(ctx->required_property))) {
                return HSA_STATUS_SUCCESS;
              }
            }
            ctx->pools.push_back(pool);
          }
          return HSA_STATUS_SUCCESS;
        },
        &context);

    if (context.pools.empty()) {
      throw std::runtime_error("No suitable memory pool found matching the "
                               "specified Segment and GlobalMemoryProperty");
    }

    pool_ = context.pools[0];
  }

private:
  hsa_amd_memory_pool_t pool_;
};

} // namespace hipster

#endif // HIPSTER_POOL_BASE_HPP
