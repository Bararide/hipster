#ifndef HIPSTER_UNIFIRED_POOL_ALLOCATOR_HPP
#define HIPSTER_UNIFIRED_POOL_ALLOCATOR_HPP

#include "pool_base.hpp"
#include <memory_resource>
#include <stdexcept>
#include <vector>

namespace hipster {

template <PoolType PoolT,
          GlobalMemoryProperty PropertyT = GlobalMemoryProperty::ANY>
class HsaPmrResource
    : public PoolBase<HsaPmrResource<PoolT, PropertyT>, PoolT, PropertyT>,
      public std::pmr::memory_resource {
public:
  using BasePool = PoolBase<HsaPmrResource<PoolT, PropertyT>, PoolT, PropertyT>;

  explicit HsaPmrResource(hsa_agent_t agent) : BasePool(agent) {}

  size_t size() const { return BasePool::size(); }
  size_t allocMaxSize() const { return BasePool::allocMaxSize(); }

  uint64_t poolHandle() const noexcept { return BasePool::pool().handle; }

private:
  void *do_allocate(size_t bytes, size_t alignment) override {
    return BasePool::allocate(bytes, 0);
  }

  void do_deallocate(void *p, size_t bytes, size_t alignment) override {
    BasePool::free(p);
  }

  bool
  do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
    if (auto *hsa_other = dynamic_cast<const HsaPmrResource *>(&other)) {
      return this->poolHandle() == hsa_other->poolHandle();
    }
    return false;
  }
};

} // namespace hipster

#endif // HIPSTER_UNIFIRED_POOL_ALLOCATOR_HPP