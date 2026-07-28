#ifndef HIPSTER_UNIFIRED_ASYNC_DMA_TRANSFER_HPP
#define HIPSTER_UNIFIRED_ASYNC_DMA_TRANSFER_HPP

#include "cache.hpp"
#include "dma_buffer.hpp"
#include "pool_gpu.hpp"
#include "signal.hpp"
#include "utils.hpp"

namespace hipster {

struct AsyncOp {
  uint64_t id;
  std::function<void(int)> callback;
};

template <typename T> class DmaAllocator {
public:
  using value_type = T;
  DmaAllocator(DmaBuffer &buf) : buffer_(&buf) {}

  T *allocate(std::size_t n) {
    return static_cast<T *>(buffer_->cpu()) + current_offset_;
  }
  void deallocate(T *, std::size_t) noexcept {}

private:
  DmaBuffer *buffer_;
  size_t current_offset_ = 0;
};

template <typename FlushPolicy = FlushPolicyAuto> class AsyncDmaTransfer {
public:
  explicit AsyncDmaTransfer(const GpuAgent &gpu_agent, GpuPool &gpu_pool)
      : gpu_agent_(gpu_agent), gpu_pool_(gpu_pool) {
    hsa_amd_profiling_async_copy_enable(true);
  }

  ~AsyncDmaTransfer() = default;

  bool prepare(size_t size, const char *name) {
    return buf_.create(size, gpu_agent_, name);
  }

  uint64_t writeAsync(const void *data, size_t offset, size_t len) {
    char *dst = static_cast<char *>(buf_.cpu()) + offset;
    std::memcpy(dst, data, len);

    size_t current_min = dirty_min_.load(std::memory_order_relaxed);
    while (offset < current_min &&
           !dirty_min_.compare_exchange_weak(current_min, offset,
                                             std::memory_order_relaxed)) {
    }

    size_t current_max = dirty_max_.load(std::memory_order_relaxed);
    size_t new_max = offset + len;
    while (new_max > current_max &&
           !dirty_max_.compare_exchange_weak(current_max, new_max,
                                             std::memory_order_relaxed)) {
    }

    return next_id_++;
  }

  struct Chunk {
    const void *data;
    size_t offset;
    size_t len;
  };

  std::vector<uint64_t> write_batch(std::span<const Chunk> chunks) {
    std::vector<uint64_t> ids;
    ids.reserve(chunks.size());

    for (const auto &c : chunks) {
      ids.push_back(writeAsync(c.data, c.offset, c.len));
    }
    return ids;
  }

  void flushDirtyRange() {
    if (dirty_min_ < dirty_max_) {
      size_t len = dirty_max_ - dirty_min_;
      char *start = static_cast<char *>(buf_.cpu()) + dirty_min_;

      CacheFlush<FlushPolicy>::flush(start, len);

      dirty_min_ = std::numeric_limits<size_t>::max();
      dirty_max_ = 0;
    }
  }
  bool copyToGpu(void *gpu_dest, size_t size) {
    flushDirtyRange();

    // if (buf_.sync(DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START)) {
    //   return false;
    // }

    HsaSignal sig(1, 0);

    hsa_agent_t agent = gpu_agent_.agent();

    hsa_status_t status = hsa_amd_memory_async_copy(
        gpu_dest, agent, buf_.gpu(), agent, size, 0, nullptr, sig.get());

    hsa_amd_profiling_async_copy_time_t copy_time;
    hsa_amd_profiling_get_async_copy_time(sig.get(), &copy_time);

    bool ok = false;
    if (status == HSA_STATUS_SUCCESS) {
      hsa_signal_wait_acquire(sig.get(), HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);

      // if (buf_.sync(DMA_BUF_SYNC_END)) {
      //   return false;
      // }

      ok = true;
    }

    uint64_t system_tick_start, system_tick_end;
    double duration_ns = (copy_time.end - copy_time.start);

    std::cout << "Real copy time on SDMA: " << duration_ns / 1000.0 << " mcs\n";

    return ok;
  }

  [[nodiscard]] const DmaBuffer &buffer() const noexcept { return buf_; }

private:
  const GpuAgent &gpu_agent_;
  GpuPool &gpu_pool_;
  DmaBuffer buf_;

  std::atomic<size_t> dirty_min_{std::numeric_limits<size_t>::max()};
  std::atomic<size_t> dirty_max_{0};

  std::atomic<uint64_t> next_id_{1};
};

using AsyncDmaTransferAuto = AsyncDmaTransfer<FlushPolicyAuto>;

} // namespace hipster

#endif