#ifndef HIPSTER_UNIFIRED_ASYNC_DMA_TRANSFER_HPP
#define HIPSTER_UNIFIRED_ASYNC_DMA_TRANSFER_HPP

#include "dma_buffer.hpp"
#include "pool_gpu.hpp"

namespace hipster {

namespace cpu_features {

[[nodiscard]] inline bool has_clwb() noexcept {
  unsigned eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx >> 24) & 1u;
}

[[nodiscard]] inline bool has_clflushopt() noexcept {
  unsigned eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx >> 23) & 1u;
}

[[nodiscard]] inline bool has_avx512f() noexcept {
  unsigned eax, ebx, ecx, edx;
  __cpuid_count(7, 0, eax, ebx, ecx, edx);
  return (ebx >> 16) & 1u;
}

struct Caps {
  bool clwb;
  bool clflushopt;
  bool avx512f;
  static const Caps &get() noexcept {
    static const Caps c{has_clwb(), has_clflushopt(), has_avx512f()};
    return c;
  }
};

} // namespace cpu_features

struct FlushPolicyClwb {};
struct FlushPolicyClflushopt {};
struct FlushPolicyNone {};
struct FlushPolicyAuto {};

template <typename Policy> struct CacheFlush;

template <> struct CacheFlush<FlushPolicyClwb> {
  __attribute__((target("clwb"))) static void flush(char *p,
                                                    size_t len) noexcept {
    for (size_t i = 0; i < len; i += 64) {
      _mm_clwb(p + i);
    }
    _mm_sfence();
  }
};

template <> struct CacheFlush<FlushPolicyClflushopt> {
  __attribute__((target("clflushopt"))) static void flush(char *p,
                                                          size_t len) noexcept {
    for (size_t i = 0; i < len; i += 64) {
      _mm_clflushopt(p + i);
    }
    _mm_sfence();
  }
};

template <> struct CacheFlush<FlushPolicyNone> {
  static void flush(char *, size_t) noexcept { _mm_sfence(); }
};

template <> struct CacheFlush<FlushPolicyAuto> {
  __attribute__((target("clwb,clflushopt"))) static void
  flush(char *p, size_t len) noexcept {
    const auto &caps = cpu_features::Caps::get();
    if (caps.clwb) {
      for (size_t i = 0; i < len; i += 64) {
        _mm_clwb(p + i);
      }
    } else if (caps.clflushopt) {
      for (size_t i = 0; i < len; i += 64) {
        _mm_clflushopt(p + i);
      }
    }
    _mm_sfence();
  }
};

inline void prefetch_write(const char *p, size_t len) noexcept {
  static const bool is_amd = [] {
    unsigned int eax, ebx, ecx, edx;
    __cpuid(0, eax, ebx, ecx, edx);
    return ebx == 0x68747541 && ecx == 0x444d4163 && edx == 0x69746e65;
  }();

  for (size_t i = 0; i < len; i += 64) {
    if (is_amd) {
      asm volatile("prefetchw (%0)" : : "r"(p + i) : "memory");
    } else {
      __builtin_prefetch(p + i, 1, 1);
    }
  }
}

struct AsyncOp {
  uint64_t id;
  std::function<void(int)> callback;
};

template <typename FlushPolicy = FlushPolicyAuto> class AsyncDmaTransfer {
public:
  static constexpr size_t kRingEntries = 256;
  static constexpr size_t kBatchSize = 32;

  explicit AsyncDmaTransfer(const GpuAgent &gpu_agent, GpuPool &gpu_pool)
      : gpu_agent_(gpu_agent), gpu_pool_(gpu_pool),
        uring_(eio::SQEntries{kRingEntries}, eio::Flags{0}) {
    completion_thread_ = std::jthread([this] { completion_loop(); });
  }

  ~AsyncDmaTransfer() {
    running_ = false;
    wakeup_completion_thread();
    if (completion_thread_.joinable())
      completion_thread_.join();
    if (buffer_registered_)
      io_uring_unregister_buffers(uring_.native_handle());
  }

  bool prepare(size_t size) {
    if (!buf_.create(size, gpu_agent_))
      return false;
    iov_.iov_base = buf_.cpu();
    iov_.iov_len = buf_.size();
    buffer_registered_ =
        io_uring_register_buffers(uring_.native_handle(), &iov_, 1) == 0;
    return true;
  }

  uint64_t write_async(const void *data, size_t offset, size_t len,
                       std::function<void(int)> cb) {
    uint64_t id = next_id_++;
    char *dst = static_cast<char *>(buf_.cpu()) + offset;

    prefetch_write(dst, len);
    memcpy(dst, data, len);
    CacheFlush<FlushPolicy>::flush(dst, len);

    {
      std::lock_guard lock(ops_mutex_);
      ops_[id] = {id, std::move(cb)};
    }

    submit_op(id, len, offset);
    return id;
  }

  struct Chunk {
    const void *data;
    size_t offset;
    size_t len;
  };

  std::vector<uint64_t> write_batch(std::span<const Chunk> chunks,
                                    std::function<void(int)> cb = {}) {
    std::vector<uint64_t> ids;
    ids.reserve(chunks.size());

    size_t pending = 0;
    for (const auto &c : chunks) {
      uint64_t id = next_id_++;
      char *dst = static_cast<char *>(buf_.cpu()) + c.offset;

      prefetch_write(dst, c.len);
      memcpy(dst, c.data, c.len);
      CacheFlush<FlushPolicy>::flush(dst, c.len);

      {
        std::lock_guard lock(ops_mutex_);
        ops_[id] = {id, cb};
      }

      prep_op(id, c.len, c.offset);
      ids.push_back(id);

      if (++pending >= kBatchSize) {
        (void)uring_.submit();
        pending = 0;
      }
    }
    if (pending > 0) {
      (void)uring_.submit();
    }

    return ids;
  }

  void wait(uint64_t id) {
    std::unique_lock lock(ops_mutex_);
    ops_cv_.wait(lock, [&] { return !ops_.count(id); });
  }

  void wait_all() {
    std::unique_lock lock(ops_mutex_);
    ops_cv_.wait(lock, [&] { return ops_.empty(); });
  }

  bool copy_to_gpu(void *gpu_dest, size_t size) {
    buf_.sync(DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START);

    hsa_signal_t sig;
    hsa_signal_create(1, 0, nullptr, &sig);

    hsa_agent_t agent = gpu_agent_.agent();
    hsa_status_t status = hsa_amd_memory_async_copy(
        gpu_dest, agent, buf_.gpu(), agent, size, 0, nullptr, sig);

    bool ok = false;
    if (status == HSA_STATUS_SUCCESS) {
      hsa_signal_wait_acquire(sig, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
      buf_.sync(DMA_BUF_SYNC_END);
      ok = true;
    }
    hsa_signal_destroy(sig);
    return ok;
  }

  [[nodiscard]] const DmaBuffer &buffer() const noexcept { return buf_; }
  [[nodiscard]] bool fixed_buffer() const noexcept {
    return buffer_registered_;
  }

private:
  const GpuAgent &gpu_agent_;
  GpuPool &gpu_pool_;
  DmaBuffer buf_;
  eio::Uring<eio::SQEntries, eio::Flags> uring_;
  struct iovec iov_ {};
  bool buffer_registered_ = false;

  std::unordered_map<uint64_t, AsyncOp> ops_;
  std::mutex ops_mutex_;
  std::condition_variable ops_cv_;
  std::atomic<uint64_t> next_id_{1};
  std::atomic<bool> running_{true};
  std::jthread completion_thread_;

  void prep_op(uint64_t id, size_t len, size_t offset) {
    if (buffer_registered_) {
      eio::WriteFixed op(buf_.fd(), 0, len, offset, id, 0);
      uring_.prep(op);
    } else {
      eio::Write op(buf_.fd(), static_cast<char *>(buf_.cpu()) + offset, len,
                    offset, id);
      uring_.prep(op);
    }
  }

  void submit_op(uint64_t id, size_t len, size_t offset) {
    prep_op(id, len, offset);
    (void)uring_.submit();
  }

  void wakeup_completion_thread() {
    io_uring_sqe *sqe = io_uring_get_sqe(uring_.native_handle());
    if (sqe) {
      io_uring_prep_nop(sqe);
      io_uring_sqe_set_data64(sqe, 0);
      io_uring_submit(uring_.native_handle());
    }
  }

  void completion_loop() {
    while (true) {
      try {
        auto c = uring_.wait_completion();
        if (c.user_data != 0)
          dispatch(c.user_data, c.result);
        while (auto opt = uring_.peek_completion()) {
          if (opt->user_data != 0)
            dispatch(opt->user_data, opt->result);
        }
      } catch (...) {
      }
      if (!running_)
        break;
    }
  }

  void dispatch(uint64_t id, int result) {
    {
      std::lock_guard lock(ops_mutex_);
      if (auto it = ops_.find(id); it != ops_.end()) {
        if (it->second.callback)
          it->second.callback(result);
        ops_.erase(it);
      }
    }
    ops_cv_.notify_all();
  }
};

using AsyncDmaTransferAuto = AsyncDmaTransfer<FlushPolicyAuto>;

} // namespace hipster

#endif