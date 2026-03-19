#ifndef HIPSTER_UNIFIRED_DMA_BUFFER_HPP
#define HIPSTER_UNIFIRED_DMA_BUFFER_HPP

#include "agent_gpu.hpp"

namespace hipster {

class DmaBuffer {
public:
  static constexpr size_t kAlignment = 4096;
  static constexpr size_t kHugePage = 2 * 1024 * 1024;

  DmaBuffer() = default;
  DmaBuffer(const DmaBuffer &) = delete;
  DmaBuffer &operator=(const DmaBuffer &) = delete;

  DmaBuffer(DmaBuffer &&o) noexcept
      : fd_(o.fd_), cpu_ptr_(o.cpu_ptr_), gpu_ptr_(o.gpu_ptr_), size_(o.size_) {
    o.fd_ = -1;
    o.cpu_ptr_ = nullptr;
    o.gpu_ptr_ = nullptr;
    o.size_ = 0;
  }

  ~DmaBuffer() { release(); }

  bool create(size_t size, const GpuAgent &gpu_agent) {
    size_ = align(size, kAlignment);

    fd_ = memfd_create("dma-buffer", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd_ < 0 || ftruncate(fd_, size_) < 0) {
      if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
      }
      return false;
    }

    int flags = MAP_SHARED | MAP_POPULATE;
    if (size_ >= kHugePage)
      flags |= MAP_HUGETLB;

    cpu_ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, flags, fd_, 0);
    if (cpu_ptr_ == MAP_FAILED && (flags & MAP_HUGETLB)) {
      flags &= ~MAP_HUGETLB;
      cpu_ptr_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, flags, fd_, 0);
    }
    if (cpu_ptr_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      cpu_ptr_ = nullptr;
      return false;
    }

    madvise(cpu_ptr_, size_, MADV_SEQUENTIAL);
    madvise(cpu_ptr_, size_, MADV_DONTFORK);

    hsa_agent_t agent = gpu_agent.agent();
    hsa_status_t st =
        hsa_amd_memory_lock(cpu_ptr_, size_, &agent, 1, &gpu_ptr_);
    if (st != HSA_STATUS_SUCCESS) {
      munmap(cpu_ptr_, size_);
      close(fd_);
      cpu_ptr_ = nullptr;
      fd_ = -1;
      return false;
    }

    return true;
  }

  void sync(int flags) const noexcept {
    if (fd_ < 0)
      return;
    struct dma_buf_sync s {
      .flags = static_cast<__u64>(flags)
    };
    ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s);
  }

  [[nodiscard]] int fd() const noexcept { return fd_; }
  [[nodiscard]] void *cpu() const noexcept { return cpu_ptr_; }
  [[nodiscard]] void *gpu() const noexcept { return gpu_ptr_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] bool valid() const noexcept { return fd_ >= 0 && cpu_ptr_; }

private:
  int fd_ = -1;
  void *cpu_ptr_ = nullptr;
  void *gpu_ptr_ = nullptr;
  size_t size_ = 0;

  static constexpr size_t align(size_t s, size_t a) noexcept {
    return (s + a - 1) & ~(a - 1);
  }

  void release() noexcept {
    if (gpu_ptr_) {
      hsa_amd_memory_unlock(cpu_ptr_);
      gpu_ptr_ = nullptr;
    }
    if (cpu_ptr_ && cpu_ptr_ != MAP_FAILED) {
      munmap(cpu_ptr_, size_);
      cpu_ptr_ = nullptr;
    }
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }
};

} // namespace hipster

#endif