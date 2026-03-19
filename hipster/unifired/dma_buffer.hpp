#ifndef HIPSTER_UNIFIRED_DMA_BUFFER_HPP
#define HIPSTER_UNIFIRED_DMA_BUFFER_HPP

#include "agent_gpu.hpp"

namespace hipster {

class DmaBuffer {
public:
  static constexpr size_t kAligment = 4096;

  DmaBuffer() = default;

  DmaBuffer(const DmaBuffer &) = delete;
  DmaBuffer &operator=(const DmaBuffer &) = delete;

  DmaBuffer(DmaBuffer &&o) noexcept
      : fd_(o.fd_), cpu_ptr_(o.cpu_ptr_), gpu_ptr_(o.gpu_ptr_),
        size_(o.size_) {
    o.fd_ = -1;
    o.cpu_ptr_ = nullptr;
    o.gpu_ptr_ = nullptr;
    o.size_ = 0;
  }

  ~DmaBuffer() { release(); }

  bool create(size_t size, const GpuAgent &gpu_agent) {
    size_ = align(size);

    fd_ = memfd_create("dmabuf-buffer", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (fd_ < 0)
      return false;

    if (ftruncate(fd_, size_) < 0) {
      close(fd_);
      fd_ = -1;
      return false;
    }

    cpu_ptr_ =
        mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (cpu_ptr_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      cpu_ptr_ = nullptr;
      return false;
    }

    hsa_agent_t agent = gpu_agent.agent();
    hsa_status_t status =
        hsa_amd_memory_lock(cpu_ptr_, size_, &agent, 1, &gpu_ptr_);
    if (status != HSA_STATUS_SUCCESS) {
      munmap(cpu_ptr_, size_);
      close(fd_);
      cpu_ptr_ = nullptr;
      fd_ = -1;
      return false;
    }

    return true;
  }

  void sync(int flags) const {
    if (fd_ < 0) {
      return;
    }
    struct dma_buf_sync s = {.flags = static_cast<__u64>(flags)};
    ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s);
  }

  int fd() const noexcept { return fd_; }
  void *cpu() const noexcept { return cpu_ptr_; }
  void *gpu() const noexcept { return gpu_ptr_; }
  size_t size() const noexcept { return size_; }
  bool valid() const noexcept { return fd_ >= 0 && cpu_ptr_ != nullptr; }

private:
  int fd_ = -1;
  void *cpu_ptr_ = nullptr;
  void *gpu_ptr_ = nullptr;
  size_t size_ = 0;

  static size_t align(size_t s) {
    return (s + kAligment - 1) & ~(kAligment - 1);
  }

  void release() {
    if (gpu_ptr_) {
      hsa_amd_memory_pool_free(gpu_ptr_);
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

#endif // HIPSTER_UNIFIRED_DMA_BUFFER_HPP
