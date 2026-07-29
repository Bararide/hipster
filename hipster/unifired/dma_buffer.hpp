#ifndef HIPSTER_UNIFIRED_DMA_BUFFER_HPP
#define HIPSTER_UNIFIRED_DMA_BUFFER_HPP

#include "agent_gpu.hpp"
#include "mmap.hpp"

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

  DmaBuffer &operator=(DmaBuffer &&o) noexcept {
    if (this != &o) {
      release();
      fd_ = o.fd_;
      cpu_ptr_ = o.cpu_ptr_;
      gpu_ptr_ = o.gpu_ptr_;
      size_ = o.size_;

      o.fd_ = -1;
      o.cpu_ptr_ = nullptr;
      o.gpu_ptr_ = nullptr;
      o.size_ = 0;
    }
    return *this;
  }

  ~DmaBuffer() { release(); }

  template <IsMmapOption... Options>
  bool create(size_t size, const GpuAgent &gpu_agent, const char *name,
              Options &&...opts) {
    size_ = align(size, kAlignment);

    MmapConfig cfg;
    DefaultDmaConfig{}.apply(cfg);

    (opts.apply(cfg), ...);

    fd_ = memfd_create(name, cfg.memfd_flags);
    if (fd_ < 0 || ftruncate(fd_, size_) < 0) {
      if (fd_ >= 0) {
        close(fd_);
        fd_ = -1;
      }

      return false;
    }

    cpu_ptr_ = mmap(nullptr, size_, cfg.prot, cfg.map_flags, fd_, 0);

    if (cpu_ptr_ == MAP_FAILED && (cfg.map_flags & MAP_HUGETLB)) {
      cfg.map_flags &= ~MAP_HUGETLB;
      cpu_ptr_ = mmap(nullptr, size_, cfg.prot, cfg.map_flags, fd_, 0);
    }

    if (cpu_ptr_ == MAP_FAILED) {
      close(fd_);
      fd_ = -1;
      cpu_ptr_ = nullptr;
      return false;
    }

    if (cfg.madvise != 0) {
      madvise(cpu_ptr_, size_, cfg.madvise);
    }

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

  [[nodiscard]] bool sync(int flags) const noexcept {
    if (fd_ < 0) {
      return false;
    }

    struct dma_buf_sync s {
      .flags = static_cast<__u64>(flags)
    };

    int result = ioctl(fd_, DMA_BUF_IOCTL_SYNC, &s);
    if (result < 0) {
      return false;
    }
    return true;
  }

  template <typename T>
  bool write(size_t offset_bytes, const T *data, size_t count) noexcept {
    if (!cpu_ptr_ || !data || count == 0) {
      return false;
    }

    size_t bytes_to_write = count * sizeof(T);
    if (offset_bytes + bytes_to_write > size_) {
      return false;
    }

    char *dst = static_cast<char *>(cpu_ptr_) + offset_bytes;
    std::memcpy(dst, data, bytes_to_write);

    return true;
  }

  template <typename T>
  bool read(size_t offset_bytes, T *data, size_t count) const noexcept {
    if (!cpu_ptr_ || !data || count == 0) {
      return false;
    }

    size_t bytes_to_read = count * sizeof(T);
    if (offset_bytes + bytes_to_read > size_) {
      return false;
    }

    const char *src = static_cast<const char *>(cpu_ptr_) + offset_bytes;
    std::memcpy(data, src, bytes_to_read);

    return true;
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

#endif // HIPSTER_UNIFIRED_DMA_BUFFER_HPP