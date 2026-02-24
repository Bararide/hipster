#ifndef HIPSTER_MEMORY_HPP
#define HIPSTER_MEMORY_HPP

#include "alies.hpp"

namespace hipster {

enum class MemoryType { Device, Host, Managed };

template <MemoryType MT = MemoryType::Managed> class Memory {
public:
  Memory() = default;

  ~Memory() { deallocate(); }

  Memory(const Memory &) = delete;
  Memory &operator=(const Memory &) = delete;

  Memory(Memory &&other) noexcept : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
  }

  Memory &operator=(Memory &&other) noexcept {
    if (this != &other) {
      deallocate();
      ptr_ = other.ptr_;
      size_ = other.size_;
      other.ptr_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  int allocate(size_t size) {
    if (ptr_ != nullptr) {
      std::cerr << "Memory already allocated" << std::endl;
      return 1;
    }

    size_ = size;
    hipError_t err = hipSuccess;

    if constexpr (MT == MemoryType::Device) {
      err = hipMalloc(&ptr_, size);
    } else if constexpr (MT == MemoryType::Host) {
      err = hipHostMalloc(&ptr_, size);
    } else if constexpr (MT == MemoryType::Managed) {
      err = hipMallocManaged(&ptr_, size);
    } else {
      std::cerr << "Memory type not specified" << std::endl;
      return 1;
    }

    if (err != hipSuccess) {
      std::cerr << "Allocation failed: " << hipGetErrorString(err) << std::endl;
      ptr_ = nullptr;
      size_ = 0;
      return 1;
    }

    return 0;
  }

  int copyToDevice(const void *host_data, size_t size) {
    if constexpr (MT == MemoryType::Device) {
      if (!ptr_ || !host_data) {
        return 1;
      }
      hipError_t err =
          hipMemcpyAsync(ptr_, host_data, size, hipMemcpyHostToDevice);
      return (err == hipSuccess) ? 0 : 1;
    } else {
      std::cerr << "copyToDevice only for Device memory" << std::endl;
      return 1;
    }
  }

  int copyToHost(void *host_data, size_t size) {
    if constexpr (MT == MemoryType::Device) {
      if (!ptr_ || !host_data) {
        return 1;
      }
      hipError_t err =
          hipMemcpyAsync(host_data, ptr_, size, hipMemcpyDeviceToHost);
      return (err == hipSuccess) ? 0 : 1;
    } else {
      std::cerr << "copyToHost only for Device memory" << std::endl;
      return 1;
    }
  }

  int sync() {
    if constexpr (MT == MemoryType::Managed) {
      hipError_t err = hipDeviceSynchronize();
      return (err == hipSuccess) ? 0 : 1;
    }
    return 0;
  }

  void *get() const { return ptr_; }
  size_t size() const { return size_; }
  bool isAllocated() const { return ptr_ != nullptr; }

  explicit operator bool() const { return isAllocated(); }

  template <typename T> T *as() const { return static_cast<T *>(ptr_); }

private:
  void deallocate() {
    if (!ptr_) {
      return;
    }

    hipError_t err = hipSuccess;

    if constexpr (MT == MemoryType::Device) {
      err = hipFree(ptr_);
    } else if constexpr (MT == MemoryType::Host) {
      err = hipHostFree(ptr_);
    } else if constexpr (MT == MemoryType::Managed) {
      err = hipFree(ptr_);
    }

    if (err != hipSuccess) {
      std::cerr << "Allocation failed: " << hipGetErrorString(err) << std::endl;
    }

    ptr_ = nullptr;
    size_ = 0;
  }

  void *ptr_ = nullptr;
  size_t size_ = 0;
};

template <typename T> using DeviceMemory = Memory<MemoryType::Device>;

template <typename T> using HostMemory = Memory<MemoryType::Host>;

template <typename T> using ManagedMemory = Memory<MemoryType::Managed>;

} // namespace hipster

#endif // HIPSTER_MEMORY_HPP