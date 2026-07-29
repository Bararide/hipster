#ifndef HIPSTER_HIPSTER_HPP
#define HIPSTER_HIPSTER_HPP

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

#include "dataframe.hpp"
#include "event.hpp"
#include "graph.hpp"
#include "memory.hpp"
#include "row.hpp"
#include "stream.hpp"
#include "utils.hpp"

namespace hipster {

struct LaunchConfig {
  dim3 grid_dim;
  dim3 block_dim;
  size_t shared_memory_bytes;

  LaunchConfig(dim3 grid, dim3 block, size_t shared = 0)
      : grid_dim(grid), block_dim(block), shared_memory_bytes(shared) {}
};

template <DeviceVersion V = DeviceVersion::RDNA3> class Hipster {
public:
  Hipster(int device_n = 0) : device_(device_n) {
    hipError_t err = hipSetDevice(device_);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipSetDevice failed: ") +
                               hipGetErrorString(err));
    }

    err = hipGetDeviceProperties(&props_, device_);
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipGetDeviceProperties failed: ") +
                               hipGetErrorString(err));
    }

    initOptimalParams();
  }

  Hipster(const Hipster &) = delete;
  Hipster &operator=(const Hipster &) = delete;

  Hipster(Hipster &&other) noexcept = delete;
  Hipster &operator=(Hipster &&other) noexcept = delete;

  template <typename T>
  Memory<MemoryType::Managed> allocateManaged(size_t elements) {
    Memory<MemoryType::Managed> mem;
    if (mem.allocate(elements * sizeof(T)) != 0) {
      throw std::runtime_error("Failed to allocate managed memory");
    }
    return mem;
  }

  template <typename T>
  Memory<MemoryType::Device> allocateDevice(size_t elements) {
    Memory<MemoryType::Device> mem;
    if (mem.allocate(elements * sizeof(T)) != 0) {
      throw std::runtime_error("Failed to allocate device memory");
    }
    return mem;
  }

  template <typename T> Memory<MemoryType::Host> allocateHost(size_t elements) {
    Memory<MemoryType::Host> mem;
    if (mem.allocate(elements * sizeof(T)) != 0) {
      throw std::runtime_error("Failed to allocate host memory");
    }
    return mem;
  }

  HipStream createStream(unsigned int flags = HipStreamNonBlocking,
                         int priority = 0) {
    return HipStream(flags, priority);
  }

  HipEvent createEvent(unsigned int flags = HipEventDisableTiming) {
    return HipEvent(flags);
  }

  HipGraph createGraph() { return HipGraph(); }

  LaunchConfig getOptimalLaunchConfig(size_t n, size_t elements_per_thread = 1,
                                      size_t shared_mem_per_block = 0) {
    dim3 block_dim = getOptimalBlockDim(elements_per_thread);
    dim3 grid_dim = getOptimalGridDim(n, block_dim.x);
    size_t total_shared = block_dim.x * shared_mem_per_block;

    return LaunchConfig(grid_dim, block_dim, total_shared);
  }

  dim3 getOptimalBlockDim(int elements_per_thread = 1) const {
    return dim3(optimal_block_size_);
  }

  dim3 getOptimalGridDim(int n, int block_size = 0) const {
    if (block_size == 0) {
      block_size = optimal_block_size_;
    }
    return dim3((n + block_size - 1) / block_size);
  }

  template <typename T> void fillHostMemory(T *ptr, size_t count, T value) {
    std::fill(ptr, ptr + count, value);
  }

  template <typename T> double reduceHostMemory(const T *ptr, size_t count) {
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i) {
      sum += static_cast<double>(ptr[i]);
    }
    return sum;
  }

  static void checkLastError(const std::string &context = "") {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      std::string message = "HIP error";
      if (!context.empty()) {
        message += " in " + context;
      }
      message += ": " + std::string(hipGetErrorString(err));
      throw std::runtime_error(message);
    }
  }

  static void checkHipError(hipError_t err, const std::string &context = "") {
    if (err != hipSuccess) {
      std::string message = "HIP error";
      if (!context.empty()) {
        message += " in " + context;
      }
      message += ": " + std::string(hipGetErrorString(err));
      throw std::runtime_error(message);
    }
  }

  void synchronize() {
    checkHipError(hipDeviceSynchronize(), "device synchronize");
  }

  void synchronize(HipStream &stream) { stream.synchronize(); }

  void synchronize(HipEvent &event) { event.synchronize(); }

  std::string getDeviceName() const { return props_.name; }
  int getDeviceId() const { return device_; }
  size_t getTotalMemory() const { return props_.totalGlobalMem; }
  int getMaxThreadsPerBlock() const { return props_.maxThreadsPerBlock; }
  int getMultiProcessorCount() const { return props_.multiProcessorCount; }
  int getWarpSize() const { return props_.warpSize; }
  bool isIntegrated() const { return props_.integrated; }

  template <typename Kernel, typename... Args>
  void launchKernel(Kernel kernel, const LaunchConfig &config,
                    HipStream &stream, Args... args) {
    void *kernel_args[] = {(void *)&args...};

    hipError_t err =
        hipLaunchKernel(reinterpret_cast<const void *>(kernel), config.grid_dim,
                        config.block_dim, kernel_args,
                        config.shared_memory_bytes, stream.get());

    checkHipError(err, "hipLaunchKernel");
  }

private:
  void initOptimalParams() {
    if constexpr (V == DeviceVersion::RDNA3) {
      optimal_block_size_ = 256;
      optimal_grid_multiplier_ = 4;
    } else {
      optimal_block_size_ = 256;
      optimal_grid_multiplier_ = 2;
    }
  }

  int device_;
  hipDeviceProp_t props_;
  int optimal_block_size_ = 256;
  int optimal_grid_multiplier_ = 4;
};

} // namespace hipster

#endif // HIPSTER_HIPSTER_HPP