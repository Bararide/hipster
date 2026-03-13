#ifndef HIPSTER_HIPSTER_HPP
#define HIPSTER_HIPSTER_HPP

#include "event.hpp"
#include "graph.hpp"
#include "memory.hpp"
#include "stream.hpp"
#include "dataframe.hpp"
#include "utils.hpp"

namespace hipster {

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

  Hipster(Hipster &&other) noexcept = default;
  Hipster &operator=(Hipster &&other) noexcept = default;

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

  template <typename Kernel, typename... Args>
  void launchKernel(Kernel kernel, dim3 gridDim, dim3 blockDim,
                    HipStream &stream, Args... args) {
    void *kernel_args[] = {(void *)&args...};
    hipLaunchKernelGGL(kernel, gridDim, blockDim, 0, stream.get(), args...);
  }

  template <typename Kernel, typename... Args>
  HipGraphNode
  addKernelToGraph(HipGraph &graph, const std::vector<HipGraphNode> &deps,
                   dim3 gridDim, dim3 blockDim, Kernel kernel, Args... args) {
    void *kernel_args[] = {(void *)&args...};
    return graph.addKernelNode((const void *)kernel, deps, gridDim, blockDim,
                               kernel_args, 0);
  }

  template <typename F> double measureTimeMicroseconds(F &&func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
  }

  template <typename F> double measureGpuTime(F &&func, HipStream &stream) {
    HipEvent start(0);
    HipEvent stop(0);

    start.record(stream.get());
    func();
    stop.record(stream.get());

    while (!stop.ready()) {
      stream.waitEvent(stop);
    }

    float ms = 0.0f;
    hipError_t err = hipEventElapsedTime(&ms, start.get(), stop.get());
    if (err != hipSuccess) {
      throw std::runtime_error(std::string("hipEventElapsedTime failed: ") +
                               hipGetErrorString(err));
    }

    return ms * 1000.0;
  }

  std::string getDeviceName() const { return props_.name; }

  int getDeviceId() const { return device_; }

  size_t getTotalMemory() const { return props_.totalGlobalMem; }

  int getMaxThreadsPerBlock() const { return props_.maxThreadsPerBlock; }

  int getMultiProcessorCount() const { return props_.multiProcessorCount; }

  int getWarpSize() const { return props_.warpSize; }

  bool isIntegrated() const { return props_.integrated; }

  dim3 getOptimalBlockDim(int elements_per_thread = 1) const {
    return dim3(optimal_block_size_);
  }

  dim3 getOptimalGridDim(int n, int block_size = 0) const {
    if (block_size == 0)
      block_size = optimal_block_size_;
    return dim3((n + block_size - 1) / block_size);
  }

  template <typename Kernel>
  int getMaxActiveBlocks(Kernel kernel, int block_size, size_t shared_mem = 0) {
    int num_blocks;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kernel,
                                                 block_size, shared_mem);
    return num_blocks;
  }

  void synchronize() {
    hipError_t err = hipSuccess;
    err = hipDeviceSynchronize();

    if (err != hipSuccess) {
      std::cerr << "hipDeviceSynchronize failed: " << hipGetErrorString(err)
                << std::endl;
    }
  }

  void synchronize(HipStream &stream) { stream.synchronize(); }

  void synchronize(HipEvent &event) { event.synchronize(); }

private:
  void initOptimalParams() {
    if constexpr (V == DeviceVersion::RDNA3) {
      optimal_block_size_ = 256;
      optimal_grid_multiplier_ = 4;
    } else {
      optimal_block_size_ = 256;
      optimal_grid_multiplier_ = 2;
    }

    // int min_grid_size;

    // hipError_t err = hipSuccess;

    // err = hipOccupancyMaxPotentialBlockSize(
    //     &min_grid_size, &optimal_block_size_, (const void *)nullptr, 0, 0);

    // if (err != hipSuccess) {
    //   std::cerr << "hipOccupancyMaxPotentialBlockSize failed: "
    //             << hipGetErrorString(err) << std::endl;
    // }
  }

  int device_;
  hipDeviceProp_t props_;
  int optimal_block_size_ = 256;
  int optimal_grid_multiplier_ = 4;
};

} // namespace hipster

#endif // HIPSTER_HIPSTER_HPP