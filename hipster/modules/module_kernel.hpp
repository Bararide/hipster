#ifndef HIPSTER_MODELS_MODULE_KERNEL_HPP
#define HIPSTER_MODELS_MODULE_KERNEL_HPP

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipster {

class HipModuleKernel {
public:
  hipModule_t module = nullptr;
  hipFunction_t function = nullptr;

  HipModuleKernel() = delete;

  HipModuleKernel(const unsigned char *data, const std::string &kernel_name) {
    hipError_t err = hipModuleLoadData(&module, data);
    if (err != hipSuccess) {
      throw std::runtime_error("hipModuleLoadData failed: " +
                               std::string(hipGetErrorString(err)));
    }

    err = hipModuleGetFunction(&function, module, kernel_name.c_str());
    if (err != hipSuccess) {
      (void)hipModuleUnload(module);
      module = nullptr;
      throw std::runtime_error("Function not found: " +
                               std::string(hipGetErrorString(err)));
    }
  }

  HipModuleKernel(const HipModuleKernel &) = delete;
  HipModuleKernel &operator=(const HipModuleKernel &) = delete;

  HipModuleKernel(HipModuleKernel &&other) noexcept
      : module(other.module), function(other.function) {
    other.module = nullptr;
    other.function = nullptr;
  }

  HipModuleKernel &operator=(HipModuleKernel &&other) noexcept {
    if (this != &other) {
      cleanup();
      module = other.module;
      function = other.function;
      other.module = nullptr;
      other.function = nullptr;
    }
    return *this;
  }

  ~HipModuleKernel() { cleanup(); }

  [[nodiscard]] bool isValid() const noexcept {
    return module != nullptr && function != nullptr;
  }

  template <typename... Args>
  void launch(dim3 grid, dim3 block, size_t shared, hipStream_t stream,
              Args... args) {
    if (!isValid()) {
      spdlog::critical(
          "Cannot launch: HipModuleKernel is invalid (null function)");
      return;
    }

    void *kernel_args[] = {(void *)&args...};

    hipError_t err = hipModuleLaunchKernel(function, grid.x, grid.y, grid.z,
                                           block.x, block.y, block.z, shared,
                                           stream, kernel_args, nullptr);
    if (err != hipSuccess) {
      spdlog::critical("Fail in hip module launch kernel: {}",
                       hipGetErrorString(err));
    }
  }

private:
  void cleanup() noexcept {
    if (module) {
      hipError_t err = hipModuleUnload(module);
      if (err != hipSuccess) {
        spdlog::error("Failed to unload HIP module: {}",
                      hipGetErrorString(err));
      }
      module = nullptr;
      function = nullptr;
    }
  }
};

} // namespace hipster

#endif // HIPSTER_MODELS_MODULE_KERNEL_HPP