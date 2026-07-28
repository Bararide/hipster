#ifndef HIPSTER_UNIFIRED_UTILS_HPP
#define HIPSTER_UNIFIRED_UTILS_HPP

#include <hsa/hsa.h>
#include <iostream>

namespace hipster {

inline void validateZeroCopyPointer(const void *cpu_ptr, const void *gpu_ptr,
                                    size_t size) {
  if (cpu_ptr == gpu_ptr) {
    std::cout << "[OK] (" << cpu_ptr << ").\n";
  } else {
    std::cout << "[WARN] (CPU: " << cpu_ptr << ", GPU: " << gpu_ptr << ").\n";
  }

  hsa_amd_pointer_info_t info;
  info.size = sizeof(hsa_amd_pointer_info_t);

  hsa_status_t status =
      hsa_amd_pointer_info(cpu_ptr, &info, nullptr, nullptr, nullptr);
  if (status == HSA_STATUS_SUCCESS &&
      info.type != HSA_EXT_POINTER_TYPE_UNKNOWN) {
    switch (info.type) {
    case HSA_EXT_POINTER_TYPE_HSA:
      std::cout << "HSA Allocation\n";
      break;
    case HSA_EXT_POINTER_TYPE_LOCKED:
      std::cout << "Locked Host Memory\n";
      break;
    default:
      std::cout << "Other\n";
    }
  } else {
    std::cout << "[INFO] HSA Runtime dont watch this memfd-region \n";
  }
}

static inline void checkStatus(hsa_status_t status, const std::string &msg) {
  if (status != HSA_STATUS_SUCCESS) {
    throw std::runtime_error(msg + " (HSA Status: " + std::to_string(status) +
                             ")");
  }
}

} // namespace hipster

#endif // HIPSTER_UNIFIRED_UTILS_HPP