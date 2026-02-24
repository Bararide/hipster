#ifndef HIPSTER_ALIES_HPP
#define HIPSTER_ALIES_HPP

#include <chrono>
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <rocblas/rocblas.h>
#include <rocrand/rocrand.h>
#include <thread>

namespace hipster {

namespace Clock = std::chrono;

#define IS_RDNA3(props) (props.gcnArch == 1100 || props.gcnArch == 1103)
#define IS_RDNA2(props) (props.gcnArch == 1030 || props.gcnArch == 1031)
#define IS_APU(props) (props.integrated && props.maxThreadsPerBlock <= 1024)

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

enum class DeviceVersion { RDNA3, RDNA2, UNKNOWN };

} // namespace hipster

#endif // HIPSTER_ALIES_HPP