#ifndef HIPSTER_UTILS_HPP
#define HIPSTER_UTILS_HPP

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

namespace hipster {

inline int getDeviceId(int device_id) {
    hipError_t err = hipInit(0);

    if (err != hipSuccess) {
      std::cerr << "HIP init failed: " << hipGetErrorString(err) << std::endl;
      return -1;
    }

    int count = 0;
    err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
      std::cerr << "hipGetDeviceCount failed: " << hipGetErrorString(err)
                << std::endl;
      return -1;
    }

    if (count == 0) {
      return -1;
    }

    if (device_id >= count) {
      device_id = 0;
    }

    err = hipSetDevice(device_id);
    if (err != hipSuccess) {
      std::cerr << "hipSetDevice failed: " << hipGetErrorString(err)
                << std::endl;
      return -1;
    }

    return device_id;
}

}

#endif // HIPSTER_UTILS_HPP