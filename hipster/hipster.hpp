#ifndef HIPSTER_HIPSTER_HPP
#define HIPSTER_HIPSTER_HPP

#include "memory.hpp"

namespace hipster {

template <DeviceVersion V = DeviceVersion::RDNA3> class Hipster {
public:
  Hipster(const int device_n) : props_{} {
    hipError_t err = hipGetDeviceProperties(&props_, device_n);
    if (err != hipSuccess) {
      throw std::runtime_error(hipGetErrorString(err));
    }
  }

private:
  hipDeviceProp_t props_;
};

} // namespace hipster

#endif // HIPSTER_HIPSTER_HPP