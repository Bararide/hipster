#include <cstdint>
#include <hip/hip_runtime.h>

extern "C" __global__ void graph_weight_sum_kernel(const uint32_t *row_ptr,
                                                   const uint32_t *col_idx,
                                                   const float *weights,
                                                   float *out_sums,
                                                   uint32_t num_vertices) {
  uint32_t v = blockIdx.x * blockDim.x + threadIdx.x;

  if (v < num_vertices) {
    uint32_t start = row_ptr[v];
    uint32_t end = row_ptr[v + 1];

    float local_sum = 0.0f;

    for (uint32_t e = start; e < end; ++e) {
      local_sum += weights[e];
    }

    out_sums[v] = local_sum;
  }
}