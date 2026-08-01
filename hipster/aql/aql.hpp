#ifndef HIPSTER_AQL_AQL_HPP
#define HIPSTER_AQL_AQL_HPP

#include <cstdint>
#include <cstring>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

namespace hipster {

struct __attribute__((packed)) alignas(16) graph_weight_sum_kernargs_t {
  const uint32_t *row_ptr;
  const uint32_t *col_idx;
  const float *weights;
  float *out_sums;
  uint32_t num_vertices;
  uint32_t _reserved;
};

inline hsa_kernel_dispatch_packet_t
create_graph_kernel_packet(uint64_t kernel_object, void *kernarg_address,
                           uint32_t num_vertices, uint16_t block_size_x,
                           hsa_signal_t completion_signal) {
  hsa_kernel_dispatch_packet_t packet;
  std::memset(&packet, 0, sizeof(hsa_kernel_dispatch_packet_t));

  // packet.setup = 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  packet.setup = 0;
  packet.workgroup_size_x = block_size_x;
  packet.workgroup_size_y = 1;
  packet.workgroup_size_z = 1;

  packet.grid_size_x = num_vertices;
  packet.grid_size_y = 1;
  packet.grid_size_z = 1;

  packet.kernel_object = kernel_object;
  packet.kernarg_address = kernarg_address;
  packet.completion_signal = completion_signal;

  return packet;
}

inline void submit_aql_packet(hsa_queue_t *queue,
                              const hsa_kernel_dispatch_packet_t &packet) {
  uint64_t index = hsa_queue_add_write_index_relaxed(queue, 1);
  uint32_t queue_mask = queue->size - 1;

  auto *queue_base =
      reinterpret_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address);
  auto *packet_slot = &queue_base[index & queue_mask];

  std::memcpy(&packet_slot->workgroup_size_x, &packet.workgroup_size_x,
              sizeof(hsa_kernel_dispatch_packet_t) - sizeof(uint16_t));
  packet_slot->setup = packet.setup;
  packet_slot->kernel_object = packet.kernel_object;
  packet_slot->kernarg_address = packet.kernarg_address;
  packet_slot->completion_signal = packet.completion_signal;

  uint16_t header =
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
      (1 << HSA_PACKET_HEADER_BARRIER) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
      (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

  __atomic_store_n(reinterpret_cast<uint16_t *>(packet_slot), header,
                   __ATOMIC_RELEASE);

  hsa_signal_store_relaxed(queue->doorbell_signal, index);
}

} // namespace hipster

#endif // HIPSTER_AQL_AQL_HPP
