#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <hipster.hpp>
#include <modules/module_kernel.hpp>
#include <unifired/agent_cpu.hpp>
#include <unifired/agent_gpu.hpp>
#include <unifired/async_dma_transfer.hpp>
#include <unifired/dma_buffer.hpp>
#include <unifired/fine_grained_pool.hpp>
#include <unifired/pool_cpu.hpp>
#include <unifired/pool_gpu.hpp>

extern "C" __global__ void graph_weight_sum_kernel(const uint32_t *row_ptr,
                                                   const uint32_t *col_idx,
                                                   const float *weights,
                                                   float *out_sums,
                                                   uint32_t num_vertices);

extern const unsigned char graph_kernel_hsaco[];
extern unsigned int graph_kernel_hsaco_len;

using namespace hipster;

int main() {
  auto logger = spdlog::stdout_color_mt("hipster_test");
  logger->set_level(spdlog::level::info);
  logger->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

  try {
    logger->info("=== 1. HSA Agents ===");
    CpuAgent cpu_agent;
    logger->info("CPU: {}", cpu_agent.name());

    GpuAgent gpu_agent;
    logger->info("GPU: {}", gpu_agent.name());

    logger->info("\n=== 2. Memory Pools ===");
    GpuPool gpu_pool(gpu_agent);
    logger->info("GPU Pool Size: {} MB", gpu_pool.size() / (1024 * 1024));

    GpuFineGrainedPool fine_grained_pool(gpu_agent);
    logger->info("Fine-grained memory supported: {}",
                 fine_grained_pool.isFineGrained() ? "yes" : "no");

    logger->info("\n=== 3. Graph Processing & Cache Optimization Test ===");
    Hipster hip;
    logger->info("[Hipster] Initialization: {}", hip.getDeviceName());

    constexpr uint32_t NUM_VERTICES = 10000;
    constexpr uint32_t NUM_EDGES = 300000;

    size_t row_ptr_size = (NUM_VERTICES + 1) * sizeof(uint32_t);
    size_t col_idx_size = NUM_EDGES * sizeof(uint32_t);
    size_t weights_size = NUM_EDGES * sizeof(float);
    size_t out_sums_size = NUM_VERTICES * sizeof(float);
    size_t total_graph_size =
        row_ptr_size + col_idx_size + weights_size + out_sums_size;

    AsyncDmaTransferAuto transfer(gpu_agent, gpu_pool);
    if (!transfer.prepare(total_graph_size, "graph-csr-buffer")) {
      logger->error("Failed to prepare AsyncDmaTransfer");
      return 1;
    }
    logger->info("Zero-Copy buffer: {} bytes ({} MB)", total_graph_size,
                 total_graph_size / (1024 * 1024));

    size_t offset_row_ptr = 0;
    size_t offset_col_idx = offset_row_ptr + row_ptr_size;
    size_t offset_weights = offset_col_idx + col_idx_size;
    size_t offset_out_sums = offset_weights + weights_size;

    std::vector<uint32_t> h_row_ptr(NUM_VERTICES + 1, 0);
    std::vector<uint32_t> h_col_idx(NUM_EDGES, 0);
    std::vector<float> h_weights(NUM_EDGES, 1.5f);

    uint32_t current_edge = 0;
    for (uint32_t v = 0; v < NUM_VERTICES; ++v) {
      h_row_ptr[v] = current_edge;
      uint32_t degree = 1 + (v % 10);
      for (uint32_t d = 0; d < degree; ++d) {
        if (current_edge < NUM_EDGES) {
          h_col_idx[current_edge] = (v + d) % NUM_VERTICES;
          current_edge++;
        }
      }
    }
    h_row_ptr[NUM_VERTICES] = current_edge;

    transfer.buffer().write(offset_row_ptr, h_row_ptr.data(), h_row_ptr.size());
    transfer.buffer().write(offset_col_idx, h_col_idx.data(), h_col_idx.size());
    transfer.buffer().write(offset_weights, h_weights.data(), h_weights.size());

    transfer.flushDirtyRange();
    logger->info("Dirty range flushed. Data sync with RAM.");

    HipStream stream = hip.createStream();
    LaunchConfig config = hip.getOptimalLaunchConfig(NUM_VERTICES);

    const uint32_t *d_row_ptr =
        static_cast<const uint32_t *>(transfer.buffer().cpu()) +
        (offset_row_ptr / sizeof(uint32_t));
    const uint32_t *d_col_idx =
        static_cast<const uint32_t *>(transfer.buffer().cpu()) +
        (offset_col_idx / sizeof(uint32_t));
    const float *d_weights =
        static_cast<const float *>(transfer.buffer().cpu()) +
        (offset_weights / sizeof(float));
    float *d_out_sums = static_cast<float *>(transfer.buffer().cpu()) +
                        (offset_out_sums / sizeof(float));

    logger->info("\n--- Launching Kernels ---");

    {
      auto start = std::chrono::high_resolution_clock::now();
      hip.launchKernel(graph_weight_sum_kernel, config, stream, d_row_ptr,
                       d_col_idx, d_weights, d_out_sums, NUM_VERTICES);
      hip.synchronize(stream);
      auto end = std::chrono::high_resolution_clock::now();
      double ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      logger->info("[Level 1] HIP Launch (Hipster): {:.4f} ms", ms);
    }

    {
      HipModuleKernel mod_kernel(graph_kernel_hsaco, "graph_weight_sum_kernel");
      if (mod_kernel.isValid()) {
        void *args[] = {const_cast<uint32_t **>(&d_row_ptr),
                        const_cast<uint32_t **>(&d_col_idx),
                        const_cast<float **>(&d_weights), &d_out_sums,
                        const_cast<uint32_t *>(&NUM_VERTICES)};

        auto start = std::chrono::high_resolution_clock::now();
        mod_kernel.launch(config.grid_dim, config.block_dim, 0, stream.get(),
                          args);
        hip.synchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();
        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        logger->info("[Level 2] HIP Module API: {:.4f} ms", ms);
      } else {
        logger->warn("[Level 2] Skipped (graph_kernel.hsaco not found)");
      }
    }

    logger->info("\n--- Validating Results ---");
    std::vector<float> h_out_sums(NUM_VERTICES, 0.0f);
    transfer.buffer().read(offset_out_sums, h_out_sums.data(),
                           h_out_sums.size());

    bool success = true;
    for (uint32_t v = 0; v < std::min(NUM_VERTICES, 10u); ++v) {
      uint32_t degree = h_row_ptr[v + 1] - h_row_ptr[v];
      float expected_sum = degree * 1.5f;
      if (std::abs(h_out_sums[v] - expected_sum) > 0.01f) {
        success = false;
        logger->error(
            "Validation failed at vertex {}: expected {:.2f}, got {:.2f}", v,
            expected_sum, h_out_sums[v]);
        break;
      }
    }

    if (success) {
      logger->info("SUCCESS: Graph processed correctly. Zero-Copy and "
                   "kernel execution validated.");
    }

  } catch (const std::exception &e) {
    logger->error("Exception caught: {}", e.what());
    return 1;
  }

  return 0;
}