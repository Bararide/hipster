#include <chrono>
#include <iomanip>
#include <random>
#include <vector>

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <hipster.hpp>
#include <unifired/agent_cpu.hpp>
#include <unifired/agent_gpu.hpp>
#include <unifired/async_dma_transfer.hpp>
#include <unifired/dma_buffer.hpp>
#include <unifired/fine_grained_pool.hpp>
#include <unifired/pool_cpu.hpp>
#include <unifired/pool_gpu.hpp>

__global__ void graph_weight_sum_kernel(const uint32_t *row_ptr,
                                        const uint32_t *col_idx,
                                        const float *weights, float *out_sums,
                                        uint32_t num_vertices);

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

    constexpr uint32_t NUM_VERTICES = 100000;
    constexpr uint32_t NUM_EDGES = 500000;

    size_t row_ptr_size = (NUM_VERTICES + 1) * sizeof(uint32_t);
    size_t col_idx_size = NUM_EDGES * sizeof(uint32_t);
    size_t weights_size = NUM_EDGES * sizeof(float);
    size_t out_sums_size = NUM_VERTICES * sizeof(float);

    size_t total_graph_size =
        row_ptr_size + col_idx_size + weights_size + out_sums_size;

    AsyncDmaTransferAuto transfer(gpu_agent, gpu_pool);
    if (!transfer.prepare(total_graph_size, "graph-csr-buffer")) {
      logger->error("error AsyncDmaTransfer");
      return 1;
    }
    logger->info("Zero-Copy buffer: {} ({})", total_graph_size,
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

    logger->info("graph_weight_sum_kernel (Grid: {}x{}x{}, Block: {}x{}x{})",
                 config.grid_dim.x, config.grid_dim.y, config.grid_dim.z,
                 config.block_dim.x, config.block_dim.y, config.block_dim.z);

    auto start_time = std::chrono::high_resolution_clock::now();

    hip.launchKernel(graph_weight_sum_kernel, config, stream,
                     static_cast<uint32_t *>(transfer.buffer().cpu()) +
                         (offset_row_ptr / sizeof(uint32_t)), // row_ptr
                     static_cast<uint32_t *>(transfer.buffer().cpu()) +
                         (offset_col_idx / sizeof(uint32_t)), // col_idx
                     static_cast<float *>(transfer.buffer().cpu()) +
                         (offset_weights / sizeof(float)), // weights
                     static_cast<float *>(transfer.buffer().cpu()) +
                         (offset_out_sums / sizeof(float)), // out_sums
                     NUM_VERTICES);

    hip.synchronize(stream);
    auto end_time = std::chrono::high_resolution_clock::now();

    double duration_ms =
        std::chrono::duration<double, std::milli>(end_time - start_time)
            .count();
    logger->info("kernel comleted {:.4f} ms", duration_ms);

    std::vector<float> h_out_sums(NUM_VERTICES, 0.0f);
    transfer.buffer().read(offset_out_sums, h_out_sums.data(),
                           h_out_sums.size());

    bool success = true;
    for (uint32_t v = 0; v < std::min(NUM_VERTICES, 10u); ++v) {
      uint32_t degree = h_row_ptr[v + 1] - h_row_ptr[v];
      float expected_sum = degree * 1.5f;
      if (std::abs(h_out_sums[v] - expected_sum) > 0.01f) {
        success = false;
        logger->error("vertex {}: {:.2f}, {:.2f}", v, expected_sum,
                      h_out_sums[v]);
        break;
      }
    }

    if (success) {
      logger->info("success");
    }

  } catch (const std::exception &e) {
    logger->error("{}", e.what());
    return 1;
  }

  return 0;
}