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
#include <unifired/pool_allocator.hpp>
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

    logger->info("\n=== 3. Graph Processing & PMR Zero-Copy Test ===");
    Hipster hip;
    logger->info("[Hipster] Initialization: {}", hip.getDeviceName());

    constexpr uint32_t NUM_VERTICES = 10000;
    constexpr uint32_t NUM_EDGES = 300000;

    HsaPmrResource<PoolType::GLOBAL, GlobalMemoryProperty::ANY> pmr_resource(
        GpuAgent{}.agent());
    logger->info("PMR Resource created. Max alloc: {} MB",
                 pmr_resource.allocMaxSize() / (1024 * 1024));

    std::pmr::polymorphic_allocator<uint32_t> alloc_u32(&pmr_resource);
    std::pmr::polymorphic_allocator<float> alloc_f32(&pmr_resource);

    std::pmr::vector<uint32_t> d_row_ptr(NUM_VERTICES + 1, 0, alloc_u32);
    std::pmr::vector<uint32_t> d_col_idx(NUM_EDGES, 0, alloc_u32);
    std::pmr::vector<float> d_weights(NUM_EDGES, 1.5f, alloc_f32);
    std::pmr::vector<float> d_out_sums(NUM_VERTICES, 0.0f, alloc_f32);

    logger->info("PMR Vectors allocated directly in HSA Fine-Grained memory.");

    logger->info("CPU populating graph data directly in device memory...");
    uint32_t current_edge = 0;
    for (uint32_t v = 0; v < NUM_VERTICES; ++v) {
      d_row_ptr[v] = current_edge;
      uint32_t degree = 1 + (v % 10);
      for (uint32_t d = 0; d < degree; ++d) {
        if (current_edge < NUM_EDGES) {
          d_col_idx[current_edge] = (v + d) % NUM_VERTICES;
          d_weights[current_edge] = 1.5f;
          current_edge++;
        }
      }
    }
    d_row_ptr[NUM_VERTICES] = current_edge;

    HipStream stream = hip.createStream();
    LaunchConfig config = hip.getOptimalLaunchConfig(NUM_VERTICES);

    logger->info("\n--- Launching Kernels ---");

    {
      HipModuleKernel mod_kernel(graph_kernel_hsaco, "graph_weight_sum_kernel");
      if (mod_kernel.isValid()) {

        auto start = std::chrono::high_resolution_clock::now();
        mod_kernel.launch(config.grid_dim, config.block_dim, 0, stream.get(),
                          d_row_ptr.data(), d_col_idx.data(), d_weights.data(),
                          d_out_sums.data(), NUM_VERTICES);
        hip.synchronize(stream);
        auto end = std::chrono::high_resolution_clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        logger->info("[Level 2] HIP Module API (PMR Data): {:.4f} ms", ms);
      } else {
        logger->warn("[Level 2] Skipped (graph_kernel.hsaco not found)");
      }
    }

    logger->info("\n--- Validating Results ---");
    bool success = true;
    for (uint32_t v = 0; v < std::min(NUM_VERTICES, 10u); ++v) {
      uint32_t degree = d_row_ptr[v + 1] - d_row_ptr[v];
      float expected_sum = degree * 1.5f;
      if (std::abs(d_out_sums[v] - expected_sum) > 0.01f) {
        success = false;
        logger->error(
            "Validation failed at vertex {}: expected {:.2f}, got {:.2f}", v,
            expected_sum, d_out_sums[v]);
        break;
      }
    }

    if (success) {
      logger->info(
          "SUCCESS: Graph processed correctly using std::pmr Zero-Copy!");
    }

  } catch (const std::exception &e) {
    logger->error("Exception caught: {}", e.what());
    return 1;
  }

  return 0;
}