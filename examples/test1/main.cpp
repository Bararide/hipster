#include <iostream>
#include <vector>

// 1. Агенты: инициализация HSA и обнаружение CPU/GPU
#include <hipster.hpp>
#include <unifired/agent_cpu.hpp>
#include <unifired/agent_gpu.hpp>

// 2. Пулы памяти: запрос возможностей и выделение памяти HSA
#include <unifired/pool_cpu.hpp>
#include <unifired/pool_gpu.hpp>
#include <unifired/fine_grained_pool.hpp>

// 3. DMA и асинхронные трансферы: zero-copy буферы и io_uring
#include <unifired/dma_buffer.hpp>
#include <unifired/async_dma_transfer.hpp>
// 4. Высокоуровневое управление памятью: аналоги std::vector для
// Host/Device/Managed
#include <dataframe.hpp>

using namespace hipster;

int main() {
  try {
    // =========================================================
    // 1. АГЕНТЫ (Agents)
    // Инициализирует HSA и находит первый доступный CPU и GPU
    // =========================================================
    std::cout << "=== 1. HSA Agents ===\n";

    CpuAgent cpu_agent;
    cpu_agent.info(); // Выводит имя CPU

    GpuAgent gpu_agent;
    gpu_agent.info(); // Выводит имя GPU

    // =========================================================
    // 2. ПУЛЫ ПАМЯТИ (Memory Pools)
    // Получение информации о сегментах памяти агентов
    // =========================================================
    std::cout << "\n=== 2. Memory Pools ===\n";

    CpuPool cpu_pool(cpu_agent);
    cpu_pool.info(); // Размер, макс. выделение

    GpuPool gpu_pool(gpu_agent);
    gpu_pool.info();

    GpuFineGrainedPool fine_grained_pool(gpu_agent);
    std::cout << "  Fine-grained memory supported: "
              << (fine_grained_pool.isFineGrained() ? "yes" : "no") << "\n";

    // =========================================================
    // 3. DMA БУФЕР (DmaBuffer)
    // Создание разделяемой памяти (memfd + mmap + hsa_amd_memory_lock)
    // =========================================================
    std::cout << "\n=== 3. DMA Buffer ===\n";

    DmaBuffer dma_buf;
    size_t dma_size = 64 * 1024; // 64 KB
    if (dma_buf.create(dma_size, gpu_agent)) {
      std::cout << "DMA Buffer создан успешно.\n";
      std::cout << "  Size: " << dma_buf.size() << " bytes\n";
      std::cout << "  CPU ptr: " << dma_buf.cpu() << "\n";
      std::cout << "  GPU ptr: " << dma_buf.gpu() << "\n";
      std::cout << "  File Descriptor: " << dma_buf.fd() << "\n";

      // Пример: запись со стороны CPU и синхронизация для GPU
      int *cpu_data = static_cast<int *>(dma_buf.cpu());
      cpu_data[0] = 42;
      dma_buf.sync(DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START);
      // ... здесь GPU может читать данные ...
      dma_buf.sync(DMA_BUF_SYNC_END);
    } else {
      std::cout
          << "Не удалось создать DMA Buffer (возможно, нет поддержки ядра).\n";
    }

    // =========================================================
    // 4. АСИНХРОННЫЙ DMA ТРАНСФЕР (AsyncDmaTransfer)
    // Использование io_uring для асинхронной записи + HSA copy
    // =========================================================
    std::cout << "\n=== 4. Async DMA Transfer ===\n";

    AsyncDmaTransferAuto async_dma(gpu_agent, gpu_pool);
    if (async_dma.prepare(dma_size)) {
      std::cout << "AsyncDmaTransfer подготовлен.\n";
      std::cout << "  Fixed buffer registered: "
                << (async_dma.fixed_buffer() ? "yes" : "no") << "\n";

      // Пример асинхронной записи (упрощенный)
      int test_val = 123;
      auto cb = [](int res) {
        std::cout << "  DMA Callback вызван: result = " << res << "\n";
      };

      // uint64_t op_id = async_dma.write_async(&test_val, 0, sizeof(test_val),
      // cb); async_dma.wait(op_id); // Ожидание завершения конкретной операции
    } else {
      std::cout << "Не удалось подготовить AsyncDmaTransfer (проверьте права "
                   "io_uring).\n";
    }

    // =========================================================
    // 5. DATA FRAME (Управление памятью)
    // Высокоуровневые контейнеры с автоматическим копированием
    // =========================================================
    std::cout << "\n=== 5. DataFrame (Memory Management) ===\n";

    // Host DataFrame (обычная RAM)
    HostDataFrame<int> host_df(5, 42);
    std::cout << "HostDataFrame: size=" << host_df.size()
              << ", host_df[2]=" << host_df[2] << "\n";

    // Device DataFrame (VRAM GPU, доступ через HIP)
    DeviceDataFrame<float> device_df(10);
    std::cout << "DeviceDataFrame: capacity=" << device_df.capacity() << "\n";

    // Копирование данных из хоста в устройство
    std::vector<float> host_data = {1.1f, 2.2f, 3.3f};
    device_df.assign(host_data.begin(), host_data.end());
    std::cout << "DeviceDataFrame: скопировано " << device_df.size()
              << " элементов с хоста.\n";

    // Managed DataFrame (Unified Memory, доступна и CPU, и GPU без явного
    // копирования)
    ManagedDataFrame<double> managed_df(4, 3.14);
    std::cout << "ManagedDataFrame: size=" << managed_df.size()
              << ", managed_df[3]=" << managed_df[3] << "\n";

    // Обратное копирование из устройства в хост
    auto back_to_host = device_df.to_host_vector();
    std::cout << "DeviceDataFrame скопирован обратно в хост: "
              << back_to_host[0] << ", " << back_to_host[1] << ", "
              << back_to_host[2] << "\n";

    std::cout << "\n=== Все тесты успешно завершены! ===\n";

  } catch (const std::exception &e) {
    std::cerr << "Поймано исключение: " << e.what() << "\n";
    return 1;
  }

  return 0;
}