#include <iostream>
#include <vector>

// 1. Агенты: инициализация HSA и обнаружение CPU/GPU
#include <hipster.hpp>
#include <unifired/agent_cpu.hpp>
#include <unifired/agent_gpu.hpp>

// 2. Пулы памяти: запрос возможностей и выделение памяти HSA
#include <unifired/fine_grained_pool.hpp>
#include <unifired/pool_cpu.hpp>
#include <unifired/pool_gpu.hpp>

// 3. DMA и асинхронные трансферы: zero-copy буферы и io_uring
#include <unifired/async_dma_transfer.hpp>
#include <unifired/dma_buffer.hpp>
// 4. Высокоуровневое управление памятью: аналоги std::vector для
// Host/Device/Managed
#include <dataframe.hpp>

#include <chrono>
#include <random>
#include <vector>

#include <hip/hip_runtime.h>

__global__ void simple_multiply_kernel(int *data) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    data[0] *= 2;
  }
}

using namespace hipster;

void benchmark_hipster_real_world(const GpuAgent &gpu_agent,
                                  GpuPool &gpu_pool) {
  std::cout << "\n=== 7. Benchmark: Реальная работа AsyncDmaTransfer ===\n";

  AsyncDmaTransferAuto transfer(gpu_agent, gpu_pool);
  size_t buf_size = 64 * 1024 * 1024; // 64 МБ
  if (!transfer.prepare(buf_size)) {
    std::cerr << "Не удалось подготовить буфер!\n";
    return;
  }

  constexpr size_t NUM_UPDATES = 500000;
  constexpr size_t EDGE_SIZE = 64;

  std::vector<std::vector<uint8_t>> payloads(
      NUM_UPDATES, std::vector<uint8_t>(EDGE_SIZE, 0xAA));
  std::vector<size_t> random_offsets(NUM_UPDATES);
  std::mt19937 g(42);
  std::uniform_int_distribution<size_t> dist(0, (buf_size / EDGE_SIZE) - 1);
  for (size_t i = 0; i < NUM_UPDATES; ++i) {
    random_offsets[i] = dist(g) * EDGE_SIZE;
  }

  // --- СЦЕНАРИЙ А: Имитация СТАРОГО поведения (flush на каждой операции) ---
  // Мы делаем это вручную, чтобы показать, от чего мы избавились
  auto start_old = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < NUM_UPDATES; ++i) {
    char *dst =
        static_cast<char *>(transfer.buffer().cpu()) + random_offsets[i];
    std::memcpy(dst, payloads[i].data(), EDGE_SIZE);
    // Имитация старого write_async:
    CacheFlush<FlushPolicyAuto>::flush(dst, EDGE_SIZE);
  }

  auto end_old = std::chrono::high_resolution_clock::now();
  double time_old_ms =
      std::chrono::duration<double, std::milli>(end_old - start_old).count();
  double mops_old = (NUM_UPDATES / (time_old_ms / 1000.0)) / 1e6;

  // --- СЦЕНАРИЙ Б: НОВОЕ поведение библиотеки (Dirty Tracking) ---
  auto start_new = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < NUM_UPDATES; ++i) {
    // Вызываем реальный быстрый метод библиотеки
    transfer.write_async(payloads[i].data(), random_offsets[i], EDGE_SIZE);
  }

  transfer.flushDirtyRange();
  auto end_new = std::chrono::high_resolution_clock::now();
  double time_new_ms =
      std::chrono::duration<double, std::milli>(end_new - start_new).count();
  double mops_new = (NUM_UPDATES / (time_new_ms / 1000.0)) / 1e6;

  std::cout << std::left << std::setw(40) << "Старый подход (flush per op)"
            << " | Time: " << std::setw(8) << std::fixed << std::setprecision(2)
            << time_old_ms << " ms " << " | Speed: " << std::setw(8)
            << std::setprecision(3) << mops_old << " MOPS\n";

  std::cout << std::left << std::setw(40) << "Новый подход (Dirty Tracking)"
            << " | Time: " << std::setw(8) << std::fixed << std::setprecision(2)
            << time_new_ms << " ms " << " | Speed: " << std::setw(8)
            << std::setprecision(3) << mops_new << " MOPS\n";

  std::cout << "Ускорение: " << std::fixed << std::setprecision(2)
            << (time_old_ms / time_new_ms) << "x\n";
}

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

    // benchmark_hipster_real_world(gpu_agent, gpu_pool);

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
      validateZeroCopyPointer(dma_buf.cpu(), dma_buf.gpu(), dma_buf.size());
      std::cout << "  File Descriptor: " << dma_buf.fd() << "\n\n";

      // =========================================================
      // ДЕМОНСТРАЦИЯ ZERO-COPY ВЫЧИСЛЕНИЙ
      // =========================================================

      // 1. CPU инициализирует данные
      int *cpu_data = static_cast<int *>(dma_buf.cpu());
      cpu_data[0] = 42;
      std::cout << "[CPU] Исходное значение в буфере: " << cpu_data[0] << "\n";

      // 2. Подготовка к передаче управления GPU
      // В вашей библиотеке это делает flush_dirty_range() + buf_.sync(...)
      // Для простоты здесь мы полагаемся на то, что запись в mmap-память
      // на APU обычно видна сразу, но синхронизация DMA всё равно хороша.
      dma_buf.sync(DMA_BUF_SYNC_READ | DMA_BUF_SYNC_START);

      // 3. ЗАПУСК ВЫЧИСЛЕНИЙ НА GPU
      // Ключевой момент: мы передаем dma_buf.gpu(), который равен dma_buf.cpu()
      // GPU будет писать прямо в ту же физическую память, которую читает CPU.
      int *gpu_data_ptr = static_cast<int *>(dma_buf.gpu());

      std::cout << "[GPU] Запуск ядра simple_multiply_kernel...\n";

      // Подготовка аргументов для запуска ядра (массив указателей на аргументы)
      void *kernel_args[] = {&gpu_data_ptr};

      // Используем стабильный C-API для запуска, который работает в .cpp файлах
      hipError_t err = hipLaunchKernel(
          reinterpret_cast<const void *>(simple_multiply_kernel), dim3(1),
          dim3(1),     // gridDim, blockDim
          kernel_args, // аргументы ядра
          0,           // shared memory (байт)
          0            // stream (0 = default stream)
      );

      if (err != hipSuccess) {
        std::cerr << "[GPU] Ошибка запуска ядра: " << hipGetErrorString(err)
                  << "\n";
      }

      // 4. Ожидаем завершения работы GPU
      err = hipDeviceSynchronize();
      if (err != hipSuccess) {
        std::cerr << "[GPU] Ошибка синхронизации: " << hipGetErrorString(err)
                  << "\n";
      }

      // 5. CPU читает результат (Zero-Copy в действии!)
      // Нам не нужно делать memcpy или map/unmap. Мы просто читаем память.
      dma_buf.sync(DMA_BUF_SYNC_END);

      std::cout << "[CPU] Значение после работы GPU: " << cpu_data[0] << "\n";

      if (cpu_data[0] == 84) {
        std::cout << "УСПЕХ: GPU успешно модифицировал данные в Zero-Copy "
                     "буфере!\n";
      } else {
        std::cout << "ОШИБКА: Ожидалось 84, но получено " << cpu_data[0]
                  << "\n";
      }
      std::cout
          << "=========================================================\n\n";

    } else {
      std::cout << "Не удалось создать DMA Buffer (возможно, нет поддержки "
                   "ядра или прав).\n";
    }
    // =========================================================
    // 4. АСИНХРОННЫЙ DMA ТРАНСФЕР (AsyncDmaTransfer)
    // Использование io_uring для асинхронной записи + HSA copy
    // =========================================================
    // std::cout << "\n=== 4. Async DMA Transfer ===\n";

    // AsyncDmaTransferAuto async_dma(gpu_agent, gpu_pool);
    // if (async_dma.prepare(dma_size)) {
    //   std::cout << "AsyncDmaTransfer подготовлен.\n";
    //   std::cout << "  Fixed buffer registered: " << (async_dma.fixed_buffer()
    //   ? "yes" : "no") << "\n";

    //   // Пример асинхронной записи (упрощенный)
    //   int test_val = 123;
    //   auto cb = [](int res) {
    //     std::cout << "  DMA Callback вызван: result = " << res << "\n";
    //   };

    //   uint64_t op_id =
    //       async_dma.write_async(&test_val, 0, sizeof(test_val), cb);
    //   async_dma.wait(op_id); // Ожидание завершения конкретной
    //   // операции
    // } else {
    //   std::cout << "Не удалось подготовить AsyncDmaTransfer (проверьте права
    //                "
    //                "io_uring).\n";
    // }

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

    std::cout << "\n=== 6. Benchmark: Симуляция разреженного графа (Random "
                 "Access) ===\n";

    // Создаем буфер, который ЗАВЕДОМО не влезает в L3 кэш вашего Ryzen 8845HS
    // (L3 там ~16 МБ) 32 МБ гарантируют промахи L3 при случайном доступе
    constexpr size_t GRAPH_SIZE_BYTES = 32 * 1024 * 1024;
    constexpr size_t ELEMENTS = GRAPH_SIZE_BYTES / sizeof(uint64_t);

    std::vector<uint64_t> mock_graph(ELEMENTS, 1);

    // Генерируем случайные индексы (симуляция нерегулярного обхода рёбер
    std::vector<size_t> random_indices(ELEMENTS);
    for (size_t i = 0; i < ELEMENTS; ++i)
      random_indices[i] = i;
    std::mt19937 g(42); // Фиксированный seed для воспроизводимости
    std::shuffle(random_indices.begin(), random_indices.end(), g);

    std::cout << "Запуск случайного доступа к "
              << (GRAPH_SIZE_BYTES / 1024 / 1024) << " МБ данных...\n";

    auto start = std::chrono::high_resolution_clock::now();
    volatile uint64_t sink =
        0; // volatile предотвращает оптимизацию цикла компилятором

    // Симуляция обхода графа: прыгаем по случайным адресам
    for (size_t i = 0; i < ELEMENTS; ++i) {
      sink += mock_graph[random_indices[i]];
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    double bandwidth =
        (GRAPH_SIZE_BYTES / (1024.0 * 1024.0)) / (duration_ms / 1000.0);

    std::cout << "Время выполнения: " << duration_ms << " мс\n";
    std::cout << "Эффективная пропускная способность: " << bandwidth
              << " МБ/с\n";

  } catch (const std::exception &e) {
    std::cerr << "Поймано исключение: " << e.what() << "\n";
    return 1;
  }

  return 0;
}