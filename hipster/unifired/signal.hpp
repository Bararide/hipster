#ifndef HIPSTER_UNIFIRED_SIGNAL_HPP
#define HIPSTER_UNIFIRED_SIGNAL_HPP

namespace hipster {

class HsaSignal {
public:
  explicit HsaSignal(hsa_signal_value_t initial_value = 1,
                     uint32_t num_consumers = 0)
      : signal{} {
    hsa_status_t status =
        hsa_signal_create(initial_value, num_consumers, nullptr, &signal);
    if (status != HSA_STATUS_SUCCESS) {
      const char *err_string = nullptr;
      hsa_status_string(status, &err_string);
      throw std::runtime_error(std::string("Failed to create HSA signal: ") +
                               (err_string ? err_string : "Unknown error"));
    }
  }

  ~HsaSignal() {
    if (signal.handle != 0) {
      hsa_signal_destroy(signal);
    }
  }

  HsaSignal(const HsaSignal &) = delete;
  HsaSignal &operator=(const HsaSignal &) = delete;

  HsaSignal(HsaSignal &&other) noexcept : signal(other.signal) {
    other.signal.handle = 0;
  }

  [[nodiscard]] hsa_signal_t get() const noexcept { return signal; }

private:
  hsa_signal_t signal;
};

template <typename Func, typename... Args>
static inline void checkHsaStatus(Func func, Args &&...args) {
  hsa_status_t status = func(std::forward<Args>(args)...);
  if (status != HSA_STATUS_SUCCESS) {
    const char *err_string = nullptr;
    hsa_status_string(status, &err_string);
    throw std::runtime_error(std::string("HSA API Error: ") +
                             (err_string ? err_string : "Unknown error"));
  }
}

} // namespace hipster

#endif // HIPSTER_UNIFIRED_SIGNAL_HPP