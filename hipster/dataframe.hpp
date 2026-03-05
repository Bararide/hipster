#ifndef HIPSTER_MEMORY_DATA_FRAME_HPP
#define HIPSTER_MEMORY_DATA_FRAME_HPP

#include "memory.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

namespace hipster {

namespace detail {

inline void hipThrowIfError(hipError_t err, const char *context) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             hipGetErrorString(err));
  }
}

template <MemoryType MT> struct backend_ops;

template <> struct backend_ops<MemoryType::Host> {
  static constexpr bool host_visible = true;

  template <typename T>
  static void copy_same(T *dst, const T *src, size_t count) {
    if (!count || dst == src) {
      return;
    }
    std::copy_n(src, count, dst);
  }

  template <typename T>
  static void copy_from_host(T *dst, const T *src, size_t count) {
    copy_same(dst, src, count);
  }

  template <typename T>
  static void copy_to_host(T *dst, const T *src, size_t count) {
    copy_same(dst, src, count);
  }

  template <typename T> static void fill(T *dst, size_t count, const T &value) {
    if (!count) {
      return;
    }
    std::fill_n(dst, count, value);
  }
};

template <>
struct backend_ops<MemoryType::Managed> : backend_ops<MemoryType::Host> {};

template <> struct backend_ops<MemoryType::Device> {
  static constexpr bool host_visible = false;

  template <typename T>
  static void copy_same(T *dst, const T *src, size_t count) {
    if (!count || dst == src) {
      return;
    }
    detail::hipThrowIfError(
        hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToDevice),
        "hipMemcpy device->device");
  }

  template <typename T>
  static void copy_from_host(T *dst, const T *src, size_t count) {
    if (!count) {
      return;
    }
    detail::hipThrowIfError(
        hipMemcpy(dst, src, count * sizeof(T), hipMemcpyHostToDevice),
        "hipMemcpy host->device");
  }

  template <typename T>
  static void copy_to_host(T *dst, const T *src, size_t count) {
    if (!count) {
      return;
    }
    detail::hipThrowIfError(
        hipMemcpy(dst, src, count * sizeof(T), hipMemcpyDeviceToHost),
        "hipMemcpy device->host");
  }

  template <typename T> static void fill(T *dst, size_t count, const T &value) {
    if (!count) {
      return;
    }
    std::vector<T> staging(count, value);
    copy_from_host(dst, staging.data(), count);
  }
};

} // namespace detail

template <typename T, MemoryType V = MemoryType::Managed> class DataFrame {
  static_assert(
      std::is_trivially_copyable_v<T>,
      "DataFrame currently supports only trivially copyable value types. "
      "Provide a specialization if you need complex objects.");

  using backend_type = detail::backend_ops<V>;

public:
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;
  using reference = T &;
  using const_reference = const T &;
  using pointer = T *;
  using const_pointer = const T *;
  using iterator = T *;
  using const_iterator = const T *;

  DataFrame() = default;

  explicit DataFrame(size_type count) { resize(count); }

  DataFrame(size_type count, const T &value) { resize(count, value); }

  DataFrame(std::initializer_list<T> init) { assign(init.begin(), init.end()); }

  DataFrame(const DataFrame &other) { copy_from(other); }

  DataFrame(DataFrame &&) noexcept = default;

  ~DataFrame() = default;

  DataFrame &operator=(const DataFrame &other) {
    if (this != &other) {
      copy_from(other);
    }
    return *this;
  }

  DataFrame &operator=(DataFrame &&) noexcept = default;

  void assign(std::initializer_list<T> init) {
    assign(init.begin(), init.end());
  }

  template <typename InputIt> void assign(InputIt first, InputIt last) {
    std::vector<T> staging(first, last);
    assign_from_host(staging.data(), staging.size());
  }

  void assign(size_type count, const T &value) {
    reserve(count);
    backend_type::fill(data_, count, value);
    size_ = count;
    sync_if_needed();
  }

  void resize(size_type new_size) {
    if (new_size > capacity_) {
      reserve(new_size);
    }
    size_ = new_size;
    sync_if_needed();
  }

  void resize(size_type new_size, const T &value) {
    size_type old_size = size_;
    if (new_size > capacity_) {
      reserve(new_size);
    }
    if (new_size > old_size) {
      backend_type::fill(data_ + old_size, new_size - old_size, value);
    }
    size_ = new_size;
    sync_if_needed();
  }

  void reserve(size_type new_capacity) {
    if (new_capacity <= capacity_) {
      return;
    }
    reallocate(new_capacity);
  }

  void shrink_to_fit() {
    if (size_ == capacity_) {
      return;
    }
    if (size_ == 0) {
      release();
      return;
    }
    reallocate(size_);
  }

  void clear() noexcept { size_ = 0; }

  void push_back(const T &value) {
    ensure_capacity_for_growth(size_ + 1);
    backend_type::copy_from_host(data_ + size_, &value, 1);
    ++size_;
    sync_if_needed();
  }

  void push_back(T &&value) { push_back(static_cast<const T &>(value)); }

  template <typename... Args> void emplace_back(Args &&...args) {
    T temp(std::forward<Args>(args)...);
    push_back(temp);
  }

  void pop_back() {
    if (size_ == 0) {
      throw std::underflow_error("DataFrame::pop_back on empty container");
    }
    --size_;
  }

  pointer data() noexcept { return data_; }
  const_pointer data() const noexcept { return data_; }

  [[nodiscard]] size_type size() const noexcept { return size_; }
  [[nodiscard]] size_type capacity() const noexcept { return capacity_; }
  [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

  iterator begin() noexcept { return data_; }
  const_iterator begin() const noexcept { return data_; }
  const_iterator cbegin() const noexcept { return data_; }

  iterator end() noexcept { return data_ + size_; }
  const_iterator end() const noexcept { return data_ + size_; }
  const_iterator cend() const noexcept { return data_ + size_; }

  template <MemoryType M = V,
            typename std::enable_if_t<M != MemoryType::Device, int> = 0>
  reference operator[](size_type pos) {
    return data_[pos];
  }

  template <MemoryType M = V,
            typename std::enable_if_t<M != MemoryType::Device, int> = 0>
  const_reference operator[](size_type pos) const {
    return data_[pos];
  }

  template <MemoryType M = V,
            typename std::enable_if_t<M != MemoryType::Device, int> = 0>
  reference at(size_type pos) {
    if (pos >= size_) {
      throw std::out_of_range("DataFrame::at index out of range");
    }
    return data_[pos];
  }

  template <MemoryType M = V,
            typename std::enable_if_t<M != MemoryType::Device, int> = 0>
  const_reference at(size_type pos) const {
    if (pos >= size_) {
      throw std::out_of_range("DataFrame::at index out of range");
    }
    return data_[pos];
  }

  template <MemoryType M = V,
            typename std::enable_if_t<M != MemoryType::Device, int> = 0>
  reference front() {
    return data_[0];
  }

  template <MemoryType M = V,
            typename std::enable_if_t<M != MemoryType::Device, int> = 0>
  const_reference front() const {
    return data_[0];
  }

  template <MemoryType M = V,
            std::enable_if_t<M != MemoryType::Device, int> = 0>
  reference back() {
    return data_[size_ - 1];
  }

  template <MemoryType M = V,
            std::enable_if_t<M != MemoryType::Device, int> = 0>
  const_reference back() const {
    return data_[size_ - 1];
  }

  void sync() { memory_.sync(); }

  Memory<V> &get_memory() noexcept { return memory_; }
  const Memory<V> &get_memory() const noexcept { return memory_; }

  std::vector<T> to_host_vector() const {
    std::vector<T> host(size_);
    backend_type::copy_to_host(host.data(), data_, size_);
    return host;
  }

  void swap(DataFrame &other) noexcept {
    using std::swap;
    swap(memory_, other.memory_);
    swap(data_, other.data_);
    swap(size_, other.size_);
    swap(capacity_, other.capacity_);
  }

private:
  static constexpr size_type kMinGrowth = 16;

  static size_type growth_policy(size_type current) {
    if (current == 0) {
      return kMinGrowth;
    }
    return current + current / 2 + 1;
  }

  static size_type bytes_for(size_type count) { return count * sizeof(T); }

  void ensure_capacity_for_growth(size_type desired_size) {
    if (desired_size <= capacity_) {
      return;
    }
    reserve(std::max(desired_size, growth_policy(capacity_)));
  }

  void reallocate(size_type new_capacity) {
    Memory<V> new_memory;
    if (new_capacity != 0) {
      allocate_bytes(new_memory, bytes_for(new_capacity));
    }
    pointer new_data = new_memory.template as<T>();
    if (data_ && new_data) {
      backend_type::copy_same(new_data, data_, size_);
    }
    memory_ = std::move(new_memory);
    data_ = new_data;
    capacity_ = new_capacity;
  }

  static void allocate_bytes(Memory<V> &mem, size_type bytes) {
    if (bytes == 0) {
      return;
    }
    if (mem.allocate(bytes) != 0) {
      throw std::runtime_error("hip allocation failed in DataFrame");
    }
  }

  void assign_from_host(const_pointer host_ptr, size_type count) {
    if (count == 0) {
      clear();
      return;
    }
    reserve(count);
    backend_type::copy_from_host(data_, host_ptr, count);
    size_ = count;
    sync_if_needed();
  }

  void copy_from(const DataFrame &other) {
    if (other.size_ == 0) {
      clear();
      return;
    }
    reserve(other.size_);
    backend_type::copy_same(data_, other.data_, other.size_);
    size_ = other.size_;
    sync_if_needed();
  }

  void sync_if_needed() {
    if constexpr (V == MemoryType::Managed) {
      memory_.sync();
    }
  }

  void release() {
    memory_ = Memory<V>();
    data_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  Memory<V> memory_;
  pointer data_ = nullptr;
  size_type size_ = 0;
  size_type capacity_ = 0;
};

template <typename T> using DeviceDataFrame = DataFrame<T, MemoryType::Device>;
template <typename T> using HostDataFrame = DataFrame<T, MemoryType::Host>;
template <typename T>
using ManagedDataFrame = DataFrame<T, MemoryType::Managed>;

} // namespace hipster

#endif // HIPSTER_MEMORY_DATA_FRAME_HPP