#ifndef HIPSTER_MEMORY_DATA_FRAME_HPP
#define HIPSTER_MEMORY_DATA_FRAME_HPP

#include "memory.hpp"
#include <algorithm>
#include <initializer_list>
#include <iterator>
#include <stdexcept>

namespace hipster {

template <typename T, MemoryType V = MemoryType::Managed> class DataFrame {
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

  DataFrame(std::initializer_list<T> init) {
    resize(init.size());
    std::copy(init.begin(), init.end(), data_);
  }

  ~DataFrame() = default;

  DataFrame(const DataFrame &other) {
    resize(other.size_);
    if constexpr (V == MemoryType::Managed) {
      std::copy(other.begin(), other.end(), begin());
      memory_.sync();
    } else if constexpr (V == MemoryType::Device) {
      memory_.copyToDevice(other.data_, other.size_ * sizeof(T));
      memory_.sync();
    } else {
      std::copy(other.begin(), other.end(), begin());
    }
  }

  DataFrame(DataFrame &&other) noexcept = default;

  DataFrame &operator=(const DataFrame &other) {
    if (this != &other) {
      resize(other.size_);
      if constexpr (V == MemoryType::Managed) {
        std::copy(other.begin(), other.end(), begin());
        memory_.sync();
      } else if constexpr (V == MemoryType::Device) {
        memory_.copyToDevice(other.data_, other.size_ * sizeof(T));
        memory_.sync();
      } else {
        std::copy(other.begin(), other.end(), begin());
      }
    }
    return *this;
  }

  DataFrame &operator=(DataFrame &&other) noexcept = default;

  void resize(size_type new_size) {
    if (new_size == size_)
      return;

    Memory<V> new_memory;
    if (new_size > 0) {
      new_memory.allocate(new_size * sizeof(T));
    }

    if (data_ && new_memory.get()) {
      T *new_data = new_memory.template as<T>();
      size_type copy_size = std::min(size_, new_size);
      if constexpr (V == MemoryType::Managed) {
        std::copy(data_, data_ + copy_size, new_data);
        memory_.sync();
      } else if constexpr (V == MemoryType::Device) {
        new_memory.copyToDevice(data_, copy_size * sizeof(T));
        new_memory.sync();
      } else {
        std::copy(data_, data_ + copy_size, new_data);
      }
    }

    memory_ = std::move(new_memory);
    data_ = memory_.template as<T>();
    size_ = new_size;
  }

  void push_back(const T &value) {
    resize(size_ + 1);
    data_[size_ - 1] = value;
    if constexpr (V == MemoryType::Managed) {
      memory_.sync();
    }
  }

  void pop_back() {
    if (size_ > 0) {
      resize(size_ - 1);
    }
  }

  T *data() { return data_; }
  const T *data() const { return data_; }

  size_type size() const { return size_; }
  size_type capacity() const { return size_; }
  bool empty() const { return size_ == 0; }

  iterator begin() { return data_; }
  const_iterator begin() const { return data_; }
  const_iterator cbegin() const { return data_; }

  iterator end() { return data_ + size_; }
  const_iterator end() const { return data_ + size_; }
  const_iterator cend() const { return data_ + size_; }

  reference operator[](size_type pos) { return data_[pos]; }
  const_reference operator[](size_type pos) const { return data_[pos]; }

  reference at(size_type pos) {
    if (pos >= size_)
      throw std::out_of_range("DataFrame index out of range");
    return data_[pos];
  }

  const_reference at(size_type pos) const {
    if (pos >= size_)
      throw std::out_of_range("DataFrame index out of range");
    return data_[pos];
  }

  reference front() { return data_[0]; }
  const_reference front() const { return data_[0]; }

  reference back() { return data_[size_ - 1]; }
  const_reference back() const { return data_[size_ - 1]; }

  void sync() { memory_.sync(); }

  Memory<V> &get_memory() { return memory_; }
  const Memory<V> &get_memory() const { return memory_; }

private:
  Memory<V> memory_;
  T *data_ = nullptr;
  size_type size_ = 0;
};

template <typename T> using DeviceDataFrame = DataFrame<T, MemoryType::Device>;
template <typename T> using HostDataFrame = DataFrame<T, MemoryType::Host>;
template <typename T>
using ManagedDataFrame = DataFrame<T, MemoryType::Managed>;

} // namespace hipster

#endif // HIPSTER_MEMORY_DATA_FRAME_HPP