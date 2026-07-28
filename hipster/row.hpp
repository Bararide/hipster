#ifndef HIPSTER_HIPSTER_ROW_HPP
#define HIPSTER_HIPSTER_ROW_HPP

#include "memory.hpp"

namespace hipster {

template <typename Derived, MemoryType MemType = MemoryType::Managed,
          typename T = float>
class Row {
public:
  Row() = default;

  explicit Row(size_t size) { allocate(size); }

  Row(const Row &) = delete;
  Row &operator=(const Row &) = delete;

  Row(Row &&other) noexcept : row_(std::move(other.row_)) {}

  Row &operator=(Row &&other) noexcept {
    if (this != &other) {
      row_ = std::move(other.row_);
    }
    return *this;
  }

  int allocate(size_t size) { return row_.allocate(size * sizeof(T)); }

  T *data() { return row_.template as<T>(); }
  const T *data() const { return row_.template as<T>(); }

  size_t size() const { return row_.size() / sizeof(T); }
  bool empty() const { return size() == 0; }

  T *begin() { return data(); }
  T *end() { return data() + size(); }
  const T *begin() const { return data(); }
  const T *end() const { return data() + size(); }
  const T *cbegin() const { return data(); }
  const T *cend() const { return data() + size(); }

  T &operator[](size_t idx) { return data()[idx]; }
  const T &operator[](size_t idx) const { return data()[idx]; }

  explicit operator bool() const { return row_.operator bool(); }

  int sync() { return row_.sync(); }

  const Memory<MemType> &memory() const { return row_; }
  Memory<MemType> &memory() { return row_; }

protected:
  Memory<MemType> row_;

private:
  Derived &derived() { return static_cast<Derived &>(*this); }
  const Derived &derived() const { return static_cast<const Derived &>(*this); }
};

template <typename T, MemoryType MemType = MemoryType::Managed>
Row<void, MemType, T> make_row(const std::vector<T> &data) {
  Row<void, MemType, T> row(data.size());
  if constexpr (MemType == MemoryType::Device) {
    hipMemcpy(row.data(), data.data(), data.size() * sizeof(T),
              hipMemcpyHostToDevice);
  } else {
    std::copy(data.begin(), data.end(), row.begin());
  }
  return row;
}

template <typename T, MemoryType MemType = MemoryType::Managed>
Row<void, MemType, T> make_row(std::vector<T> &&data) {
  Row<void, MemType, T> row(data.size());
  std::move(data.begin(), data.end(), row.begin());
  return row;
}

} // namespace hipster

#endif // HIPSTER_HIPSTER_ROW_HPP