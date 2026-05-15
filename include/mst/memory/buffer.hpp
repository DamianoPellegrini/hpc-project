#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace mst::memory {

/// Host-resident memory.
struct host_memory {};
/// Device-resident CUDA memory.
struct device_memory {};
/// Host memory suitable for pinned transfer APIs.
struct pinned_memory {};
/// Ordinary pageable host memory.
struct pageable_memory {};
/// Unified host/device memory.
struct unified_memory {};

template <class value_t, class residency_t>
class host_buffer {
public:
  host_buffer() = default;

  explicit host_buffer(std::vector<value_t> values)
      : values_(std::move(values)) {}

  std::span<value_t> span() noexcept { return values_; }
  std::span<const value_t> span() const noexcept { return values_; }
  std::size_t size() const noexcept { return values_.size(); }
  const std::vector<value_t> &values() const noexcept { return values_; }

private:
  std::vector<value_t> values_;
};

template <class value_t>
class device_buffer {
public:
  device_buffer() = default;
  device_buffer(value_t *data, std::size_t size) : data_(data), size_(size) {}

  value_t *data() noexcept { return data_; }
  const value_t *data() const noexcept { return data_; }
  std::size_t size() const noexcept { return size_; }

private:
  value_t *data_ = nullptr;
  std::size_t size_ = 0;
};

} // namespace mst::memory
