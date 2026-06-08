#pragma once

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

namespace mst::memory {

/// Tag di residenza: memoria residente sull'host.
struct host_memory {};
/// Tag di residenza: memoria residente sul device CUDA.
struct device_memory {};
/// Tag di residenza: memoria host "pinned" (page-locked), buona per il DMA asincrono.
struct pinned_memory {};
/// Tag di residenza: memoria host paginabile ordinaria, nessuna garanzia di velocità verso il device.
struct pageable_memory {};
/// Tag di residenza: memoria unificata, visibile sia da host sia da device.
struct unified_memory {};

/// Buffer host parametrizzato anche su un tag di residenza: la rappresentazione
/// è sempre uno `std::vector`, ma il tag marca a livello di tipo quale
/// strategia CUDA è prevista, evitando di mischiare buffer con politiche diverse.
template <class value_t, class residency_t> class host_buffer {
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

/// Vista non proprietaria su memoria device: solo puntatore e dimensione,
/// la vita dell'allocazione resta a chi la gestisce (es. `device_allocation`).
template <class value_t> class device_buffer {
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
