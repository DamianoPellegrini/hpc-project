#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/types.hpp"
#include "mst/dsu/disjoint_set.hpp"

namespace mst::dsu {

/// DSU lock-free condiviso fra thread (backend OpenMP): genitori come
/// `std::atomic<int>`, niente mutex. `unite` usa compare-exchange con retry,
/// `find_index` fa path-halving "opportunistico" (CAS fallito? pazienza,
/// vuol dire che qualcun altro ha già aggiornato).
class parallel_disjoint_set {
public:
  /// Ogni vertice parte come componente a sé (store `relaxed`: siamo ancora
  /// prima della condivisione fra thread, nessuna sincronizzazione serve).
  explicit parallel_disjoint_set(int vertex_count)
      : parent_(static_cast<std::size_t>(vertex_count)) {
    for (int index = 0; index < vertex_count; ++index) {
      parent_[static_cast<std::size_t>(index)].store(index,
                                                     std::memory_order_relaxed);
    }
  }

  /// Path-halving lock-free: prova a far puntare il nodo al nonno con CAS,
  /// e se fallisce prosegue lo stesso (tanto qualcuno ce l'ha già messo).
  int find_index(int vertex) {
    int current = vertex;
    while (true) {
      const int parent = parent_[static_cast<std::size_t>(current)].load(
          std::memory_order_acquire);
      const int grandparent = parent_[static_cast<std::size_t>(parent)].load(
          std::memory_order_acquire);
      if (parent == grandparent) {
        return parent;
      }
      int expected = parent;
      parent_[static_cast<std::size_t>(current)].compare_exchange_weak(
          expected, grandparent, std::memory_order_acq_rel,
          std::memory_order_acquire);
      current = parent;
    }
  }

  /// La componente di un vertice è la sua radice.
  mst::core::component_id find(mst::core::vertex_id vertex) {
    return mst::core::make_component_id(find_index(vertex.value()));
  }

  /// Versione lock-free di unite: leggi le radici, scarta se coincidono
  /// (ciclo), scegli sempre la radice con indice minore come nuovo genitore
  /// (regola identica per tutti i thread, evita cicli fra radici "in gara"),
  /// e prova il CAS — se fallisce, ritenta da capo e conta la contesa.
  std::optional<mst::core::mst_edge>
  unite(mst::core::candidate_edge candidate,
        std::atomic<std::uint64_t> *retry_counter = nullptr) {
    while (true) {
      const int left = find_index(candidate.value.u.value());
      const int right = find_index(candidate.value.v.value());
      if (left == right) {
        return std::nullopt;
      }

      const int parent = left < right ? left : right;
      const int child = left < right ? right : left;
      int expected = child;
      if (parent_[static_cast<std::size_t>(child)].compare_exchange_strong(
              expected, parent, std::memory_order_acq_rel,
              std::memory_order_acquire)) {
        return mst::core::mst_edge{candidate.value};
      }
      if (retry_counter != nullptr) {
        retry_counter->fetch_add(1, std::memory_order_relaxed);
      }
    }
  }

  /// Aggancia il vertice direttamente alla sua radice corrente: chiamata
  /// in parallelo su tutti i vertici a fine round per tenere le catene corte.
  void compress_vertex(mst::core::vertex_id vertex) {
    const int root = find_index(vertex.value());
    parent_[vertex.index()].store(root, std::memory_order_release);
  }

  /// Congela lo stato in uno snapshot immutabile: la scansione degli archi
  /// lo legge senza rischiare corse con le contrazioni dello stesso round.
  parent_snapshot snapshot() const {
    std::vector<mst::core::vertex_id> packed;
    packed.reserve(parent_.size());
    for (const auto &parent : parent_) {
      packed.push_back(mst::core::make_vertex_id(
          parent.load(std::memory_order_acquire)));
    }
    return parent_snapshot{std::move(packed)};
  }

  /// Conta le radici, cioè le componenti rimaste: a 1 il loop si ferma.
  int component_count() {
    int count = 0;
    for (std::size_t index = 0; index < parent_.size(); ++index) {
      if (find_index(static_cast<int>(index)) == static_cast<int>(index)) {
        ++count;
      }
    }
    return count;
  }

private:
  std::vector<std::atomic<int>> parent_;
};

} // namespace mst::dsu
