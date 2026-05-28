#pragma once

#include <atomic>
#include <cstddef>
#include <optional>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/types.hpp"
#include "mst/dsu/disjoint_set.hpp"

namespace mst::dsu {

class parallel_disjoint_set {
public:
  explicit parallel_disjoint_set(int vertex_count)
      : parent_(static_cast<std::size_t>(vertex_count)) {
    for (int index = 0; index < vertex_count; ++index) {
      parent_[static_cast<std::size_t>(index)].store(index,
                                                     std::memory_order_relaxed);
    }
  }

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

  mst::core::component_id find(mst::core::vertex_id vertex) {
    return mst::core::make_component_id(find_index(vertex.value()));
  }

  std::optional<mst::core::mst_edge>
  unite(mst::core::candidate_edge candidate) {
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
    }
  }

  void compress_vertex(mst::core::vertex_id vertex) {
    const int root = find_index(vertex.value());
    parent_[vertex.index()].store(root, std::memory_order_release);
  }

  parent_snapshot snapshot() const {
    std::vector<mst::core::vertex_id> packed;
    packed.reserve(parent_.size());
    for (const auto &parent : parent_) {
      packed.push_back(mst::core::make_vertex_id(
          parent.load(std::memory_order_acquire)));
    }
    return parent_snapshot{std::move(packed)};
  }

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
