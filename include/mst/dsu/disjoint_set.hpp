#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/types.hpp"

namespace mst::dsu {

/// Read-only parent array snapshot safe to share across parallel scans.
class parent_snapshot {
public:
  explicit parent_snapshot(std::vector<mst::core::vertex_id> parent)
      : parent_(std::move(parent)) {}

  const std::vector<mst::core::vertex_id> &parent() const noexcept {
    return parent_;
  }

private:
  std::vector<mst::core::vertex_id> parent_;
};

inline mst::core::vertex_id find_root(const parent_snapshot &snapshot,
                                      mst::core::vertex_id vertex) {
  mst::core::vertex_id current = vertex;
  while (snapshot.parent()[static_cast<std::size_t>(mst::core::as_index(current))] !=
         current) {
    current =
        snapshot.parent()[static_cast<std::size_t>(mst::core::as_index(current))];
  }
  return current;
}

template <class compression_state_t>
class disjoint_set;

template <>
class disjoint_set<mst::core::uncompressed_parents> {
public:
  /// Create a DSU with one singleton component per vertex.
  explicit disjoint_set(int vertex_count)
      : parent_(static_cast<std::size_t>(vertex_count)),
        size_(static_cast<std::size_t>(vertex_count), 1) {
    for (int index = 0; index < vertex_count; ++index) {
      parent_[static_cast<std::size_t>(index)] = mst::core::make_vertex_id(index);
    }
  }

  mst::core::component_id find(mst::core::vertex_id vertex) {
    return mst::core::make_component_id(
        mst::core::as_index(find_vertex(vertex)));
  }

  std::optional<mst::core::mst_edge> unite(mst::core::candidate_edge candidate) {
    mst::core::vertex_id left =
        mst::core::make_vertex_id(mst::core::as_index(find_vertex(candidate.value.u)));
    mst::core::vertex_id right =
        mst::core::make_vertex_id(mst::core::as_index(find_vertex(candidate.value.v)));
    if (left == right) {
      return std::nullopt;
    }

    if (size_[static_cast<std::size_t>(mst::core::as_index(left))] <
        size_[static_cast<std::size_t>(mst::core::as_index(right))]) {
      std::swap(left, right);
    }

    parent_[static_cast<std::size_t>(mst::core::as_index(right))] = left;
    size_[static_cast<std::size_t>(mst::core::as_index(left))] +=
        size_[static_cast<std::size_t>(mst::core::as_index(right))];
    return mst::core::mst_edge{candidate.value};
  }

  int component_count() {
    int count = 0;
    for (std::size_t index = 0; index < parent_.size(); ++index) {
      const auto vertex = mst::core::make_vertex_id(static_cast<int>(index));
      if (find_vertex(vertex) == vertex) {
        ++count;
      }
    }
    return count;
  }

  parent_snapshot compressed_snapshot() {
    for (std::size_t index = 0; index < parent_.size(); ++index) {
      const auto vertex = mst::core::make_vertex_id(static_cast<int>(index));
      parent_[index] = find_vertex(vertex);
    }
    return parent_snapshot{parent_};
  }

  void set_parent_snapshot(parent_snapshot snapshot) {
    parent_ = snapshot.parent();
    size_.assign(parent_.size(), 0);
    for (std::size_t index = 0; index < parent_.size(); ++index) {
      const auto root = find_vertex(mst::core::make_vertex_id(static_cast<int>(index)));
      ++size_[static_cast<std::size_t>(mst::core::as_index(root))];
    }
  }

  const std::vector<mst::core::vertex_id> &parent() const noexcept {
    return parent_;
  }

private:
  mst::core::vertex_id find_vertex(mst::core::vertex_id vertex) {
    auto &parent =
        parent_[static_cast<std::size_t>(mst::core::as_index(vertex))];
    if (parent != vertex) {
      parent = find_vertex(parent);
    }
    return parent;
  }

  std::vector<mst::core::vertex_id> parent_;
  std::vector<int> size_;
};

} // namespace mst::dsu
