#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace mst {

struct VertexId {
  int value;
};

struct Weight {
  int value;
};

inline constexpr VertexId vertex_id(int value) { return VertexId{value}; }
inline constexpr Weight weight(int value) { return Weight{value}; }
inline constexpr int index_of(VertexId value) { return value.value; }
inline constexpr int value_of(Weight value) { return value.value; }

inline constexpr bool operator==(VertexId lhs, VertexId rhs) {
  return lhs.value == rhs.value;
}

inline constexpr bool operator!=(VertexId lhs, VertexId rhs) {
  return !(lhs == rhs);
}

inline constexpr bool operator<(VertexId lhs, VertexId rhs) {
  return lhs.value < rhs.value;
}

inline constexpr bool operator==(Weight lhs, Weight rhs) {
  return lhs.value == rhs.value;
}

inline constexpr bool operator!=(Weight lhs, Weight rhs) {
  return !(lhs == rhs);
}

inline constexpr bool operator<(Weight lhs, Weight rhs) {
  return lhs.value < rhs.value;
}

struct Edge {
  VertexId u;
  VertexId v;
  Weight weight;
};

struct Graph {
  int vertex_count;
  std::vector<Edge> edges;
};

using Candidate = std::optional<Edge>;

inline constexpr Weight kInfiniteWeight{std::numeric_limits<int>::max()};

// Boruvka works in rounds. In each round every component picks its cheapest
// outgoing edge, then those edges are merged into the forest.
inline Candidate invalid_candidate() { return std::nullopt; }

inline Graph make_test_graph() {
  return Graph{
      12,
      {
          {vertex_id(0), vertex_id(1), weight(4)},
          {vertex_id(0), vertex_id(2), weight(3)},
          {vertex_id(0), vertex_id(3), weight(8)},
          {vertex_id(1), vertex_id(2), weight(1)},
          {vertex_id(1), vertex_id(4), weight(7)},
          {vertex_id(1), vertex_id(5), weight(9)},
          {vertex_id(2), vertex_id(4), weight(2)},
          {vertex_id(2), vertex_id(5), weight(6)},
          {vertex_id(2), vertex_id(6), weight(7)},
          {vertex_id(3), vertex_id(5), weight(5)},
          {vertex_id(3), vertex_id(6), weight(4)},
          {vertex_id(3), vertex_id(7), weight(9)},
          {vertex_id(4), vertex_id(5), weight(3)},
          {vertex_id(4), vertex_id(7), weight(8)},
          {vertex_id(4), vertex_id(8), weight(6)},
          {vertex_id(5), vertex_id(7), weight(4)},
          {vertex_id(5), vertex_id(8), weight(2)},
          {vertex_id(6), vertex_id(8), weight(7)},
          {vertex_id(6), vertex_id(9), weight(6)},
          {vertex_id(7), vertex_id(8), weight(1)},
          {vertex_id(7), vertex_id(10), weight(5)},
          {vertex_id(8), vertex_id(10), weight(3)},
          {vertex_id(8), vertex_id(11), weight(9)},
          {vertex_id(9), vertex_id(11), weight(2)},
          {vertex_id(10), vertex_id(11), weight(4)},
          {vertex_id(1), vertex_id(3), weight(9)},
          {vertex_id(4), vertex_id(9), weight(10)},
          {vertex_id(5), vertex_id(10), weight(7)},
          {vertex_id(6), vertex_id(11), weight(8)},
      },
  };
}

// The algorithms treat the graph as undirected, so an edge is compared by the
// unordered pair of its endpoints rather than the original direction.
inline std::pair<int, int> normalized_endpoints(const Edge &edge) {
  return {std::min(index_of(edge.u), index_of(edge.v)),
          std::max(index_of(edge.u), index_of(edge.v))};
}

inline std::pair<int, int> normalized_endpoints(const Candidate &candidate) {
  if (!candidate) {
    return {-1, -1};
  }
  return normalized_endpoints(*candidate);
}

inline bool better_candidate(const Candidate &lhs, const Candidate &rhs) {
  if (lhs.has_value() != rhs.has_value()) {
    return lhs.has_value();
  }
  if (!lhs) {
    return false;
  }
  if (lhs->weight != rhs->weight) {
    return lhs->weight < rhs->weight;
  }
  const auto lhs_endpoints = normalized_endpoints(lhs);
  const auto rhs_endpoints = normalized_endpoints(rhs);
  if (lhs_endpoints.first != rhs_endpoints.first) {
    return lhs_endpoints.first < rhs_endpoints.first;
  }
  return lhs_endpoints.second < rhs_endpoints.second;
}

inline void consider_candidate(Candidate &current, VertexId u, VertexId v, Weight w) {
  const Candidate next = Edge{u, v, w};
  if (better_candidate(next, current)) {
    current = next;
  }
}

// Compact textual form used in the final MST summary.
inline std::string candidate_to_string(const Candidate &candidate) {
  if (!candidate) {
    return "invalid";
  }

  std::ostringstream out;
  out << index_of(candidate->u) << "-" << index_of(candidate->v) << "("
      << value_of(candidate->weight) << ")";
  return out.str();
}

// Read-only root lookup for a parent array snapshot. This is useful inside
// OpenMP regions, where mutating the shared DSU would create data races.
inline VertexId find_root(const std::vector<VertexId> &parent, VertexId x) {
  while (parent[static_cast<std::size_t>(index_of(x))] != x) {
    x = parent[static_cast<std::size_t>(index_of(x))];
  }
  return x;
}

inline std::vector<int> pack_vertices(const std::vector<VertexId> &vertices) {
  std::vector<int> packed;
  packed.reserve(vertices.size());
  for (const VertexId value : vertices) {
    packed.push_back(index_of(value));
  }
  return packed;
}

inline std::vector<VertexId> unpack_vertices(const std::vector<int> &vertices) {
  std::vector<VertexId> unpacked;
  unpacked.reserve(vertices.size());
  for (const int value : vertices) {
    unpacked.push_back(vertex_id(value));
  }
  return unpacked;
}

class DisjointSet {
public:
  explicit DisjointSet(int n) : parent_(n), size_(n, 1) {
    for (int index = 0; index < n; ++index) {
      parent_[static_cast<std::size_t>(index)] = vertex_id(index);
    }
  }

  // Path-compressing root lookup used by the sequential coordinator logic.
  VertexId find(VertexId x) {
    if (parent_[static_cast<std::size_t>(index_of(x))] != x) {
      parent_[static_cast<std::size_t>(index_of(x))] = find(parent_[static_cast<std::size_t>(index_of(x))]);
    }
    return parent_[static_cast<std::size_t>(index_of(x))];
  }

  // Union-by-size keeps the forest shallow, which matters because Boruvka
  // performs many find operations in each round.
  bool unite(VertexId left, VertexId right) {
    left = find(left);
    right = find(right);
    if (left == right) {
      return false;
    }

    if (size_[static_cast<std::size_t>(index_of(left))] <
        size_[static_cast<std::size_t>(index_of(right))]) {
      std::swap(left, right);
    }

    parent_[static_cast<std::size_t>(index_of(right))] = left;
    size_[static_cast<std::size_t>(index_of(left))] +=
        size_[static_cast<std::size_t>(index_of(right))];
    return true;
  }

  // After MPI broadcasts a fresh parent array, every rank rebuilds the local
  // metadata from that snapshot so the next round starts from a consistent DSU.
  void set_parent(std::vector<VertexId> parent) {
    parent_ = std::move(parent);
    if (size_.size() != parent_.size()) {
      size_.assign(parent_.size(), 1);
    }
    rebuild_metadata();
  }

  // Recompute component sizes and compress paths in one pass.
  void rebuild_metadata() {
    std::fill(size_.begin(), size_.end(), 0);
    for (std::size_t index = 0; index < parent_.size(); ++index) {
      const VertexId root = find(vertex_id(static_cast<int>(index)));
      ++size_[static_cast<std::size_t>(index_of(root))];
    }
  }

  // The number of connected components is just the number of roots remaining
  // in the DSU forest.
  int component_count() {
    int count = 0;
    for (std::size_t index = 0; index < parent_.size(); ++index) {
      if (find(vertex_id(static_cast<int>(index))) == vertex_id(static_cast<int>(index))) {
        ++count;
      }
    }
    return count;
  }

  const std::vector<VertexId> &parent() const { return parent_; }

private:
  std::vector<VertexId> parent_;
  std::vector<int> size_;
};

inline std::string mst_summary(const std::vector<Edge> &edges,
                               int total_weight) {
  std::vector<Edge> sorted = edges;
  std::sort(sorted.begin(), sorted.end(), [](const Edge &lhs, const Edge &rhs) {
    const auto lhs_endpoints = normalized_endpoints(lhs);
    const auto rhs_endpoints = normalized_endpoints(rhs);
    if (lhs_endpoints.first != rhs_endpoints.first) {
      return lhs_endpoints.first < rhs_endpoints.first;
    }
    if (lhs_endpoints.second != rhs_endpoints.second) {
      return lhs_endpoints.second < rhs_endpoints.second;
    }
    return lhs.weight < rhs.weight;
  });

  std::ostringstream out;
  out << "MST weight = " << total_weight << ", edges = " << sorted.size()
      << '\n';
  for (const Edge &edge : sorted) {
    out << "  " << index_of(edge.u) << "-" << index_of(edge.v) << " ("
        << value_of(edge.weight) << ")\n";
  }
  return out.str();
}

} // namespace mst
