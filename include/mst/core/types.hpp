#pragma once

#include <compare>
#include <cstddef>
#include <limits>

namespace mst::core {

/// Vertex identifier inside a validated graph.
struct vertex_id {
  int value;

  friend constexpr auto operator<=>(vertex_id, vertex_id) = default;
};

/// Component identifier derived from a DSU root.
struct component_id {
  int value;

  friend constexpr auto operator<=>(component_id, component_id) = default;
};

/// MPI rank identifier for distributed partition ownership.
struct rank_id {
  int value;

  friend constexpr auto operator<=>(rank_id, rank_id) = default;
};

/// Stable index into an edge collection.
struct edge_index {
  std::size_t value;

  friend constexpr auto operator<=>(edge_index, edge_index) = default;
};

/// Logical partition identifier for backend-owned slices.
struct partition_id {
  int value;

  friend constexpr auto operator<=>(partition_id, partition_id) = default;
};

/// Boruvka round counter.
struct round_index {
  int value;

  friend constexpr auto operator<=>(round_index, round_index) = default;
};

/// Edge weight used by all backends.
struct edge_weight {
  int value;

  friend constexpr auto operator<=>(edge_weight, edge_weight) = default;
};

/// Graph state before input validation.
struct raw {};
/// Graph state after vertex-bound validation.
struct validated {};
/// Local component state before a backend-wide synchronization point.
struct unsynchronized {};
/// Component state after synchronization.
struct synchronized_state {};
/// DSU parents before path compression.
struct uncompressed_parents {};
/// DSU parents after path compression.
struct compressed_parents {};
/// Forest state guaranteed to contain no admitted cycle.
struct acyclic {};

inline constexpr edge_weight infinite_weight{std::numeric_limits<int>::max()};

constexpr vertex_id make_vertex_id(int value) noexcept { return vertex_id{value}; }
constexpr component_id make_component_id(int value) noexcept { return component_id{value}; }
constexpr edge_weight make_edge_weight(int value) noexcept { return edge_weight{value}; }

constexpr int as_index(vertex_id id) noexcept { return id.value; }
constexpr int as_index(component_id id) noexcept { return id.value; }
constexpr int as_value(edge_weight weight) noexcept { return weight.value; }

} // namespace mst::core
