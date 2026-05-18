#pragma once

#include <compare>
#include <cstddef>
#include <limits>

namespace mst::core {

/// Vertex identifier inside a validated graph.
struct vertex_id {
  constexpr vertex_id() noexcept = default;
  explicit constexpr vertex_id(int raw_value) noexcept : value_(raw_value) {}

  constexpr std::size_t index() const noexcept {
    return static_cast<std::size_t>(value_);
  }
  constexpr int value() const noexcept { return value_; }

  explicit constexpr operator int() const noexcept { return value(); }
  explicit constexpr operator std::size_t() const noexcept { return index(); }

private:
  int value_ = 0;

  friend constexpr bool operator==(vertex_id lhs, vertex_id rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(vertex_id lhs,
                                                    vertex_id rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }
};

/// Component identifier derived from a DSU root.
struct component_id {
  constexpr component_id() noexcept = default;
  explicit constexpr component_id(int raw_value) noexcept : value_(raw_value) {}

  constexpr std::size_t index() const noexcept {
    return static_cast<std::size_t>(value_);
  }
  constexpr int value() const noexcept { return value_; }

  explicit constexpr operator int() const noexcept { return value(); }
  explicit constexpr operator std::size_t() const noexcept { return index(); }

private:
  int value_ = 0;

  friend constexpr bool operator==(component_id lhs,
                                   component_id rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(component_id lhs,
                                                    component_id rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }
};

/// MPI rank identifier for distributed partition ownership.
struct rank_id {
  int value;

  friend constexpr bool operator==(rank_id lhs, rank_id rhs) noexcept {
    return lhs.value == rhs.value;
  }

  friend constexpr std::strong_ordering operator<=>(rank_id lhs,
                                                    rank_id rhs) noexcept {
    return lhs.value <=> rhs.value;
  }
};

/// Stable index into an edge collection.
struct edge_index {
  std::size_t value;

  friend constexpr bool operator==(edge_index lhs, edge_index rhs) noexcept {
    return lhs.value == rhs.value;
  }

  friend constexpr std::strong_ordering operator<=>(edge_index lhs,
                                                    edge_index rhs) noexcept {
    return lhs.value <=> rhs.value;
  }
};

/// Logical partition identifier for backend-owned slices.
struct partition_id {
  int value;

  friend constexpr bool operator==(partition_id lhs,
                                   partition_id rhs) noexcept {
    return lhs.value == rhs.value;
  }

  friend constexpr std::strong_ordering operator<=>(partition_id lhs,
                                                    partition_id rhs) noexcept {
    return lhs.value <=> rhs.value;
  }
};

/// Boruvka round counter.
struct round_index {
  int value;

  friend constexpr bool operator==(round_index lhs, round_index rhs) noexcept {
    return lhs.value == rhs.value;
  }

  friend constexpr std::strong_ordering operator<=>(round_index lhs,
                                                    round_index rhs) noexcept {
    return lhs.value <=> rhs.value;
  }
};

/// Edge weight used by all backends.
struct edge_weight {
  constexpr edge_weight() noexcept = default;
  explicit constexpr edge_weight(int raw_value) noexcept : value_(raw_value) {}

  constexpr int value() const noexcept { return value_; }

  explicit constexpr operator int() const noexcept { return value(); }

private:
  int value_ = 0;

  friend constexpr bool operator==(edge_weight lhs, edge_weight rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(edge_weight lhs,
                                                    edge_weight rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }
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

constexpr vertex_id make_vertex_id(int value) noexcept {
  return vertex_id{value};
}
constexpr component_id make_component_id(int value) noexcept {
  return component_id{value};
}
constexpr edge_weight make_edge_weight(int value) noexcept {
  return edge_weight{value};
}

} // namespace mst::core
