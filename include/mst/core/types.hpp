#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
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
  constexpr edge_index() noexcept = default;
  explicit constexpr edge_index(std::size_t raw_value) noexcept
      : value_(raw_value) {}

  friend constexpr bool operator==(edge_index lhs, edge_index rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(edge_index lhs,
                                                    edge_index rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }

  constexpr std::size_t value() const noexcept { return value_; }

private:
  std::size_t value_ = 0;
};

/// Number of edges in a graph or backend-owned slice.
struct edge_count {
  constexpr edge_count() noexcept = default;
  explicit constexpr edge_count(std::size_t raw_value) noexcept
      : value_(raw_value) {}

  constexpr std::size_t value() const noexcept { return value_; }

private:
  std::size_t value_ = 0;

  friend constexpr bool operator==(edge_count lhs, edge_count rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(edge_count lhs,
                                                    edge_count rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }
};

/// Number of vertices requested for generated graphs.
struct graph_vertex_count {
  constexpr graph_vertex_count() noexcept = default;
  explicit constexpr graph_vertex_count(int raw_value) noexcept
      : value_(raw_value) {}

  constexpr int value() const noexcept { return value_; }

private:
  int value_ = 0;

  friend constexpr bool operator==(graph_vertex_count lhs,
                                   graph_vertex_count rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering
  operator<=>(graph_vertex_count lhs, graph_vertex_count rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }
};

/// Deterministic key for identifying an edge across backends.
struct edge_key {
  constexpr edge_key() noexcept = default;
  explicit constexpr edge_key(std::uint64_t raw_value) noexcept
      : value_(raw_value) {}

  constexpr std::uint64_t value() const noexcept { return value_; }

private:
  std::uint64_t value_ = 0;

  friend constexpr bool operator==(edge_key lhs, edge_key rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(edge_key lhs,
                                                    edge_key rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }
};

/// Packed candidate ordering key: lower weight, then lower edge index.
struct candidate_key {
  constexpr candidate_key() noexcept
      : value_(std::numeric_limits<std::uint64_t>::max()) {}
  explicit constexpr candidate_key(std::uint64_t raw_value) noexcept
      : value_(raw_value) {}

  constexpr std::uint64_t value() const noexcept { return value_; }
  constexpr bool empty() const noexcept {
    return value_ == std::numeric_limits<std::uint64_t>::max();
  }

private:
  std::uint64_t value_ = std::numeric_limits<std::uint64_t>::max();

  friend constexpr bool operator==(candidate_key lhs,
                                   candidate_key rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(candidate_key lhs,
                                                    candidate_key rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
  }
};

/// Seed used by deterministic generated graph examples.
struct random_seed {
  constexpr random_seed() noexcept = default;
  explicit constexpr random_seed(std::uint64_t raw_value) noexcept
      : value_(raw_value) {}

  constexpr std::uint64_t value() const noexcept { return value_; }

private:
  std::uint64_t value_ = 0;

  friend constexpr bool operator==(random_seed lhs,
                                   random_seed rhs) noexcept {
    return lhs.value_ == rhs.value_;
  }

  friend constexpr std::strong_ordering operator<=>(random_seed lhs,
                                                    random_seed rhs) noexcept {
    return lhs.value_ <=> rhs.value_;
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
inline constexpr candidate_key empty_candidate_key{};

constexpr vertex_id make_vertex_id(int value) noexcept {
  return vertex_id{value};
}
constexpr component_id make_component_id(int value) noexcept {
  return component_id{value};
}
constexpr edge_weight make_edge_weight(int value) noexcept {
  return edge_weight{value};
}
constexpr edge_index make_edge_index(std::size_t value) noexcept {
  return edge_index{value};
}
constexpr edge_count make_edge_count(std::size_t value) noexcept {
  return edge_count{value};
}
constexpr graph_vertex_count make_graph_vertex_count(int value) noexcept {
  return graph_vertex_count{value};
}
constexpr edge_key make_edge_key(std::uint64_t value) noexcept {
  return edge_key{value};
}
constexpr random_seed make_random_seed(std::uint64_t value) noexcept {
  return random_seed{value};
}

constexpr candidate_key make_candidate_key(edge_weight weight,
                                           edge_index index) noexcept {
  return candidate_key{
      (static_cast<std::uint64_t>(
           static_cast<std::uint32_t>(weight.value())) << 32) |
      static_cast<std::uint64_t>(
          static_cast<std::uint32_t>(index.value()))};
}

constexpr edge_index edge_index_from_candidate_key(candidate_key key) noexcept {
  return make_edge_index(
      static_cast<std::size_t>(key.value() & 0xffffffffULL));
}

} // namespace mst::core
