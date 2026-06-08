#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace mst::core {

// Tipi forti attorno a interi grezzi (il compilatore blocca, es., uno
// scambio fra `edge_index` e `vertex_id`) e tag vuoti come `raw`/`validated`
// che codificano nel tipo stesso lo stato di un dato (typestate), spostando
// dei controlli dal runtime al tempo di compilazione.

/// Identificatore di vertice all'interno di un grafo validato.
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

/// Identificatore di componente, derivato dalla radice nel DSU.
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

/// Identificatore di rank MPI, usato per la suddivisione distribuita dei dati.
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

/// Indice stabile all'interno di una collezione di archi.
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

/// Numero di archi in un grafo o in una porzione di proprietà di un backend.
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

/// Numero di vertici richiesto per i grafi generati.
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

/// Chiave deterministica per identificare un arco fra backend diversi.
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

/// Chiave d'ordinamento dei candidati impacchettata: prima il peso minore,
/// poi, a parità di peso, l'indice d'arco minore.
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

/// Seed usato dagli esempi di grafi generati in modo deterministico.
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

/// Identificatore logico di partizione per le porzioni di proprietà di un
/// backend.
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

/// Contatore del round di Boruvka.
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

/// Peso di un arco, usato da tutti i backend.
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

/// Stato del grafo prima della validazione dell'input.
struct raw {};
/// Stato del grafo dopo la validazione dei limiti sui vertici.
struct validated {};
/// Stato locale delle componenti prima di un punto di sincronizzazione fra
/// backend.
struct unsynchronized {};
/// Stato delle componenti dopo la sincronizzazione.
struct synchronized_state {};
/// Genitori del DSU prima della compressione dei cammini.
struct uncompressed_parents {};
/// Genitori del DSU dopo la compressione dei cammini.
struct compressed_parents {};
/// Stato della foresta garantito privo di cicli ammessi.
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
