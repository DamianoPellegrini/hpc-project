#pragma once

#include <optional>
#include <utility>

#include "mst/core/types.hpp"

namespace mst::core {

/// Arco non orientato e pesato, memorizzato nel grafo condiviso del core.
struct edge {
  vertex_id u;
  vertex_id v;
  edge_weight weight;

  friend constexpr bool operator==(edge lhs, edge rhs) noexcept {
    return lhs.u == rhs.u && lhs.v == rhs.v && lhs.weight == rhs.weight;
  }
};

/// Arco proposto da una componente, non ancora passato per il DSU (potrebbe
/// chiudere un ciclo).
struct candidate_edge {
  edge value;
  edge_index index = make_edge_index(0);
};

/// Arco ammesso nell'MST a seguito di un'unione riuscita.
struct mst_edge {
  edge value;
};

using maybe_candidate_edge = std::optional<candidate_edge>;

/// Chiave (peso, indice) di un candidato: serve a confrontarli e scegliere il migliore.
inline constexpr candidate_key key_for(candidate_edge candidate) noexcept {
  return make_candidate_key(candidate.value.weight, candidate.index);
}

/// Estremi ordinati (minore, maggiore): per confrontare archi a prescindere da come sono memorizzati `u`/`v`.
inline constexpr std::pair<int, int>
normalized_endpoints(edge edge_value) noexcept {
  const int left = edge_value.u.value();
  const int right = edge_value.v.value();
  if (left < right) {
    return {left, right};
  }
  return {right, left};
}

/// Ordine totale e deterministico fra candidati: presente batte assente,
/// poi vince la `candidate_key` più piccola, e a parità decidono gli
/// estremi normalizzati (per non lasciare mai un pareggio).
inline constexpr bool better_candidate(maybe_candidate_edge lhs,
                                       maybe_candidate_edge rhs) noexcept {
  if (lhs.has_value() != rhs.has_value()) {
    return lhs.has_value();
  }
  if (!lhs) {
    return false;
  }
  const candidate_key lhs_key = key_for(*lhs);
  const candidate_key rhs_key = key_for(*rhs);
  if (lhs_key != rhs_key) {
    return lhs_key < rhs_key;
  }
  return normalized_endpoints(lhs->value) < normalized_endpoints(rhs->value);
}

/// Aggiorna `current` con `next` se è un candidato migliore (qui con indice di default).
inline void consider_candidate(maybe_candidate_edge &current, edge next) {
  const maybe_candidate_edge candidate = candidate_edge{next};
  if (better_candidate(candidate, current)) {
    current = candidate;
  }
}

/// Come sopra, ma con indice esplicito: serve a preservare la posizione
/// originale dell'arco per ricostruire la `candidate_key` coerentemente fra backend.
inline void consider_candidate(maybe_candidate_edge &current, edge next,
                               edge_index index) {
  const maybe_candidate_edge candidate = candidate_edge{next, index};
  if (better_candidate(candidate, current)) {
    current = candidate;
  }
}

} // namespace mst::core
