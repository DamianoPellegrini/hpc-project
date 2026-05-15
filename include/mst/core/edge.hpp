#pragma once

#include <optional>
#include <utility>

#include "mst/core/types.hpp"

namespace mst::core {

/// Undirected weighted edge stored in the shared core graph.
struct edge {
  vertex_id u;
  vertex_id v;
  edge_weight weight;
};

/// Candidate edge proposed by a component before union validation.
struct candidate_edge {
  edge value;
};

/// Edge admitted into the MST by a successful union.
struct mst_edge {
  edge value;
};

using maybe_candidate_edge = std::optional<candidate_edge>;

inline constexpr std::pair<int, int> normalized_endpoints(edge edge_value) noexcept {
  const int left = as_index(edge_value.u);
  const int right = as_index(edge_value.v);
  if (left < right) {
    return {left, right};
  }
  return {right, left};
}

inline constexpr bool better_candidate(maybe_candidate_edge lhs,
                                       maybe_candidate_edge rhs) noexcept {
  if (lhs.has_value() != rhs.has_value()) {
    return lhs.has_value();
  }
  if (!lhs) {
    return false;
  }
  if (lhs->value.weight != rhs->value.weight) {
    return lhs->value.weight < rhs->value.weight;
  }
  return normalized_endpoints(lhs->value) < normalized_endpoints(rhs->value);
}

inline void consider_candidate(maybe_candidate_edge &current, edge next) {
  const maybe_candidate_edge candidate = candidate_edge{next};
  if (better_candidate(candidate, current)) {
    current = candidate;
  }
}

} // namespace mst::core
