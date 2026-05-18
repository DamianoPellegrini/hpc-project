#pragma once

#include <concepts>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"

namespace mst::boruvka {

/// Final Boruvka result returned by a backend implementation.
struct result {
  std::vector<mst::core::mst_edge> edges;
  int total_weight = 0;
};

/// Static contract shared by all execution backends.
template <class backend_t>
concept backend = requires(backend_t engine, mst::core::validated_graph graph,
                           mst::core::round_index round) {
  typename backend_t::execution_domain;
  typename backend_t::memory_space;
  typename backend_t::reduction_policy;
  typename backend_t::contraction_policy;
  engine.initialize(graph);
  engine.find_local_minima(round);
  engine.reduce_component_minima(round);
  engine.apply_contractions(round);
  { engine.result() } -> std::same_as<result>;
};

} // namespace mst::boruvka
