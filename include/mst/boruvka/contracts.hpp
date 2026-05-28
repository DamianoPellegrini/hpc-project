#pragma once

#include <concepts>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/types.hpp"

namespace mst::boruvka {

/// Final Boruvka result returned by a backend implementation.
struct result {
  std::vector<mst::core::mst_edge> edges;
  int total_weight = 0;
};

template <class engine_t>
concept candidate_scanner = requires(engine_t engine,
                                     mst::core::round_index round) {
  engine.find_local_minima(round);
};

template <class engine_t>
concept candidate_reducer = requires(engine_t engine,
                                     mst::core::round_index round) {
  engine.reduce_component_minima(round);
};

template <class engine_t>
concept component_contractor = requires(engine_t engine,
                                        mst::core::round_index round) {
  engine.apply_contractions(round);
};

template <class engine_t>
concept parent_compressor = requires(engine_t engine,
                                     mst::core::round_index round) {
  engine.compress_parents(round);
};

template <class engine_t>
concept result_provider = requires(engine_t engine) {
  { engine.result() } -> std::same_as<result>;
};

template <class engine_t>
concept boruvka_round_engine =
    requires(engine_t engine, mst::core::validated_graph graph) {
      typename engine_t::execution_domain;
      typename engine_t::memory_space;
      typename engine_t::reduction_policy;
      typename engine_t::contraction_policy;
      engine.initialize(graph);
    } && candidate_scanner<engine_t> && candidate_reducer<engine_t> &&
    component_contractor<engine_t> && parent_compressor<engine_t> &&
    result_provider<engine_t>;

} // namespace mst::boruvka
