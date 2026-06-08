#pragma once

#include <concepts>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/types.hpp"

namespace mst::boruvka {

/// Output finale di un backend: archi ammessi nell'MST e peso totale.
struct result {
  std::vector<mst::core::mst_edge> edges;
  int total_weight = 0;
};

/// Fase 1: ogni componente cerca il proprio arco uscente più leggero verso
/// una componente diversa — il cuore di Boruvka, fatto in parallelo.
template <class engine_t>
concept candidate_scanner = requires(engine_t engine,
                                     mst::core::round_index round) {
  engine.find_local_minima(round);
};

/// Fase 2: riduce i minimi locali calcolati da thread/rank/blocchi diversi
/// in un unico minimo per componente.
template <class engine_t>
concept candidate_reducer = requires(engine_t engine,
                                     mst::core::round_index round) {
  engine.reduce_component_minima(round);
};

/// Fase 3: prova ad ammettere ogni candidato fondendo le componenti via DSU
/// — qui il numero di componenti scende davvero.
template <class engine_t>
concept component_contractor = requires(engine_t engine,
                                        mst::core::round_index round) {
  engine.apply_contractions(round);
};

/// Fase 4: comprime i cammini del DSU così il round successivo parte da
/// una struttura più piatta.
template <class engine_t>
concept parent_compressor = requires(engine_t engine,
                                     mst::core::round_index round) {
  engine.compress_parents(round);
};

/// Espone il risultato finale a ciclo terminato.
template <class engine_t>
concept result_provider = requires(engine_t engine) {
  { engine.result() } -> std::same_as<result>;
};

/// Il contratto completo: dichiara i tipi che descrivono dove e come gira
/// (dominio, memoria, politiche), sa inizializzarsi da un grafo validato e
/// implementa tutte e quattro le fasi del round più il risultato finale.
/// Riassume, a livello di tipo, la struttura comune a ogni backend.
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
