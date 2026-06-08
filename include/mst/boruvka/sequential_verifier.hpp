#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "mst/boruvka/contracts.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/dsu/disjoint_set.hpp"

namespace mst::boruvka {

/// Esito del confronto fra backend parallelo e riferimento sequenziale:
/// oltre al booleano, tiene i valori attesi/ottenuti per stamparli o
/// metterli nel report in caso di discrepanza.
struct verification_result {
  bool success = false;
  int expected_total_weight = 0;
  std::size_t expected_edge_count = 0;
  int actual_total_weight = 0;
  std::size_t actual_edge_count = 0;
};

/// Boruvka sequenziale "di riferimento": fa da oracolo per verificare i
/// backend paralleli. Stessa logica di tutti gli altri, solo a un thread:
/// scansiona gli archi proponendo per ogni componente il vicino più
/// leggero (`consider_candidate` risolve i pareggi in modo deterministico),
/// poi ogni componente prova a fondersi col proprio candidato. Si ferma
/// quando non resta che una componente o un round non ammette nulla
/// (grafo non connesso). O(log V) round, perché ogni round dimezza
/// (almeno) le componenti.
inline result sequential_cpu_mst(const mst::core::validated_graph &graph) {
  mst::dsu::disjoint_set<mst::core::uncompressed_parents> dsu(
      graph.vertex_count());
  result output;

  while (dsu.component_count() > 1) {
    std::vector<mst::core::maybe_candidate_edge> best(
        static_cast<std::size_t>(graph.vertex_count()));

    // Scansione: ogni arco fra componenti distinte si propone come
    // candidato per entrambe.
    for (std::size_t index = 0; index < graph.edges().size(); ++index) {
      const mst::core::edge &edge = graph.edges()[index];
      const mst::core::component_id left = dsu.find(edge.u);
      const mst::core::component_id right = dsu.find(edge.v);
      if (left == right) {
        continue;
      }

      const mst::core::edge_index edge_index =
          mst::core::make_edge_index(index);
      mst::core::consider_candidate(best[left.index()], edge, edge_index);
      mst::core::consider_candidate(best[right.index()], edge, edge_index);
    }

    // Contrazione: ogni candidato prova a fondere le sue componenti;
    // `changed` resta falso se non si ammette nulla, e allora ci si ferma.
    bool changed = false;
    for (const auto &candidate : best) {
      if (!candidate) {
        continue;
      }
      if (auto admitted = dsu.unite(*candidate)) {
        output.edges.push_back(*admitted);
        output.total_weight += admitted->value.weight.value();
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  return output;
}

/// Confronta il backend col riferimento sequenziale su peso totale e
/// numero di archi (non sull'identità arco-per-arco: con pesi a parità
/// possono esistere più MST validi e diversi, ma tutti corretti).
inline verification_result verify_against_sequential_cpu(
    const mst::core::validated_graph &graph,
    const std::vector<mst::core::mst_edge> &actual_edges,
    int actual_total_weight) {
  const result expected = sequential_cpu_mst(graph);
  const bool weights_match = expected.total_weight == actual_total_weight;
  const bool edge_counts_match = expected.edges.size() == actual_edges.size();

  return verification_result{
      weights_match && edge_counts_match,
      expected.total_weight,
      expected.edges.size(),
      actual_total_weight,
      actual_edges.size(),
  };
}

/// Scrive l'esito della verifica nel report JSON.
inline void write_verification_json(std::ostream &out,
                                    verification_result verification) {
  out << "  \"verification\": {\n";
  out << "    \"sequential_cpu_success\": "
      << (verification.success ? "true" : "false") << ",\n";
  out << "    \"expected_total_weight\": "
      << verification.expected_total_weight << ",\n";
  out << "    \"actual_total_weight\": "
      << verification.actual_total_weight << ",\n";
  out << "    \"expected_edge_count\": "
      << verification.expected_edge_count << ",\n";
  out << "    \"actual_edge_count\": " << verification.actual_edge_count
      << "\n";
  out << "  }";
}

} // namespace mst::boruvka
