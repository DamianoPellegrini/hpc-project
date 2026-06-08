#pragma once

#include <cassert>
#include <utility>
#include <vector>

#include "mst/core/edge.hpp"

namespace mst::core {

template <class state_t> class graph {
public:
  /// Costruisce un grafo nel typestate dato (`raw` o `validated`).
  graph(int vertex_count, std::vector<edge> edges)
      : vertex_count_(vertex_count), edges_(std::move(edges)) {}

  int vertex_count() const noexcept { return vertex_count_; }
  const std::vector<edge> &edges() const noexcept { return edges_; }

private:
  int vertex_count_;
  std::vector<edge> edges_;
};

using raw_graph = graph<raw>;
using validated_graph = graph<validated>;

/// Vero se ogni estremo è in [0, vertex_count): la condizione per passare da `raw_graph` a `validated_graph`.
inline bool has_valid_vertex_ids(const raw_graph &graph_value) {
  if (graph_value.vertex_count() <= 0) {
    return false;
  }
  for (const edge &edge_value : graph_value.edges()) {
    if (edge_value.u.value() < 0 || edge_value.v.value() < 0) {
      return false;
    }
    if (edge_value.u.value() >= graph_value.vertex_count() ||
        edge_value.v.value() >= graph_value.vertex_count()) {
      return false;
    }
  }
  return true;
}

/// Promuove un grafo a `validated`. L'`assert` cattura in debug i vertici
/// fuori range; in release l'invariante resta a carico di chi costruisce il grafo.
inline validated_graph validate(raw_graph graph_value) {
  assert(has_valid_vertex_ids(graph_value) &&
         "graph contains vertex IDs outside [0, vertex_count)");
  return validated_graph{graph_value.vertex_count(), graph_value.edges()};
}

/// Grafo di prova condiviso (12 vertici, pesi noti): tutti i backend lo
/// usano per confrontarsi fra loro e col riferimento sequenziale.
inline raw_graph make_test_graph() {
  return raw_graph{
      12,
      {
          {make_vertex_id(0), make_vertex_id(1), make_edge_weight(4)},
          {make_vertex_id(0), make_vertex_id(2), make_edge_weight(3)},
          {make_vertex_id(0), make_vertex_id(3), make_edge_weight(8)},
          {make_vertex_id(1), make_vertex_id(2), make_edge_weight(1)},
          {make_vertex_id(1), make_vertex_id(4), make_edge_weight(7)},
          {make_vertex_id(1), make_vertex_id(5), make_edge_weight(9)},
          {make_vertex_id(2), make_vertex_id(4), make_edge_weight(2)},
          {make_vertex_id(2), make_vertex_id(5), make_edge_weight(6)},
          {make_vertex_id(2), make_vertex_id(6), make_edge_weight(7)},
          {make_vertex_id(3), make_vertex_id(5), make_edge_weight(5)},
          {make_vertex_id(3), make_vertex_id(6), make_edge_weight(4)},
          {make_vertex_id(3), make_vertex_id(7), make_edge_weight(9)},
          {make_vertex_id(4), make_vertex_id(5), make_edge_weight(3)},
          {make_vertex_id(4), make_vertex_id(7), make_edge_weight(8)},
          {make_vertex_id(4), make_vertex_id(8), make_edge_weight(6)},
          {make_vertex_id(5), make_vertex_id(7), make_edge_weight(4)},
          {make_vertex_id(5), make_vertex_id(8), make_edge_weight(2)},
          {make_vertex_id(6), make_vertex_id(8), make_edge_weight(7)},
          {make_vertex_id(6), make_vertex_id(9), make_edge_weight(6)},
          {make_vertex_id(7), make_vertex_id(8), make_edge_weight(1)},
          {make_vertex_id(7), make_vertex_id(10), make_edge_weight(5)},
          {make_vertex_id(8), make_vertex_id(10), make_edge_weight(3)},
          {make_vertex_id(8), make_vertex_id(11), make_edge_weight(9)},
          {make_vertex_id(9), make_vertex_id(11), make_edge_weight(2)},
          {make_vertex_id(10), make_vertex_id(11), make_edge_weight(4)},
          {make_vertex_id(1), make_vertex_id(3), make_edge_weight(9)},
          {make_vertex_id(4), make_vertex_id(9), make_edge_weight(10)},
          {make_vertex_id(5), make_vertex_id(10), make_edge_weight(7)},
          {make_vertex_id(6), make_vertex_id(11), make_edge_weight(8)},
      },
  };
}

} // namespace mst::core
