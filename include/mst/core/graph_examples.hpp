#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

#include "mst/core/graph.hpp"

namespace mst::core {

/// Parametri validati per un grafo casuale connesso: costruttore privato,
/// si passa per `create` che controlla i vincoli (vertici, peso, archi extra
/// disponibili) e ritorna `std::nullopt` se non tornano.
class random_connected_graph_config {
public:
  /// Valida i parametri e costruisce la config, o `std::nullopt` se non vanno bene.
  static std::optional<random_connected_graph_config>
  create(graph_vertex_count vertices, edge_count extra_edges,
         random_seed seed, edge_weight max_weight) {
    if (vertices.value() < 2 || max_weight.value() <= 0) {
      return std::nullopt;
    }

    const auto vertex_count = static_cast<std::size_t>(vertices.value());
    const std::size_t tree_edge_count = vertex_count - 1;
    const std::size_t complete_edge_count = vertex_count * (vertex_count - 1) / 2;
    const std::size_t available_extra_edges =
        complete_edge_count - tree_edge_count;
    if (extra_edges.value() > available_extra_edges) {
      return std::nullopt;
    }

    return random_connected_graph_config{vertices, extra_edges, seed,
                                         max_weight};
  }

  graph_vertex_count vertices() const noexcept { return vertices_; }
  edge_count extra_edges() const noexcept { return extra_edges_; }
  random_seed seed() const noexcept { return seed_; }
  edge_weight max_weight() const noexcept { return max_weight_; }

private:
  random_connected_graph_config(graph_vertex_count vertices,
                                edge_count extra_edges, random_seed seed,
                                edge_weight max_weight)
      : vertices_(vertices), extra_edges_(extra_edges), seed_(seed),
        max_weight_(max_weight) {}

  graph_vertex_count vertices_;
  edge_count extra_edges_;
  random_seed seed_;
  edge_weight max_weight_;
};

/// Triangolo a 3 vertici: caso minuscolo, verificabile a mano.
inline raw_graph make_tiny_triangle_graph() {
  return raw_graph{
      3,
      {
          {make_vertex_id(0), make_vertex_id(1), make_edge_weight(3)},
          {make_vertex_id(1), make_vertex_id(2), make_edge_weight(2)},
          {make_vertex_id(0), make_vertex_id(2), make_edge_weight(5)},
      },
  };
}

/// Quadrato con diagonale: un ciclo e una "scorciatoia" di peso minimo, per testare la scelta fra archi concorrenti.
inline raw_graph make_square_with_diagonal_graph() {
  return raw_graph{
      4,
      {
          {make_vertex_id(0), make_vertex_id(1), make_edge_weight(1)},
          {make_vertex_id(1), make_vertex_id(2), make_edge_weight(2)},
          {make_vertex_id(2), make_vertex_id(3), make_edge_weight(3)},
          {make_vertex_id(3), make_vertex_id(0), make_edge_weight(4)},
          {make_vertex_id(0), make_vertex_id(2), make_edge_weight(1)},
      },
  };
}

/// Pesi tutti uguali apposta: verifica che ogni backend risolva i pareggi allo stesso modo (vince l'indice più basso).
inline raw_graph make_equal_weight_tie_graph() {
  return raw_graph{
      5,
      {
          {make_vertex_id(0), make_vertex_id(1), make_edge_weight(1)},
          {make_vertex_id(0), make_vertex_id(2), make_edge_weight(1)},
          {make_vertex_id(1), make_vertex_id(3), make_edge_weight(1)},
          {make_vertex_id(2), make_vertex_id(4), make_edge_weight(1)},
          {make_vertex_id(3), make_vertex_id(4), make_edge_weight(2)},
      },
  };
}

/// Alias del grafo di test condiviso, come esempio "sparso" nel catalogo.
inline raw_graph make_sparse_12_vertex_graph() { return make_test_graph(); }

/// Grafo denso a 16 vertici: un anello più archi a distanza 2-4, tutto
/// deterministico (niente generatore casuale).
inline raw_graph make_dense_16_vertex_graph() {
  std::vector<edge> edges;
  edges.reserve(40);
  for (int vertex = 0; vertex < 16; ++vertex) {
    const int next = (vertex + 1) % 16;
    edges.push_back({make_vertex_id(vertex), make_vertex_id(next),
                     make_edge_weight(1 + (vertex % 7))});
  }
  for (int vertex = 0; vertex < 16; ++vertex) {
    for (int step = 2; step <= 4; ++step) {
      const int other = (vertex + step) % 16;
      if (vertex < other) {
        edges.push_back({make_vertex_id(vertex), make_vertex_id(other),
                         make_edge_weight(3 + ((vertex + other) % 11))});
      }
    }
  }
  return raw_graph{16, std::move(edges)};
}

/// Coppia di vertici (normalizzata min, max) impacchettata in una chiave a
/// 64 bit: comoda per buttarla in un `unordered_set` e scartare i duplicati.
inline std::uint64_t encoded_undirected_edge_key(int left, int right) noexcept {
  const auto endpoints = std::minmax(left, right);
  return (static_cast<std::uint64_t>(
              static_cast<std::uint32_t>(endpoints.first)) << 32) |
         static_cast<std::uint64_t>(
             static_cast<std::uint32_t>(endpoints.second));
}

/// Config di default per i benchmark su grafo grande: 32768 vertici, ~200k archi extra, seed fisso, pesi fino a 10000.
inline random_connected_graph_config default_large_random_graph_config() {
  return *random_connected_graph_config::create(
      make_graph_vertex_count(32768), make_edge_count(196608),
      make_random_seed(886261), make_edge_weight(10000));
}

/// Grafo casuale ma connesso, riproducibile a parità di seed: prima uno
/// spanning tree casuale (garantisce la connessione), poi `extra_edges`
/// archi in più scartando duplicati e auto-anelli via `encoded_undirected_edge_key`.
inline raw_graph
make_random_connected_graph(const random_connected_graph_config &config) {
  std::mt19937_64 generator(config.seed().value());
  std::uniform_int_distribution<int> weight_dist(1,
                                                 config.max_weight().value());
  std::vector<edge> edges;
  const int vertex_count = config.vertices().value();
  edges.reserve(static_cast<std::size_t>(vertex_count - 1) +
                config.extra_edges().value());

  std::unordered_set<std::uint64_t> used_edges;
  used_edges.reserve(edges.capacity());
  for (int vertex = 1; vertex < vertex_count; ++vertex) {
    const int parent = static_cast<int>(
        std::uniform_int_distribution<int>(0, vertex - 1)(generator));
    const auto endpoints = std::minmax(parent, vertex);
    used_edges.insert(encoded_undirected_edge_key(endpoints.first,
                                                  endpoints.second));
    edges.push_back({make_vertex_id(endpoints.first),
                     make_vertex_id(endpoints.second),
                     make_edge_weight(weight_dist(generator))});
  }

  std::uniform_int_distribution<int> vertex_dist(0, vertex_count - 1);
  while (edges.size() <
         static_cast<std::size_t>(vertex_count - 1) +
             config.extra_edges().value()) {
    int left = vertex_dist(generator);
    int right = vertex_dist(generator);
    if (left == right) {
      continue;
    }
    const auto endpoints = std::minmax(left, right);
    if (!used_edges
             .insert(encoded_undirected_edge_key(endpoints.first,
                                                 endpoints.second))
             .second) {
      continue;
    }
    edges.push_back({make_vertex_id(endpoints.first),
                     make_vertex_id(endpoints.second),
                     make_edge_weight(weight_dist(generator))});
  }

  return raw_graph{vertex_count, std::move(edges)};
}

} // namespace mst::core
