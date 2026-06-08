#pragma once

#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "mst/app/config.hpp"
#include "mst/core/graph_examples.hpp"
#include "mst/reporting/json_report.hpp"

namespace mst::app {

/// Grafo grezzo selezionato più i metadati della scelta (sorgente, nome,
/// eventuale config casuale): servono a riportarla fedelmente nei log e nel report.
struct selected_graph {
  graph_source_kind source;
  std::string name;
  mst::core::raw_graph graph;
  std::optional<mst::core::random_connected_graph_config> random_config;
};

/// Smista per nome verso una factory del catalogo, oppure genera un grafo
/// casuale per "random"; un nome sconosciuto ricade sul grafo di test condiviso.
inline selected_graph select_graph_by_name(
    std::string_view name,
    std::optional<mst::core::random_connected_graph_config> random_config) {
  if (name == "triangle") {
    return {graph_source_kind::named_example, "triangle",
            mst::core::make_tiny_triangle_graph(), std::nullopt};
  }
  if (name == "square") {
    return {graph_source_kind::named_example, "square",
            mst::core::make_square_with_diagonal_graph(),
            std::nullopt};
  }
  if (name == "tie") {
    return {graph_source_kind::named_example, "tie",
            mst::core::make_equal_weight_tie_graph(), std::nullopt};
  }
  if (name == "dense16") {
    return {graph_source_kind::named_example, "dense16",
            mst::core::make_dense_16_vertex_graph(), std::nullopt};
  }
  if (name == "random") {
    const auto config =
        random_config.value_or(mst::core::default_large_random_graph_config());
    return {graph_source_kind::random_generated, "random",
            mst::core::make_random_connected_graph(config), config};
  }
  return {graph_source_kind::named_example, "test",
          mst::core::make_sparse_12_vertex_graph(), std::nullopt};
}

/// Spacchetta la `graph_config` e delega a `select_graph_by_name`.
inline selected_graph select_graph(const graph_config &config) {
  return select_graph_by_name(config.name, config.random_config);
}

/// Frammento JSON coi metadati del grafo selezionato (nome, sorgente, dimensioni, parametri se casuale).
inline std::string graph_metadata_json(const selected_graph &selected) {
  std::ostringstream out;
  out << "  \"graph\": {\n";
  out << "    \"name\": \""
      << mst::reporting::json_escape(selected.name) << "\",\n";
  out << "    \"source\": \""
      << graph_source_kind_name(selected.source) << "\",\n";
  out << "    \"vertex_count\": " << selected.graph.vertex_count() << ",\n";
  out << "    \"edge_count\": " << selected.graph.edges().size();
  if (selected.random_config) {
    out << ",\n";
    out << "    \"random_vertices\": "
        << selected.random_config->vertices().value() << ",\n";
    out << "    \"random_extra_edges\": "
        << selected.random_config->extra_edges().value() << ",\n";
    out << "    \"seed\": " << selected.random_config->seed().value() << ",\n";
    out << "    \"max_weight\": "
        << selected.random_config->max_weight().value();
  }
  out << "\n";
  out << "  }";
  return out.str();
}

} // namespace mst::app
