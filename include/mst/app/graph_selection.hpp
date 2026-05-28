#pragma once

#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "mst/core/graph_examples.hpp"
#include "mst/reporting/json_report.hpp"

namespace mst::app {

struct selected_graph {
  std::string name;
  mst::core::raw_graph graph;
  std::optional<mst::core::random_connected_graph_config> random_config;
};

inline std::optional<int> parse_int_env(const char *name) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') {
    return std::nullopt;
  }
  char *end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return std::nullopt;
  }
  return static_cast<int>(value);
}

inline std::optional<std::size_t> parse_size_env(const char *name) {
  if (const auto parsed = parse_int_env(name)) {
    if (*parsed >= 0) {
      return static_cast<std::size_t>(*parsed);
    }
  }
  return std::nullopt;
}

inline mst::core::random_connected_graph_config random_config_from_env() {
  const auto defaults = mst::core::default_large_random_graph_config();
  const int vertices =
      parse_int_env("MST_RANDOM_VERTICES").value_or(defaults.vertices().value());
  const std::size_t extra_edges =
      parse_size_env("MST_RANDOM_EXTRA_EDGES")
          .value_or(defaults.extra_edges().value());
  const int seed =
      parse_int_env("MST_RANDOM_SEED")
          .value_or(static_cast<int>(defaults.seed().value()));
  const int max_weight =
      parse_int_env("MST_RANDOM_MAX_WEIGHT")
          .value_or(defaults.max_weight().value());

  const auto config = mst::core::random_connected_graph_config::create(
      mst::core::make_graph_vertex_count(vertices),
      mst::core::make_edge_count(extra_edges),
      mst::core::make_random_seed(static_cast<std::uint64_t>(seed)),
      mst::core::make_edge_weight(max_weight));
  return config.value_or(defaults);
}

inline selected_graph select_graph_by_name(std::string_view name) {
  if (name == "triangle") {
    return {"triangle", mst::core::make_tiny_triangle_graph(), std::nullopt};
  }
  if (name == "square") {
    return {"square", mst::core::make_square_with_diagonal_graph(),
            std::nullopt};
  }
  if (name == "tie") {
    return {"tie", mst::core::make_equal_weight_tie_graph(), std::nullopt};
  }
  if (name == "dense16") {
    return {"dense16", mst::core::make_dense_16_vertex_graph(), std::nullopt};
  }
  if (name == "random") {
    const auto config = random_config_from_env();
    return {"random", mst::core::make_random_connected_graph(config), config};
  }
  return {"test", mst::core::make_sparse_12_vertex_graph(), std::nullopt};
}

inline selected_graph select_graph_from_env() {
  const char *name = std::getenv("MST_GRAPH");
  if (name == nullptr || *name == '\0') {
    return select_graph_by_name("test");
  }
  return select_graph_by_name(name);
}

inline std::string graph_metadata_json(const selected_graph &selected) {
  std::ostringstream out;
  out << "  \"graph\": {\n";
  out << "    \"name\": \""
      << mst::reporting::json_escape(selected.name) << "\",\n";
  out << "    \"vertex_count\": " << selected.graph.vertex_count() << ",\n";
  out << "    \"edge_count\": " << selected.graph.edges().size();
  if (selected.random_config) {
    out << ",\n";
    out << "    \"seed\": " << selected.random_config->seed().value() << ",\n";
    out << "    \"max_weight\": "
        << selected.random_config->max_weight().value();
  }
  out << "\n";
  out << "  }";
  return out.str();
}

} // namespace mst::app
