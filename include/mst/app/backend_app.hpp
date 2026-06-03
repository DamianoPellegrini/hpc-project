#pragma once

#include <filesystem>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "mst/app/config.hpp"
#include "mst/app/graph_selection.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/summary.hpp"
#include "mst/reporting/json_report.hpp"
#include "mst/visualization/render_graph.hpp"

namespace mst::app {

struct loaded_graph {
  selected_graph selected;
  mst::core::validated_graph graph;
};

inline loaded_graph load_graph(const app_config &config) {
  selected_graph selected = select_graph(config.graph);
  mst::core::validated_graph graph = mst::core::validate(selected.graph);
  return loaded_graph{std::move(selected), std::move(graph)};
}

inline void print_result(std::string_view header, const app_config &config,
                         const mst::core::validated_graph &graph,
                         const std::vector<mst::core::mst_edge> &edges,
                         int total_weight,
                         std::ostream &out = std::cout) {
  out << header << "\n";
  out << mst::core::mst_summary(edges, total_weight);
  if (should_render_graph(config)) {
    mst::visualization::render_graph_with_mst(graph, edges, total_weight, out);
  }
}

inline mst::boruvka::verification_result verify_and_print(
    const mst::core::validated_graph &graph,
    const std::vector<mst::core::mst_edge> &edges, int total_weight,
    std::ostream &out = std::cout, std::ostream &err = std::cerr) {
  const mst::boruvka::verification_result verification =
      mst::boruvka::verify_against_sequential_cpu(graph, edges, total_weight);
  if (verification.success) {
    out << "Sequential CPU verification: passed\n";
  } else {
    err << "Sequential CPU verification failed: expected weight "
        << verification.expected_total_weight << " with "
        << verification.expected_edge_count << " edges, got weight "
        << verification.actual_total_weight << " with "
        << verification.actual_edge_count << " edges\n";
  }
  return verification;
}

inline std::string configuration_metadata_json(const app_config &config) {
  std::ostringstream out;
  out << "  \"configuration\": {\n";
  out << "    \"render_requested\": ";
  if (config.render_graph) {
    out << (*config.render_graph ? "true" : "false");
  } else {
    out << "null";
  }
  out << ",\n";
  out << "    \"render_enabled\": "
      << (should_render_graph(config) ? "true" : "false") << ",\n";
  out << "    \"cuda_host_memory\": \""
      << cuda_host_memory_mode_name(config.cuda_host_memory) << "\"\n";
  out << "  }";
  return out.str();
}

inline std::string mst_metadata_json(const mst::core::validated_graph &graph,
                                     std::size_t selected_edge_count,
                                     int rounds, int total_weight) {
  std::ostringstream out;
  out << "  \"mst\": {\n";
  out << "    \"vertex_count\": " << graph.vertex_count() << ",\n";
  out << "    \"input_edge_count\": " << graph.edges().size() << ",\n";
  out << "    \"selected_edge_count\": " << selected_edge_count << ",\n";
  out << "    \"rounds\": " << rounds << ",\n";
  out << "    \"total_weight\": " << total_weight << "\n";
  out << "  }";
  return out.str();
}

inline bool write_report_if_requested(const app_config &config,
                                      std::string_view report) {
  if (config.report_path.empty()) {
    return false;
  }
  return mst::reporting::write_report(config.report_path, report);
}

inline bool handle_config_parse_result(const config_parse_result &parsed,
                                       std::string_view executable,
                                       int &exit_code,
                                       std::ostream &out = std::cout,
                                       std::ostream &err = std::cerr) {
  if (parsed.help_requested) {
    out << usage(executable);
    exit_code = 0;
    return false;
  }
  if (!parsed.success) {
    err << parsed.error << "\n\n" << usage(executable);
    exit_code = 1;
    return false;
  }
  return true;
}

} // namespace mst::app
