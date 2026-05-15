#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "mst/core/edge.hpp"

namespace mst::core {

/// Render a stable textual summary of the admitted MST edges.
inline std::string mst_summary(const std::vector<mst_edge> &edges,
                               int total_weight) {
  std::vector<mst_edge> sorted = edges;
  std::sort(sorted.begin(), sorted.end(),
            [](const mst_edge &left, const mst_edge &right) {
              const auto left_endpoints = normalized_endpoints(left.value);
              const auto right_endpoints = normalized_endpoints(right.value);
              if (left_endpoints != right_endpoints) {
                return left_endpoints < right_endpoints;
              }
              return left.value.weight < right.value.weight;
            });

  std::ostringstream out;
  out << "MST weight = " << total_weight << ", edges = " << sorted.size()
      << '\n';
  for (const mst_edge &edge_value : sorted) {
    out << "  " << edge_value.value.u.value() << "-"
        << edge_value.value.v.value() << " ("
        << edge_value.value.weight.value() << ")\n";
  }
  return out.str();
}

} // namespace mst::core
