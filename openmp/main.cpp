#include <omp.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/summary.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/reporting/json_report.hpp"
#include "mst/visualization/render_graph.hpp"

namespace mst::backend::openmp {

std::vector<mst::core::maybe_candidate_edge>
local_best_candidates(const mst::core::validated_graph &graph,
                      const mst::dsu::parent_snapshot &snapshot) {
  std::vector<mst::core::maybe_candidate_edge> best(
      static_cast<std::size_t>(graph.vertex_count()));

#pragma omp parallel
  {
    std::vector<mst::core::maybe_candidate_edge> local(
        static_cast<std::size_t>(graph.vertex_count()));

#pragma omp for nowait
    for (std::size_t index = 0; index < graph.edges().size(); ++index) {
      const mst::core::edge &edge = graph.edges()[index];
      const mst::core::vertex_id left_root =
          mst::dsu::find_root(snapshot, edge.u);
      const mst::core::vertex_id right_root =
          mst::dsu::find_root(snapshot, edge.v);
      if (left_root == right_root) {
        continue;
      }

      mst::core::consider_candidate(local[left_root.index()], edge);
      mst::core::consider_candidate(local[right_root.index()], edge);
    }

#pragma omp critical
    {
      for (int component = 0; component < graph.vertex_count(); ++component) {
        auto &global = best[static_cast<std::size_t>(component)];
        const auto &candidate = local[static_cast<std::size_t>(component)];
        if (mst::core::better_candidate(candidate, global)) {
          global = candidate;
        }
      }
    }
  }

  return best;
}

} // namespace mst::backend::openmp

int main() {
  using namespace mst::backend::openmp;
  using clock = std::chrono::steady_clock;

  const auto total_start = clock::now();

  const mst::core::validated_graph graph =
      mst::core::validate(mst::core::make_test_graph());
  mst::dsu::disjoint_set<mst::core::uncompressed_parents> dsu(
      graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;
  int rounds = 0;

  const auto mst_start = clock::now();
  while (dsu.component_count() > 1) {
    ++rounds;
    const mst::dsu::parent_snapshot snapshot = dsu.compressed_snapshot();
    const std::vector<mst::core::maybe_candidate_edge> best =
        local_best_candidates(graph, snapshot);

    bool changed = false;
    for (int component = 0; component < graph.vertex_count(); ++component) {
      const auto &candidate = best[static_cast<std::size_t>(component)];
      if (!candidate) {
        continue;
      }

      if (auto admitted = dsu.unite(*candidate)) {
        total_weight += admitted->value.weight.value();
        mst_edges.push_back(*admitted);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }
  const auto mst_end = clock::now();
  const auto total_end = clock::now();

  const double mst_loop_seconds =
      std::chrono::duration<double>(mst_end - mst_start).count();
  const double total_seconds =
      std::chrono::duration<double>(total_end - total_start).count();

  std::cout << "OpenMP Boruvka MST using " << omp_get_max_threads()
            << " threads\n";
  std::cout << mst::core::mst_summary(mst_edges, total_weight);
  mst::visualization::render_graph_with_mst(graph, mst_edges, total_weight);

  std::ostringstream report;
  report << "{\n";
  report << mst::reporting::common_metadata_json("openmp", true) << ",\n";
  report << "  \"timings\": {\n";
  report << "    \"total_seconds\": " << total_seconds << ",\n";
  report << "    \"mst_loop_seconds\": " << mst_loop_seconds << "\n";
  report << "  },\n";
  report << "  \"capabilities\": {\n";
  report << "    \"max_threads\": " << omp_get_max_threads() << ",\n";
  report << "    \"num_procs\": " << omp_get_num_procs() << ",\n";
  report << "    \"dynamic_enabled\": "
         << (omp_get_dynamic() ? "true" : "false") << ",\n";
  report << "    \"max_active_levels\": " << omp_get_max_active_levels()
         << "\n";
  report << "  },\n";
  report << "  \"mst\": {\n";
  report << "    \"vertex_count\": " << graph.vertex_count() << ",\n";
  report << "    \"input_edge_count\": " << graph.edges().size() << ",\n";
  report << "    \"selected_edge_count\": " << mst_edges.size() << ",\n";
  report << "    \"rounds\": " << rounds << ",\n";
  report << "    \"total_weight\": " << total_weight << "\n";
  report << "  }\n";
  report << "}\n";
  mst::reporting::write_report_from_env(report.str());
  return 0;
}
