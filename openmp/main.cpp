#include <omp.h>

#include <chrono>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>

#include "mst/app/graph_selection.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/summary.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/dsu/parallel_disjoint_set.hpp"
#include "mst/reporting/json_report.hpp"
#include "mst/visualization/render_graph.hpp"

namespace mst::backend::openmp {

struct openmp_profile {
  double scan_seconds = 0.0;
  double reduce_seconds = 0.0;
  double contract_seconds = 0.0;
  double compress_seconds = 0.0;
};

std::vector<mst::core::maybe_candidate_edge>
local_best_candidates(const mst::core::validated_graph &graph,
                      const mst::dsu::parent_snapshot &snapshot,
                      openmp_profile &profile) {
  using clock = std::chrono::steady_clock;

  const auto scan_start = clock::now();
  const int vertex_count = graph.vertex_count();
  const int thread_capacity = omp_get_max_threads();
  std::vector<std::vector<mst::core::maybe_candidate_edge>> local_by_thread(
      static_cast<std::size_t>(thread_capacity),
      std::vector<mst::core::maybe_candidate_edge>(
          static_cast<std::size_t>(vertex_count)));

#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    auto &local = local_by_thread[static_cast<std::size_t>(thread)];

#pragma omp for
    for (std::size_t index = 0; index < graph.edges().size(); ++index) {
      const mst::core::edge &edge = graph.edges()[index];
      const mst::core::vertex_id left_root =
          mst::dsu::find_root(snapshot, edge.u);
      const mst::core::vertex_id right_root =
          mst::dsu::find_root(snapshot, edge.v);
      if (left_root == right_root) {
        continue;
      }

      const mst::core::edge_index edge_index =
          mst::core::make_edge_index(index);
      mst::core::consider_candidate(local[left_root.index()], edge,
                                    edge_index);
      mst::core::consider_candidate(local[right_root.index()], edge,
                                    edge_index);
    }
  }
  profile.scan_seconds +=
      std::chrono::duration<double>(clock::now() - scan_start).count();

  const auto reduce_start = clock::now();
  std::vector<mst::core::maybe_candidate_edge> best(
      static_cast<std::size_t>(vertex_count));

#pragma omp parallel for
  for (int component = 0; component < vertex_count; ++component) {
    auto &global = best[static_cast<std::size_t>(component)];
    for (int thread = 0; thread < thread_capacity; ++thread) {
      const auto &candidate =
          local_by_thread[static_cast<std::size_t>(thread)]
                         [static_cast<std::size_t>(component)];
        if (mst::core::better_candidate(candidate, global)) {
          global = candidate;
        }
    }
  }
  profile.reduce_seconds +=
      std::chrono::duration<double>(clock::now() - reduce_start).count();

  return best;
}

int apply_contractions_parallel(
    const std::vector<mst::core::maybe_candidate_edge> &best,
    mst::dsu::parallel_disjoint_set &dsu,
    std::vector<mst::core::mst_edge> &mst_edges, int &total_weight,
    openmp_profile &profile) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  const int thread_capacity = omp_get_max_threads();
  const int candidate_count = static_cast<int>(best.size());
  std::vector<std::vector<mst::core::mst_edge>> local_edges(
      static_cast<std::size_t>(thread_capacity));
  std::vector<int> local_weights(static_cast<std::size_t>(thread_capacity), 0);
  int admitted_count = 0;

#pragma omp parallel reduction(+ : admitted_count)
  {
    const int thread = omp_get_thread_num();
    auto &edges = local_edges[static_cast<std::size_t>(thread)];
    int &weight = local_weights[static_cast<std::size_t>(thread)];

#pragma omp for
    for (int index = 0; index < candidate_count; ++index) {
      const auto &candidate = best[static_cast<std::size_t>(index)];
      if (!candidate) {
        continue;
      }
      if (auto admitted = dsu.unite(*candidate)) {
        edges.push_back(*admitted);
        weight += admitted->value.weight.value();
        ++admitted_count;
      }
    }
  }

  for (int thread = 0; thread < thread_capacity; ++thread) {
    total_weight += local_weights[static_cast<std::size_t>(thread)];
    auto &edges = local_edges[static_cast<std::size_t>(thread)];
    mst_edges.insert(mst_edges.end(), edges.begin(), edges.end());
  }
  profile.contract_seconds +=
      std::chrono::duration<double>(clock::now() - start).count();
  return admitted_count;
}

void compress_all_parallel(mst::dsu::parallel_disjoint_set &dsu,
                           int vertex_count, openmp_profile &profile) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
#pragma omp parallel for
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    dsu.compress_vertex(mst::core::make_vertex_id(vertex));
  }
  profile.compress_seconds +=
      std::chrono::duration<double>(clock::now() - start).count();
}

} // namespace mst::backend::openmp

int main() {
  using namespace mst::backend::openmp;
  using clock = std::chrono::steady_clock;

  const auto total_start = clock::now();

  const mst::app::selected_graph selected = mst::app::select_graph_from_env();
  const mst::core::validated_graph graph =
      mst::core::validate(selected.graph);
  mst::dsu::parallel_disjoint_set dsu(graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;
  int rounds = 0;
  openmp_profile profile;

  const auto mst_start = clock::now();
  while (dsu.component_count() > 1) {
    ++rounds;
    const mst::dsu::parent_snapshot snapshot = dsu.snapshot();
    const std::vector<mst::core::maybe_candidate_edge> best =
        local_best_candidates(graph, snapshot, profile);
    const int admitted_count = apply_contractions_parallel(
        best, dsu, mst_edges, total_weight, profile);
    compress_all_parallel(dsu, graph.vertex_count(), profile);

    if (admitted_count == 0) {
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
  const auto verification_start = clock::now();
  const mst::boruvka::verification_result verification =
      mst::boruvka::verify_against_sequential_cpu(graph, mst_edges,
                                                  total_weight);
  const double verification_seconds =
      std::chrono::duration<double>(clock::now() - verification_start).count();
  if (verification.success) {
    std::cout << "Sequential CPU verification: passed\n";
  } else {
    std::cerr << "Sequential CPU verification failed: expected weight "
              << verification.expected_total_weight << " with "
              << verification.expected_edge_count << " edges, got weight "
              << verification.actual_total_weight << " with "
              << verification.actual_edge_count << " edges\n";
  }

  std::ostringstream report;
  report << "{\n";
  report << mst::reporting::common_metadata_json("openmp",
                                                 verification.success)
         << ",\n";
  report << mst::app::graph_metadata_json(selected) << ",\n";
  report << "  \"timings\": {\n";
  report << "    \"total_seconds\": " << total_seconds << ",\n";
  report << "    \"mst_loop_seconds\": " << mst_loop_seconds << ",\n";
  report << "    \"sequential_cpu_verification_seconds\": "
         << verification_seconds << ",\n";
  report << "    \"scan_seconds\": " << profile.scan_seconds << ",\n";
  report << "    \"reduce_seconds\": " << profile.reduce_seconds << ",\n";
  report << "    \"contract_seconds\": " << profile.contract_seconds << ",\n";
  report << "    \"compress_seconds\": " << profile.compress_seconds << "\n";
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
  report << "  },\n";
  mst::boruvka::write_verification_json(report, verification);
  report << "\n";
  report << "}\n";
  mst::reporting::write_report_from_env(report.str());
  return verification.success ? 0 : 1;
}
