#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "mst/app/backend_app.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/dsu/parallel_disjoint_set.hpp"
#include "mst/reporting/json_report.hpp"

namespace mst::backend::openmp {

struct openmp_profile {
  double scan_seconds = 0.0;
  double reduce_seconds = 0.0;
  double contract_seconds = 0.0;
  double compress_seconds = 0.0;
};

struct openmp_workspace {
  openmp_workspace(int vertex_count_value, int thread_capacity_value)
      : vertex_count(vertex_count_value),
        thread_capacity(thread_capacity_value),
        local_keys_by_thread(
            static_cast<std::size_t>(thread_capacity),
            std::vector<std::uint64_t>(
                static_cast<std::size_t>(vertex_count),
                mst::core::empty_candidate_key.value())),
        best_keys(static_cast<std::size_t>(vertex_count),
                  mst::core::empty_candidate_key.value()),
        local_edges(static_cast<std::size_t>(thread_capacity)),
        local_weights(static_cast<std::size_t>(thread_capacity), 0) {}

  void reset_candidates() {
    for (auto &local : local_keys_by_thread) {
      std::fill(local.begin(), local.end(),
                mst::core::empty_candidate_key.value());
    }
    std::fill(best_keys.begin(), best_keys.end(),
              mst::core::empty_candidate_key.value());
  }

  void reset_contractions() {
    for (auto &edges : local_edges) {
      edges.clear();
    }
    std::fill(local_weights.begin(), local_weights.end(), 0);
  }

  int vertex_count;
  int thread_capacity;
  std::vector<std::vector<std::uint64_t>> local_keys_by_thread;
  std::vector<std::uint64_t> best_keys;
  std::vector<std::vector<mst::core::mst_edge>> local_edges;
  std::vector<int> local_weights;
};

const std::vector<std::uint64_t> &
local_best_candidate_keys(const mst::core::validated_graph &graph,
                          const mst::dsu::parent_snapshot &snapshot,
                          openmp_workspace &workspace,
                          openmp_profile &profile) {
  using clock = std::chrono::steady_clock;

  workspace.reset_candidates();

  const auto scan_start = clock::now();
#pragma omp parallel
  {
    const int thread = omp_get_thread_num();
    auto &local =
        workspace.local_keys_by_thread[static_cast<std::size_t>(thread)];

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
      const std::uint64_t key =
          mst::core::make_candidate_key(edge.weight, edge_index).value();
      auto &left_candidate = local[left_root.index()];
      auto &right_candidate = local[right_root.index()];
      left_candidate = std::min(left_candidate, key);
      right_candidate = std::min(right_candidate, key);
    }
  }
  profile.scan_seconds +=
      std::chrono::duration<double>(clock::now() - scan_start).count();

  const auto reduce_start = clock::now();

#pragma omp parallel for
  for (int component = 0; component < workspace.vertex_count; ++component) {
    std::uint64_t global = mst::core::empty_candidate_key.value();
    for (int thread = 0; thread < workspace.thread_capacity; ++thread) {
      const std::uint64_t candidate =
          workspace.local_keys_by_thread[static_cast<std::size_t>(thread)]
                                        [static_cast<std::size_t>(component)];
      global = std::min(global, candidate);
    }
    workspace.best_keys[static_cast<std::size_t>(component)] = global;
  }
  profile.reduce_seconds +=
      std::chrono::duration<double>(clock::now() - reduce_start).count();

  return workspace.best_keys;
}

int apply_contractions_parallel(
    const std::vector<std::uint64_t> &best_keys,
    const mst::core::validated_graph &graph,
    mst::dsu::parallel_disjoint_set &dsu,
    std::vector<mst::core::mst_edge> &mst_edges, int &total_weight,
    openmp_workspace &workspace,
    openmp_profile &profile) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  workspace.reset_contractions();
  const int candidate_count = static_cast<int>(best_keys.size());
  int admitted_count = 0;

#pragma omp parallel reduction(+ : admitted_count)
  {
    const int thread = omp_get_thread_num();
    auto &edges = workspace.local_edges[static_cast<std::size_t>(thread)];
    int &weight = workspace.local_weights[static_cast<std::size_t>(thread)];

#pragma omp for
    for (int index = 0; index < candidate_count; ++index) {
      const std::uint64_t key = best_keys[static_cast<std::size_t>(index)];
      if (key == mst::core::empty_candidate_key.value()) {
        continue;
      }
      const mst::core::candidate_key candidate_key{key};
      const mst::core::edge_index edge_index =
          mst::core::edge_index_from_candidate_key(candidate_key);
      const mst::core::candidate_edge candidate{
          graph.edges()[static_cast<std::size_t>(edge_index.value())],
          edge_index};
      if (auto admitted = dsu.unite(candidate)) {
        edges.push_back(*admitted);
        weight += admitted->value.weight.value();
        ++admitted_count;
      }
    }
  }

  for (int thread = 0; thread < workspace.thread_capacity; ++thread) {
    total_weight += workspace.local_weights[static_cast<std::size_t>(thread)];
    auto &edges = workspace.local_edges[static_cast<std::size_t>(thread)];
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

int main(int argc, char **argv) {
  using namespace mst::backend::openmp;
  using clock = std::chrono::steady_clock;

  const mst::app::config_parse_result parsed =
      mst::app::parse_app_config(argc, argv);
  int config_exit_code = EXIT_FAILURE;
  if (!mst::app::handle_config_parse_result(parsed, argv[0],
                                            config_exit_code)) {
    return config_exit_code;
  }
  const mst::app::app_config &config = parsed.config;

  const auto total_start = clock::now();

  const mst::app::loaded_graph loaded = mst::app::load_graph(config);
  const mst::app::selected_graph &selected = loaded.selected;
  const mst::core::validated_graph &graph = loaded.graph;
  mst::dsu::parallel_disjoint_set dsu(graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;
  int rounds = 0;
  int remaining_components = graph.vertex_count();
  openmp_profile profile;
  openmp_workspace workspace(graph.vertex_count(), omp_get_max_threads());

  const auto mst_start = clock::now();
  while (remaining_components > 1) {
    ++rounds;
    const mst::dsu::parent_snapshot snapshot = dsu.snapshot();
    const std::vector<std::uint64_t> &best =
        local_best_candidate_keys(graph, snapshot, workspace, profile);
    const int admitted_count = apply_contractions_parallel(
        best, graph, dsu, mst_edges, total_weight, workspace, profile);
    remaining_components -= admitted_count;
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

  mst::app::print_result(
      std::string("OpenMP Boruvka MST using ") +
          std::to_string(omp_get_max_threads()) + " threads",
      config, graph, mst_edges, total_weight);
  const auto verification_start = clock::now();
  const mst::boruvka::verification_result verification =
      mst::app::verify_and_print(graph, mst_edges, total_weight);
  const double verification_seconds =
      std::chrono::duration<double>(clock::now() - verification_start).count();

  std::ostringstream report;
  report << "{\n";
  report << mst::reporting::common_metadata_json("openmp",
                                                 verification.success)
         << ",\n";
  report << mst::app::graph_metadata_json(selected) << ",\n";
  report << mst::app::configuration_metadata_json(config) << ",\n";
  mst::reporting::write_phase_timings_json(
      report, mst::reporting::phase_timing_profile{
                  total_seconds,
                  mst_loop_seconds,
                  verification_seconds,
                  profile.scan_seconds,
                  profile.reduce_seconds,
                  profile.contract_seconds,
                  profile.compress_seconds,
              });
  report << ",\n";
  report << "  \"capabilities\": {\n";
  report << "    \"max_threads\": " << omp_get_max_threads() << ",\n";
  report << "    \"num_procs\": " << omp_get_num_procs() << ",\n";
  report << "    \"dynamic_enabled\": "
         << (omp_get_dynamic() ? "true" : "false") << ",\n";
  report << "    \"max_active_levels\": " << omp_get_max_active_levels()
         << "\n";
  report << "  },\n";
  report << mst::app::mst_metadata_json(graph, mst_edges.size(), rounds,
                                        total_weight)
         << ",\n";
  mst::boruvka::write_verification_json(report, verification);
  report << "\n";
  report << "}\n";
  mst::app::write_report_if_requested(config, report.str());
  return verification.success ? 0 : 1;
}
