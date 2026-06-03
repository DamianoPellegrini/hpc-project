#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include "mst/app/backend_app.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/execution/domain.hpp"
#include "mst/reporting/json_report.hpp"

namespace mst::backend::mpi {

constexpr int root_rank = 0;

struct mpi_workspace {
  explicit mpi_workspace(int vertex_count)
      : local_keys(static_cast<std::size_t>(vertex_count),
                   mst::core::empty_candidate_key.value()),
        reduced_keys(static_cast<std::size_t>(vertex_count),
                     mst::core::empty_candidate_key.value()) {}

  void reset_local() {
    std::fill(local_keys.begin(), local_keys.end(),
              mst::core::empty_candidate_key.value());
  }

  std::vector<std::uint64_t> local_keys;
  std::vector<std::uint64_t> reduced_keys;
};

int edge_begin_for_rank(int edge_count, int rank, int size) {
  return (edge_count * rank) / size;
}

int edge_end_for_rank(int edge_count, int rank, int size) {
  return (edge_count * (rank + 1)) / size;
}

std::vector<int> pack_snapshot(const mst::dsu::parent_snapshot &snapshot) {
  std::vector<int> packed;
  packed.reserve(snapshot.parent().size());
  for (const mst::core::vertex_id vertex : snapshot.parent()) {
    packed.push_back(vertex.value());
  }
  return packed;
}

mst::dsu::parent_snapshot unpack_snapshot(const std::vector<int> &packed) {
  std::vector<mst::core::vertex_id> parent;
  parent.reserve(packed.size());
  for (const int value : packed) {
    parent.push_back(mst::core::make_vertex_id(value));
  }
  return mst::dsu::parent_snapshot{std::move(parent)};
}

void local_best_candidate_keys(
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu, int begin,
    int end, std::vector<std::uint64_t> &best_keys) {
  for (int index = begin; index < end; ++index) {
    const mst::core::edge &edge =
        graph.edges()[static_cast<std::size_t>(index)];
    const mst::core::component_id left_root = dsu.find(edge.u);
    const mst::core::component_id right_root = dsu.find(edge.v);
    if (left_root == right_root) {
      continue;
    }

    const mst::core::edge_index edge_index =
        mst::core::make_edge_index(static_cast<std::size_t>(index));
    const std::uint64_t key =
        mst::core::make_candidate_key(edge.weight, edge_index).value();
    auto &left_candidate = best_keys[left_root.index()];
    auto &right_candidate = best_keys[right_root.index()];
    left_candidate = std::min(left_candidate, key);
    right_candidate = std::min(right_candidate, key);
  }
}

mst::execution::mpi_round<mst::execution::parents_broadcasted>
broadcast_parents(mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu,
                  int root_rank_value, MPI_Comm comm) {
  std::vector<int> packed = pack_snapshot(dsu.compressed_snapshot());
  MPI_Bcast(packed.data(), static_cast<int>(packed.size()), MPI_INT,
            root_rank_value, comm);
  dsu.set_parent_snapshot(unpack_snapshot(packed));
  return {};
}

const std::vector<std::uint64_t> &compute_local_minima(
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu,
    int edge_begin, int edge_end, mpi_workspace &workspace,
    mst::execution::mpi_round<mst::execution::parents_broadcasted>) {
  workspace.reset_local();
  local_best_candidate_keys(graph, dsu, edge_begin, edge_end,
                            workspace.local_keys);
  return workspace.local_keys;
}

const std::vector<std::uint64_t> &reduce_minima(
    const std::vector<std::uint64_t> &local_keys, mpi_workspace &workspace,
    const mst::core::validated_graph &graph, MPI_Comm comm,
    mst::execution::mpi_round<mst::execution::local_minima_computed>) {
  const int vertex_count = graph.vertex_count();
  MPI_Allreduce(local_keys.data(), workspace.reduced_keys.data(), vertex_count,
                MPI_UINT64_T, MPI_MIN, comm);

  return workspace.reduced_keys;
}

int apply_contractions(
    const std::vector<std::uint64_t> &best_keys,
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu,
    std::vector<mst::core::mst_edge> &mst_edges, int &total_weight) {
  int admitted_count = 0;
  for (const std::uint64_t key_value : best_keys) {
    if (key_value == mst::core::empty_candidate_key.value()) {
      continue;
    }
    const mst::core::candidate_key key{key_value};
    const mst::core::edge_index edge_index =
        mst::core::edge_index_from_candidate_key(key);
    const mst::core::candidate_edge candidate{
        graph.edges()[static_cast<std::size_t>(edge_index.value())],
        edge_index};
    if (auto admitted = dsu.unite(candidate)) {
      mst_edges.push_back(*admitted);
      total_weight += admitted->value.weight.value();
      ++admitted_count;
    }
  }
  return admitted_count;
}

} // namespace mst::backend::mpi

int main(std::int32_t argc, char **argv) {
  using namespace mst::backend::mpi;

  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const mst::app::config_parse_result parsed =
      mst::app::parse_app_config(argc, argv);
  if (!parsed.success || parsed.help_requested) {
    int config_exit_code = EXIT_FAILURE;
    if (rank == root_rank) {
      mst::app::handle_config_parse_result(parsed, argv[0],
                                           config_exit_code);
    } else {
      config_exit_code = parsed.help_requested ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    MPI_Finalize();
    return config_exit_code;
  }
  const mst::app::app_config &config = parsed.config;

  const double total_start = MPI_Wtime();
  const mst::app::loaded_graph loaded = mst::app::load_graph(config);
  const mst::app::selected_graph &selected = loaded.selected;
  const mst::core::validated_graph &graph = loaded.graph;
  mst::dsu::disjoint_set<mst::core::uncompressed_parents> dsu(
      graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;
  int rounds = 0;
  int remaining_components = graph.vertex_count();
  mpi_workspace workspace(graph.vertex_count());
  double local_compute_seconds = 0.0;
  double reduce_seconds = 0.0;
  double contract_seconds = 0.0;
  const double mst_start = MPI_Wtime();

  while (remaining_components > 1) {
    ++rounds;
    const int begin =
        edge_begin_for_rank(static_cast<int>(graph.edges().size()), rank, size);
    const int end =
        edge_end_for_rank(static_cast<int>(graph.edges().size()), rank, size);
    const double local_compute_start = MPI_Wtime();
    const std::vector<std::uint64_t> &local =
        compute_local_minima(graph, dsu, begin, end, workspace, {});
    local_compute_seconds += MPI_Wtime() - local_compute_start;

    const double reduce_start = MPI_Wtime();
    const std::vector<std::uint64_t> &reduced =
        reduce_minima(local, workspace, graph, MPI_COMM_WORLD, {});
    reduce_seconds += MPI_Wtime() - reduce_start;

    const double contract_start = MPI_Wtime();
    const int admitted_count =
        apply_contractions(reduced, graph, dsu, mst_edges, total_weight);
    contract_seconds += MPI_Wtime() - contract_start;
    remaining_components -= admitted_count;

    if (admitted_count == 0) {
      break;
    }
  }
  const double mst_end = MPI_Wtime();
  const double total_end = MPI_Wtime();

  double max_local_compute_seconds = 0.0;
  double avg_local_compute_seconds = 0.0;
  double max_reduce_seconds = 0.0;
  double max_contract_seconds = 0.0;
  MPI_Reduce(&local_compute_seconds, &max_local_compute_seconds, 1, MPI_DOUBLE,
             MPI_MAX, root_rank, MPI_COMM_WORLD);
  MPI_Reduce(&local_compute_seconds, &avg_local_compute_seconds, 1, MPI_DOUBLE,
             MPI_SUM, root_rank, MPI_COMM_WORLD);
  MPI_Reduce(&reduce_seconds, &max_reduce_seconds, 1, MPI_DOUBLE, MPI_MAX,
             root_rank, MPI_COMM_WORLD);
  MPI_Reduce(&contract_seconds, &max_contract_seconds, 1, MPI_DOUBLE, MPI_MAX,
             root_rank, MPI_COMM_WORLD);

  const double verification_start = MPI_Wtime();
  const mst::boruvka::verification_result verification =
      mst::boruvka::verify_against_sequential_cpu(graph, mst_edges,
                                                  total_weight);
  const double verification_seconds = MPI_Wtime() - verification_start;
  double max_verification_seconds = 0.0;
  MPI_Reduce(&verification_seconds, &max_verification_seconds, 1, MPI_DOUBLE,
             MPI_MAX, root_rank, MPI_COMM_WORLD);
  const int local_verification_success = verification.success ? 1 : 0;
  int all_verification_success = 0;
  MPI_Allreduce(&local_verification_success, &all_verification_success, 1,
                MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int version = 0;
  int subversion = 0;
  MPI_Get_version(&version, &subversion);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int processor_name_length = 0;
  MPI_Get_processor_name(processor_name, &processor_name_length);

  if (rank == root_rank) {
    mst::app::print_result("MPI Boruvka MST", config, graph, mst_edges,
                           total_weight);
    if (all_verification_success != 0) {
      std::cout << "Sequential CPU verification: passed\n";
    } else {
      std::cerr << "Sequential CPU verification failed: expected weight "
                << verification.expected_total_weight << " with "
                << verification.expected_edge_count << " edges, got weight "
                << verification.actual_total_weight << " with "
                << verification.actual_edge_count << " edges\n";
    }

    avg_local_compute_seconds /= static_cast<double>(size);

    std::ostringstream report;
    report << "{\n";
    report << mst::reporting::common_metadata_json(
                  "mpi", all_verification_success != 0)
           << ",\n";
    report << mst::app::graph_metadata_json(selected) << ",\n";
    report << mst::app::configuration_metadata_json(config) << ",\n";
    std::ostringstream backend_timing_fields;
    backend_timing_fields << "    \"backend\": {\n";
    backend_timing_fields << "      \"max_scan_seconds\": "
                          << max_local_compute_seconds << ",\n";
    backend_timing_fields << "      \"avg_scan_seconds\": "
                          << avg_local_compute_seconds << "\n";
    backend_timing_fields << "    }\n";
    mst::reporting::write_phase_timings_json(
        report, mst::reporting::phase_timing_profile{
                    total_end - total_start,
                    mst_end - mst_start,
                    max_verification_seconds,
                    max_local_compute_seconds,
                    max_reduce_seconds,
                    max_contract_seconds,
                    0.0,
                },
        backend_timing_fields.str());
    report << ",\n";
    report << "  \"capabilities\": {\n";
    report << "    \"world_size\": " << size << ",\n";
    report << "    \"mpi_version_major\": " << version << ",\n";
    report << "    \"mpi_version_minor\": " << subversion << ",\n";
    report << "    \"processor_name\": \""
           << mst::reporting::json_escape(
                  std::string(processor_name,
                              static_cast<std::size_t>(processor_name_length)))
           << "\"\n";
    report << "  },\n";
    report << mst::app::mst_metadata_json(graph, mst_edges.size(), rounds,
                                          total_weight)
           << ",\n";
    mst::boruvka::write_verification_json(report, verification);
    report << "\n";
    report << "}\n";
    mst::app::write_report_if_requested(config, report.str());
  }

  MPI_Finalize();
  return all_verification_success != 0 ? 0 : 1;
}
