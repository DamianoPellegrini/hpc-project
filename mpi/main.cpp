#include <mpi.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/summary.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/execution/domain.hpp"
#include "mst/reporting/json_report.hpp"
#include "mst/visualization/render_graph.hpp"

namespace mst::backend::mpi {

constexpr int root_rank = 0;

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

std::vector<mst::core::maybe_candidate_edge> local_best_candidates(
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu, int begin,
    int end) {
  std::vector<mst::core::maybe_candidate_edge> best(
      static_cast<std::size_t>(graph.vertex_count()));

  for (int index = begin; index < end; ++index) {
    const mst::core::edge &edge =
        graph.edges()[static_cast<std::size_t>(index)];
    const mst::core::component_id left_root = dsu.find(edge.u);
    const mst::core::component_id right_root = dsu.find(edge.v);
    if (left_root == right_root) {
      continue;
    }

    mst::core::consider_candidate(best[left_root.index()], edge);
    mst::core::consider_candidate(best[right_root.index()], edge);
  }

  return best;
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

std::vector<mst::core::maybe_candidate_edge> compute_local_minima(
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu,
    int edge_begin, int edge_end,
    mst::execution::mpi_round<mst::execution::parents_broadcasted>) {
  return local_best_candidates(graph, dsu, edge_begin, edge_end);
}

std::vector<mst::core::maybe_candidate_edge> reduce_minima(
    const std::vector<mst::core::maybe_candidate_edge> &local, int rank,
    int size, int vertex_count, int root_rank_value, MPI_Comm comm,
    mst::execution::mpi_round<mst::execution::local_minima_computed>) {
  std::vector<int> local_weights(static_cast<std::size_t>(vertex_count),
                                 mst::core::infinite_weight.value());
  std::vector<int> local_u(static_cast<std::size_t>(vertex_count), -1);
  std::vector<int> local_v(static_cast<std::size_t>(vertex_count), -1);
  for (int component = 0; component < vertex_count; ++component) {
    const auto &candidate = local[static_cast<std::size_t>(component)];
    if (!candidate) {
      continue;
    }
    local_weights[static_cast<std::size_t>(component)] =
        candidate->value.weight.value();
    local_u[static_cast<std::size_t>(component)] = candidate->value.u.value();
    local_v[static_cast<std::size_t>(component)] = candidate->value.v.value();
  }

  std::vector<int> gathered_weights;
  std::vector<int> gathered_u;
  std::vector<int> gathered_v;
  if (rank == root_rank_value) {
    const auto gathered_size = static_cast<std::size_t>(size * vertex_count);
    gathered_weights.resize(gathered_size);
    gathered_u.resize(gathered_size);
    gathered_v.resize(gathered_size);
  }

  MPI_Gather(local_weights.data(), vertex_count, MPI_INT,
             gathered_weights.data(), vertex_count, MPI_INT, root_rank_value,
             comm);
  MPI_Gather(local_u.data(), vertex_count, MPI_INT, gathered_u.data(),
             vertex_count, MPI_INT, root_rank_value, comm);
  MPI_Gather(local_v.data(), vertex_count, MPI_INT, gathered_v.data(),
             vertex_count, MPI_INT, root_rank_value, comm);

  std::vector<mst::core::maybe_candidate_edge> best;
  if (rank != root_rank_value) {
    return best;
  }

  best.resize(static_cast<std::size_t>(vertex_count));
  for (int source_rank = 0; source_rank < size; ++source_rank) {
    const int offset = source_rank * vertex_count;
    for (int component = 0; component < vertex_count; ++component) {
      const int weight =
          gathered_weights[static_cast<std::size_t>(offset + component)];
      if (weight == mst::core::infinite_weight.value()) {
        continue;
      }
      const mst::core::candidate_edge candidate{mst::core::edge{
          mst::core::make_vertex_id(
              gathered_u[static_cast<std::size_t>(offset + component)]),
          mst::core::make_vertex_id(
              gathered_v[static_cast<std::size_t>(offset + component)]),
          mst::core::make_edge_weight(weight)}};
      const mst::core::maybe_candidate_edge next = candidate;
      if (mst::core::better_candidate(
              next, best[static_cast<std::size_t>(component)])) {
        best[static_cast<std::size_t>(component)] = next;
      }
    }
  }

  return best;
}

bool apply_contractions(
    const std::vector<mst::core::maybe_candidate_edge> &best,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu,
    std::vector<mst::core::mst_edge> &mst_edges, int &total_weight) {
  bool changed = false;
  for (const auto &candidate : best) {
    if (!candidate) {
      continue;
    }
    if (auto admitted = dsu.unite(*candidate)) {
      mst_edges.push_back(*admitted);
      total_weight += admitted->value.weight.value();
      changed = true;
    }
  }
  return changed;
}

} // namespace mst::backend::mpi

int main(std::int32_t argc, char **argv) {
  using namespace mst::backend::mpi;

  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const mst::core::validated_graph graph =
      mst::core::validate(mst::core::make_test_graph());
  mst::dsu::disjoint_set<mst::core::uncompressed_parents> dsu(
      graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;
  int rounds = 0;
  double local_compute_seconds = 0.0;
  const double total_start = MPI_Wtime();
  const double mst_start = MPI_Wtime();

  while (true) {
    const auto broadcast_phase =
        broadcast_parents(dsu, root_rank, MPI_COMM_WORLD);
    if (dsu.component_count() <= 1) {
      break;
    }

    ++rounds;
    const int begin =
        edge_begin_for_rank(static_cast<int>(graph.edges().size()), rank, size);
    const int end =
        edge_end_for_rank(static_cast<int>(graph.edges().size()), rank, size);
    const double local_compute_start = MPI_Wtime();
    const auto local =
        compute_local_minima(graph, dsu, begin, end, broadcast_phase);
    local_compute_seconds += MPI_Wtime() - local_compute_start;
    const auto reduced = reduce_minima(local, rank, size, graph.vertex_count(),
                                       root_rank, MPI_COMM_WORLD, {});

    int continue_flag = 0;
    if (rank == root_rank) {
      continue_flag =
          apply_contractions(reduced, dsu, mst_edges, total_weight) ? 1 : 0;
    }

    MPI_Bcast(&continue_flag, 1, MPI_INT, root_rank, MPI_COMM_WORLD);
    if (continue_flag == 0) {
      break;
    }
  }
  const double mst_end = MPI_Wtime();
  const double total_end = MPI_Wtime();

  double max_local_compute_seconds = 0.0;
  double avg_local_compute_seconds = 0.0;
  MPI_Reduce(&local_compute_seconds, &max_local_compute_seconds, 1, MPI_DOUBLE,
             MPI_MAX, root_rank, MPI_COMM_WORLD);
  MPI_Reduce(&local_compute_seconds, &avg_local_compute_seconds, 1, MPI_DOUBLE,
             MPI_SUM, root_rank, MPI_COMM_WORLD);

  int version = 0;
  int subversion = 0;
  MPI_Get_version(&version, &subversion);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int processor_name_length = 0;
  MPI_Get_processor_name(processor_name, &processor_name_length);

  if (rank == root_rank) {
    std::cout << "MPI Boruvka MST\n";
    std::cout << mst::core::mst_summary(mst_edges, total_weight);
    mst::visualization::render_graph_with_mst(graph, mst_edges, total_weight);

    avg_local_compute_seconds /= static_cast<double>(size);

    std::ostringstream report;
    report << "{\n";
    report << mst::reporting::common_metadata_json("mpi", true) << ",\n";
    report << "  \"timings\": {\n";
    report << "    \"total_seconds\": " << (total_end - total_start) << ",\n";
    report << "    \"mst_loop_seconds\": " << (mst_end - mst_start) << ",\n";
    report << "    \"max_local_compute_seconds\": " << max_local_compute_seconds
           << ",\n";
    report << "    \"avg_local_compute_seconds\": " << avg_local_compute_seconds
           << "\n";
    report << "  },\n";
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
    report << "  \"mst\": {\n";
    report << "    \"vertex_count\": " << graph.vertex_count() << ",\n";
    report << "    \"input_edge_count\": " << graph.edges().size() << ",\n";
    report << "    \"selected_edge_count\": " << mst_edges.size() << ",\n";
    report << "    \"rounds\": " << rounds << ",\n";
    report << "    \"total_weight\": " << total_weight << "\n";
    report << "  }\n";
    report << "}\n";
    mst::reporting::write_report_from_env(report.str());
  }

  MPI_Finalize();
  return 0;
}
