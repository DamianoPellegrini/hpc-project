#include <mpi.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "mst_common.hpp"
#include "mst_visualization.hpp"

namespace {

constexpr int kRootRank = 0;

// Split the edge list evenly across ranks. For this tiny test graph the simple
// block partition is enough and keeps the implementation easy to follow.
int edge_begin_for_rank(int edge_count, int rank, int size) {
  return (edge_count * rank) / size;
}

int edge_end_for_rank(int edge_count, int rank, int size) {
  return (edge_count * (rank + 1)) / size;
}

// Each rank scans only its local edge slice, but it still evaluates candidates
// for every current component. The root rank later merges those local minima.
std::vector<mst::Candidate> local_best_candidates(const mst::Graph &graph,
                                                  mst::DisjointSet &dsu,
                                                  int begin, int end) {
  std::vector<mst::Candidate> best(graph.vertex_count,
                                   mst::invalid_candidate());

  for (int index = begin; index < end; ++index) {
    const mst::Edge &edge = graph.edges[static_cast<std::size_t>(index)];
    const mst::VertexId left_root = dsu.find(edge.u);
    const mst::VertexId right_root = dsu.find(edge.v);
    if (left_root == right_root) {
      continue;
    }

    mst::consider_candidate(
        best[static_cast<std::size_t>(mst::index_of(left_root))], edge.u,
        edge.v, edge.weight);
    mst::consider_candidate(
        best[static_cast<std::size_t>(mst::index_of(right_root))], edge.u,
        edge.v, edge.weight);
  }

  return best;
}

} // namespace

int main(int32_t argc, char **argv) {
  MPI_Init(&argc, &argv);

  int32_t rank = 0;
  int32_t size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const mst::Graph graph = mst::make_test_graph();
  mst::DisjointSet dsu(graph.vertex_count);
  std::vector<mst::Edge> mst_edges;
  int total_weight = 0;

  while (true) {
    // Rank 0 owns the authoritative forest. At the start of every round it
    // broadcasts the current parent array so all ranks agree on the same
    // component labels before scanning edges.
    std::vector<int> parent = mst::pack_vertices(dsu.parent());
    MPI_Bcast(parent.data(), static_cast<int>(parent.size()), MPI_INT,
              kRootRank, MPI_COMM_WORLD);
    dsu.set_parent(mst::unpack_vertices(parent));

    const int component_count = dsu.component_count();
    if (component_count <= 1) {
      break;
    }

    const int begin =
        edge_begin_for_rank(static_cast<int>(graph.edges.size()), rank, size);
    const int end =
        edge_end_for_rank(static_cast<int>(graph.edges.size()), rank, size);
    const std::vector<mst::Candidate> local =
        local_best_candidates(graph, dsu, begin, end);

    // Pack the local component minima into plain integer arrays so MPI_Gather
    // can move them without any custom datatypes.
    std::vector<int> local_weights(graph.vertex_count,
                                   mst::value_of(mst::kInfiniteWeight));
    std::vector<int> local_u(graph.vertex_count, -1);
    std::vector<int> local_v(graph.vertex_count, -1);
    for (size_t component = 0; component < graph.vertex_count; ++component) {
      const mst::Candidate &candidate = local[component];
      if (!candidate) {
        continue;
      }
      local_weights[component] = mst::value_of(candidate->weight);
      local_u[component] = mst::index_of(candidate->u);
      local_v[component] = mst::index_of(candidate->v);
    }

    std::vector<int> gathered_weights;
    std::vector<int> gathered_u;
    std::vector<int> gathered_v;
    if (rank == kRootRank) {
      gathered_weights.resize(
          static_cast<std::size_t>(size * graph.vertex_count));
      gathered_u.resize(static_cast<std::size_t>(size * graph.vertex_count));
      gathered_v.resize(static_cast<std::size_t>(size * graph.vertex_count));
    }

    MPI_Gather(local_weights.data(), graph.vertex_count, MPI_INT,
               gathered_weights.data(), graph.vertex_count, MPI_INT, kRootRank,
               MPI_COMM_WORLD);
    MPI_Gather(local_u.data(), graph.vertex_count, MPI_INT, gathered_u.data(),
               graph.vertex_count, MPI_INT, kRootRank, MPI_COMM_WORLD);
    MPI_Gather(local_v.data(), graph.vertex_count, MPI_INT, gathered_v.data(),
               graph.vertex_count, MPI_INT, kRootRank, MPI_COMM_WORLD);

    int continue_flag = 0;
    if (rank == kRootRank) {
      // Merge the per-rank minima into one candidate per component. The best
      // edge for a component is the cheapest outgoing edge seen on any rank.
      std::vector<mst::Candidate> best(graph.vertex_count,
                                       mst::invalid_candidate());
      for (int source_rank = 0; source_rank < size; ++source_rank) {
        const int offset = source_rank * graph.vertex_count;
        for (int component = 0; component < graph.vertex_count; ++component) {
          const int weight =
              gathered_weights[static_cast<std::size_t>(offset + component)];
          if (weight == mst::value_of(mst::kInfiniteWeight)) {
            continue;
          }
          const mst::Candidate next = mst::Edge{
              mst::vertex_id(
                  gathered_u[static_cast<std::size_t>(offset + component)]),
              mst::vertex_id(
                  gathered_v[static_cast<std::size_t>(offset + component)]),
              mst::weight(weight)};
          if (mst::better_candidate(
                  next, best[static_cast<std::size_t>(component)])) {
            best[static_cast<std::size_t>(component)] = next;
          }
        }
      }

      // Apply the chosen edges to the shared forest. Some candidates can become
      // internal after earlier unions in the same round, so unite() filters
      // them out safely.
      bool changed = false;
      for (size_t component = 0; component < graph.vertex_count; ++component) {
        const mst::Candidate &candidate = best[component];
        if (!candidate) {
          continue;
        }
        if (dsu.unite(candidate->u, candidate->v)) {
          mst_edges.push_back(*candidate);
          total_weight += mst::value_of(candidate->weight);
          changed = true;
        }
      }

      // Broadcast the updated forest to the other ranks before the next round.
      parent = mst::pack_vertices(dsu.parent());
      dsu.set_parent(mst::unpack_vertices(parent));
      continue_flag = changed ? 1 : 0;
    }

    MPI_Bcast(&continue_flag, 1, MPI_INT, kRootRank, MPI_COMM_WORLD);
    if (continue_flag == 0) {
      break;
    }
  }

  if (rank == kRootRank) {
    std::cout << "MPI Boruvka MST\n";
    std::cout << mst::mst_summary(mst_edges, total_weight);
    mst::viz::render_graph_with_mst(graph, mst_edges, total_weight);
  }

  MPI_Finalize();
  return 0;
}
