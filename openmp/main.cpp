#include <omp.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "mst_common.hpp"
#include "mst_visualization.hpp"

namespace {

// OpenMP version of the same scan: each thread inspects a subset of the edges
// and records the cheapest outgoing edge it saw for every component.
std::vector<mst::Candidate> local_best_candidates(const mst::Graph &graph,
                                                  mst::DisjointSet &dsu) {
  // The DSU is shared between threads, so we take a snapshot of the parent
  // array and perform read-only root lookups inside the parallel region.
  const std::vector<mst::VertexId> parent_snapshot = dsu.parent();
  std::vector<mst::Candidate> best(graph.vertex_count, mst::invalid_candidate());

#pragma omp parallel
  {
    std::vector<mst::Candidate> local(graph.vertex_count, mst::invalid_candidate());

#pragma omp for nowait
    for (std::size_t index = 0; index < graph.edges.size(); ++index) {
      const mst::Edge &edge = graph.edges[index];
      const mst::VertexId left_root = mst::find_root(parent_snapshot, edge.u);
      const mst::VertexId right_root = mst::find_root(parent_snapshot, edge.v);
      if (left_root == right_root) {
        continue;
      }

      mst::consider_candidate(local[static_cast<std::size_t>(mst::index_of(left_root))], edge.u,
                              edge.v, edge.weight);
      mst::consider_candidate(local[static_cast<std::size_t>(mst::index_of(right_root))], edge.u,
                              edge.v, edge.weight);
    }

#pragma omp critical
    {
      for (int component = 0; component < graph.vertex_count; ++component) {
        if (mst::better_candidate(local[static_cast<std::size_t>(component)],
                                  best[static_cast<std::size_t>(component)])) {
          best[static_cast<std::size_t>(component)] = local[static_cast<std::size_t>(component)];
        }
      }
    }
  }

  return best;
}

} // namespace

int main() {
  const mst::Graph graph = mst::make_test_graph();
  mst::DisjointSet dsu(graph.vertex_count);
  std::vector<mst::Edge> mst_edges;
  int total_weight = 0;

  while (dsu.component_count() > 1) {
    // Each Boruvka round asks every component for its cheapest outgoing edge.
    const std::vector<mst::Candidate> best = local_best_candidates(graph, dsu);

    bool changed = false;
    for (int component = 0; component < graph.vertex_count; ++component) {
      const mst::Candidate &candidate = best[static_cast<std::size_t>(component)];
      if (!candidate) {
        continue;
      }
      if (dsu.unite(candidate->u, candidate->v)) {
        mst_edges.push_back(*candidate);
        total_weight += mst::value_of(candidate->weight);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  std::cout << "OpenMP Boruvka MST using " << omp_get_max_threads() << " threads\n";
  std::cout << mst::mst_summary(mst_edges, total_weight);
  mst::viz::render_graph_with_mst(graph, mst_edges, total_weight);
  return 0;
}
