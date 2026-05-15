#include <omp.h>

#include <cstddef>
#include <iostream>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/summary.hpp"
#include "mst/dsu/disjoint_set.hpp"
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
        auto &global =
            best[static_cast<std::size_t>(component)];
        const auto &candidate =
            local[static_cast<std::size_t>(component)];
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

  const mst::core::validated_graph graph =
      mst::core::validate(mst::core::make_test_graph());
  mst::dsu::disjoint_set<mst::core::uncompressed_parents> dsu(
      graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;

  while (dsu.component_count() > 1) {
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

  std::cout << "OpenMP Boruvka MST using " << omp_get_max_threads()
            << " threads\n";
  std::cout << mst::core::mst_summary(mst_edges, total_weight);
  mst::visualization::render_graph_with_mst(graph, mst_edges, total_weight);
  return 0;
}
