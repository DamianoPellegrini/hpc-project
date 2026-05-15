#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/visualization/render_graph.hpp"

namespace {

std::string strip_ansi(std::string_view text) {
  std::string stripped;
  stripped.reserve(text.size());

  for (std::size_t index = 0; index < text.size(); ++index) {
    if (text[index] == '\x1b' && index + 1 < text.size() &&
        text[index + 1] == '[') {
      index += 2;
      while (index < text.size() &&
             ((text[index] < '@') || (text[index] > '~'))) {
        ++index;
      }
      continue;
    }

    stripped.push_back(text[index]);
  }

  return stripped;
}

bool node_labels_render_above_edge_weights() {
  using namespace mst::core;
  using namespace mst::visualization;

  const validated_graph graph = validate(raw_graph{
      3,
      {
          {make_vertex_id(0), make_vertex_id(1), make_edge_weight(9)},
      },
  });
  const std::vector<canvas_point> points{
      {2, 1},
      {8, 1},
      {5, 1},
  };

  canvas target{11, 3};
  draw_graph_content(target, graph, points, {});

  std::ostringstream out;
  target.render(out);
  const std::string rendered = strip_ansi(out.str());

  const std::size_t first_newline = rendered.find('\n');
  if (first_newline == std::string::npos) {
    return false;
  }

  const std::size_t row_start = first_newline + 1;
  if (row_start + 5 >= rendered.size()) {
    return false;
  }

  return rendered[row_start + 5] == '2';
}

} // namespace

int main() {
  using namespace mst::core;

  const raw_graph raw = make_test_graph();
  const validated_graph graph = validate(make_test_graph());
  maybe_candidate_edge current;

  consider_candidate(current,
                     edge{make_vertex_id(0), make_vertex_id(1), make_edge_weight(4)});
  consider_candidate(current,
                     edge{make_vertex_id(1), make_vertex_id(2), make_edge_weight(1)});

  if (raw.vertex_count() != 12) {
    return 1;
  }
  if (!has_valid_vertex_ids(raw)) {
    return 1;
  }
  if (graph.edges().size() != 29) {
    return 1;
  }
  if (!current || as_value(current->value.weight) != 1) {
    return 1;
  }

  mst::dsu::disjoint_set<uncompressed_parents> set{3};
  const auto admitted = set.unite(candidate_edge{
      edge{make_vertex_id(0), make_vertex_id(1), make_edge_weight(4)}});
  if (!admitted) {
    return 1;
  }

  const auto rejected = set.unite(candidate_edge{
      edge{make_vertex_id(0), make_vertex_id(1), make_edge_weight(4)}});
  if (rejected) {
    return 1;
  }
  if (!node_labels_render_above_edge_weights()) {
    return 1;
  }

  static_assert(!std::is_same_v<raw_graph, validated_graph>);
  return 0;
}
