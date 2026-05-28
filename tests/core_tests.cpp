#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph_examples.hpp"
#include "mst/core/graph.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/dsu/parallel_disjoint_set.hpp"
#include "mst/execution/domain.hpp"
#include "mst/memory/buffer.hpp"
#include "mst/boruvka/contracts.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/app/graph_selection.hpp"
#include "mst/reporting/json_report.hpp"
#include "mst/visualization/render_graph.hpp"

namespace {

template <class value_t>
concept has_public_value_member = requires(value_t value) { value.value; };

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

bool json_report_escapes_and_writes_file() {
  using namespace mst::reporting;

  const std::string escaped = json_escape("quote \" slash \\ newline\n");
  if (escaped != "quote \\\" slash \\\\ newline\\n") {
    return false;
  }

  const std::filesystem::path report_path =
      std::filesystem::temp_directory_path() / "mst-json-report-test.json";
  const bool wrote = write_report(report_path, "{\"ok\":true}\n");
  if (!wrote) {
    return false;
  }

  std::ifstream report_stream(report_path);
  std::stringstream buffer;
  buffer << report_stream.rdbuf();
  std::filesystem::remove(report_path);
  return buffer.str() == "{\"ok\":true}\n";
}

bool candidate_keys_order_by_weight_then_edge_index() {
  using namespace mst::core;

  const candidate_key heavy_early =
      make_candidate_key(make_edge_weight(8), make_edge_index(1));
  const candidate_key light_late =
      make_candidate_key(make_edge_weight(2), make_edge_index(40));
  const candidate_key same_weight_early =
      make_candidate_key(make_edge_weight(2), make_edge_index(3));
  const candidate_key same_weight_late =
      make_candidate_key(make_edge_weight(2), make_edge_index(9));

  return light_late < heavy_early && same_weight_early < same_weight_late &&
         empty_candidate_key > heavy_early;
}

bool random_graph_generation_is_deterministic_and_validated() {
  using namespace mst::core;

  const auto config = random_connected_graph_config::create(
      make_graph_vertex_count(64), make_edge_count(128),
      make_random_seed(886261), make_edge_weight(100));
  if (!config) {
    return false;
  }

  const raw_graph first = make_random_connected_graph(*config);
  const raw_graph second = make_random_connected_graph(*config);
  if (first.vertex_count() != 64 || first.edges().size() != 191) {
    return false;
  }
  if (first.edges().size() != second.edges().size()) {
    return false;
  }
  for (std::size_t index = 0; index < first.edges().size(); ++index) {
    if (first.edges()[index] != second.edges()[index]) {
      return false;
    }
  }

  const auto invalid = random_connected_graph_config::create(
      make_graph_vertex_count(1), make_edge_count(1), make_random_seed(1),
      make_edge_weight(10));
  return !invalid && has_valid_vertex_ids(first);
}

bool named_graph_examples_are_available() {
  using namespace mst::core;

  return make_tiny_triangle_graph().vertex_count() == 3 &&
         make_square_with_diagonal_graph().edges().size() == 5 &&
         make_equal_weight_tie_graph().edges().size() >= 4 &&
         make_sparse_12_vertex_graph().vertex_count() == 12 &&
         make_dense_16_vertex_graph().vertex_count() == 16;
}

bool graph_selection_metadata_is_shared() {
  const mst::app::selected_graph selected =
      mst::app::select_graph_by_name("tie");
  const std::string metadata = mst::app::graph_metadata_json(selected);

  return selected.name == "tie" && selected.graph.vertex_count() == 5 &&
         metadata.find("\"name\": \"tie\"") != std::string::npos &&
         mst::app::select_graph_by_name("triangle").graph.vertex_count() == 3 &&
         mst::app::select_graph_by_name("random").random_config.has_value();
}

bool parallel_disjoint_set_admits_edges_once() {
  mst::dsu::parallel_disjoint_set set{4};
  const mst::core::candidate_edge first{mst::core::edge{
      mst::core::make_vertex_id(0), mst::core::make_vertex_id(1),
      mst::core::make_edge_weight(3)}};
  const mst::core::candidate_edge second{mst::core::edge{
      mst::core::make_vertex_id(1), mst::core::make_vertex_id(2),
      mst::core::make_edge_weight(4)}};

  if (!set.unite(first) || !set.unite(second) || set.unite(first)) {
    return false;
  }
  set.compress_vertex(mst::core::make_vertex_id(2));
  const mst::dsu::parent_snapshot snapshot = set.snapshot();
  return mst::dsu::find_root(snapshot, mst::core::make_vertex_id(0)) ==
             mst::dsu::find_root(snapshot, mst::core::make_vertex_id(2)) &&
         set.component_count() == 2;
}

bool large_graph_rendering_is_summarized() {
  const auto config = mst::core::random_connected_graph_config::create(
      mst::core::make_graph_vertex_count(129), mst::core::make_edge_count(0),
      mst::core::make_random_seed(7), mst::core::make_edge_weight(5));
  if (!config) {
    return false;
  }
  const mst::core::validated_graph graph =
      mst::core::validate(mst::core::make_random_connected_graph(*config));
  std::ostringstream out;
  mst::visualization::render_graph_with_mst(graph, {}, 0, out);
  return out.str().find("Graph visualization skipped") != std::string::npos;
}

bool sequential_reference_weights_match_named_graphs() {
  const auto triangle = mst::boruvka::sequential_cpu_mst(
      mst::core::validate(mst::core::make_tiny_triangle_graph()));
  const auto square = mst::boruvka::sequential_cpu_mst(
      mst::core::validate(mst::core::make_square_with_diagonal_graph()));
  const auto tie = mst::boruvka::sequential_cpu_mst(
      mst::core::validate(mst::core::make_equal_weight_tie_graph()));
  const auto sparse = mst::boruvka::sequential_cpu_mst(
      mst::core::validate(mst::core::make_sparse_12_vertex_graph()));

  return triangle.total_weight == 5 && triangle.edges.size() == 2 &&
         square.total_weight == 5 && square.edges.size() == 3 &&
         tie.total_weight == 4 && tie.edges.size() == 4 &&
         sparse.total_weight == 30 && sparse.edges.size() == 11;
}

bool sequential_verifier_reports_match_and_mismatch() {
  const mst::core::validated_graph graph =
      mst::core::validate(mst::core::make_equal_weight_tie_graph());
  const mst::boruvka::result reference = mst::boruvka::sequential_cpu_mst(graph);
  const mst::boruvka::verification_result match =
      mst::boruvka::verify_against_sequential_cpu(
          graph, reference.edges, reference.total_weight);
  const mst::boruvka::verification_result mismatch =
      mst::boruvka::verify_against_sequential_cpu(
          graph, reference.edges, reference.total_weight + 1);

  std::ostringstream json;
  mst::boruvka::write_verification_json(json, match);
  return match.success && !mismatch.success &&
         match.expected_total_weight == reference.total_weight &&
         json.str().find("\"sequential_cpu_success\": true") !=
             std::string::npos;
}

struct fake_round_engine {
  using execution_domain = mst::execution::cpu_thread_domain;
  using memory_space = mst::memory::host_memory;
  using reduction_policy = mst::core::synchronized_state;
  using contraction_policy = mst::core::synchronized_state;

  void initialize(const mst::core::validated_graph &) {}
  void find_local_minima(mst::core::round_index) {}
  void reduce_component_minima(mst::core::round_index) {}
  void apply_contractions(mst::core::round_index) {}
  void compress_parents(mst::core::round_index) {}
  mst::boruvka::result result() { return {}; }
};

static_assert(mst::boruvka::candidate_scanner<fake_round_engine>);
static_assert(mst::boruvka::candidate_reducer<fake_round_engine>);
static_assert(mst::boruvka::component_contractor<fake_round_engine>);
static_assert(mst::boruvka::parent_compressor<fake_round_engine>);
static_assert(mst::boruvka::boruvka_round_engine<fake_round_engine>);

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
  if (!current || current->value.weight.value() != 1) {
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
  if (!json_report_escapes_and_writes_file()) {
    return 1;
  }
  if (!candidate_keys_order_by_weight_then_edge_index()) {
    return 1;
  }
  if (!random_graph_generation_is_deterministic_and_validated()) {
    return 1;
  }
  if (!named_graph_examples_are_available()) {
    return 1;
  }
  if (!graph_selection_metadata_is_shared()) {
    return 1;
  }
  if (!parallel_disjoint_set_admits_edges_once()) {
    return 1;
  }
  if (!large_graph_rendering_is_summarized()) {
    return 1;
  }
  if (!sequential_reference_weights_match_named_graphs()) {
    return 1;
  }
  if (!sequential_verifier_reports_match_and_mismatch()) {
    return 1;
  }

  static_assert(!std::is_same_v<raw_graph, validated_graph>);
  static_assert(std::is_constructible_v<int, vertex_id>);
  static_assert(std::is_constructible_v<std::size_t, vertex_id>);
  static_assert(make_vertex_id(3).index() == std::size_t{3});
  static_assert(make_vertex_id(3).value() == 3);
  static_assert(!has_public_value_member<vertex_id>);
  static_assert(!std::is_convertible_v<vertex_id, int>);
  static_assert(!std::is_convertible_v<vertex_id, std::size_t>);
  static_assert(std::is_constructible_v<int, component_id>);
  static_assert(std::is_constructible_v<std::size_t, component_id>);
  static_assert(make_component_id(4).index() == std::size_t{4});
  static_assert(make_component_id(4).value() == 4);
  static_assert(!has_public_value_member<component_id>);
  static_assert(!std::is_convertible_v<component_id, int>);
  static_assert(!std::is_convertible_v<component_id, std::size_t>);
  static_assert(std::is_constructible_v<int, edge_weight>);
  static_assert(make_edge_weight(5).value() == 5);
  static_assert(!has_public_value_member<edge_weight>);
  static_assert(!std::is_convertible_v<edge_weight, int>);
  static_assert(make_edge_index(7).value() == std::size_t{7});
  return 0;
}
