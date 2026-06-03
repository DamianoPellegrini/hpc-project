#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

#include "mst/core/graph_examples.hpp"

#ifndef MST_ENABLE_RENDERING
#define MST_ENABLE_RENDERING 1
#endif

#ifndef MST_DEFAULT_CUDA_HOST_MEMORY
#define MST_DEFAULT_CUDA_HOST_MEMORY "pinned"
#endif

namespace mst::app {

enum class graph_source_kind {
  named_example,
  random_generated,
};

enum class cuda_host_memory_mode {
  pageable,
  pinned,
  mapped_zero_copy,
};

struct graph_config {
  graph_source_kind source = graph_source_kind::named_example;
  std::string name = "test";
  std::optional<mst::core::random_connected_graph_config> random_config;
};

struct app_config {
  graph_config graph;
  std::filesystem::path report_path;
  std::optional<bool> render_graph;
  cuda_host_memory_mode cuda_host_memory = cuda_host_memory_mode::pinned;
};

struct config_parse_result {
  bool success = false;
  bool help_requested = false;
  app_config config;
  std::string error;
};

inline bool compile_time_rendering_enabled() noexcept {
  return MST_ENABLE_RENDERING != 0;
}

inline const char *graph_source_kind_name(graph_source_kind source) noexcept {
  switch (source) {
  case graph_source_kind::named_example:
    return "named_example";
  case graph_source_kind::random_generated:
    return "random_generated";
  }
  return "unknown";
}

inline const char *
cuda_host_memory_mode_name(cuda_host_memory_mode mode) noexcept {
  switch (mode) {
  case cuda_host_memory_mode::pageable:
    return "pageable";
  case cuda_host_memory_mode::pinned:
    return "pinned";
  case cuda_host_memory_mode::mapped_zero_copy:
    return "mapped_zero_copy";
  }
  return "unknown";
}

inline std::optional<cuda_host_memory_mode>
parse_cuda_host_memory_mode(std::string_view value) {
  if (value == "pageable") {
    return cuda_host_memory_mode::pageable;
  }
  if (value == "pinned") {
    return cuda_host_memory_mode::pinned;
  }
  if (value == "zero_copy" || value == "zerocopy" ||
      value == "mapped" || value == "mapped_zero_copy") {
    return cuda_host_memory_mode::mapped_zero_copy;
  }
  return std::nullopt;
}

inline cuda_host_memory_mode default_cuda_host_memory_mode() {
  return parse_cuda_host_memory_mode(MST_DEFAULT_CUDA_HOST_MEMORY)
      .value_or(cuda_host_memory_mode::pinned);
}

inline bool is_known_graph_name(std::string_view name) {
  return name == "test" || name == "triangle" || name == "square" ||
         name == "tie" || name == "dense16" || name == "random";
}

inline std::string graph_name_list() {
  return "test, triangle, square, tie, dense16, random";
}

template <class value_t>
std::optional<value_t> parse_integer(std::string_view value) {
  value_t parsed{};
  const char *begin = value.data();
  const char *end = value.data() + value.size();
  const auto result = std::from_chars(begin, end, parsed, 10);
  if (result.ec != std::errc{} || result.ptr != end) {
    return std::nullopt;
  }
  return parsed;
}

inline std::optional<std::string_view>
option_value(std::string_view argument, std::string_view option, int &index,
             int argc, char **argv, std::string &error) {
  if (argument == option) {
    if (index + 1 >= argc) {
      error = std::string("missing value for ") + std::string(option);
      return std::nullopt;
    }
    ++index;
    return std::string_view(argv[index]);
  }

  const std::string with_equals = std::string(option) + "=";
  if (argument.starts_with(with_equals)) {
    return argument.substr(with_equals.size());
  }

  return std::nullopt;
}

inline bool is_option(std::string_view argument, std::string_view option) {
  return argument == option ||
         argument.starts_with(std::string(option) + "=");
}

inline std::string usage(std::string_view executable) {
  std::ostringstream out;
  out << "Usage: " << executable << " [options]\n";
  out << "\n";
  out << "Options:\n";
  out << "  --graph <name>                 Graph input: "
      << graph_name_list() << "\n";
  out << "  --random-vertices <n>          Vertex count for --graph random\n";
  out << "  --random-extra-edges <n>       Extra random edges after the tree\n";
  out << "  --random-seed <n>              Deterministic random seed\n";
  out << "  --random-max-weight <n>        Maximum generated edge weight\n";
  out << "  --report <path>                Write JSON report to path\n";
  out << "  --render                       Request graph rendering\n";
  out << "  --no-render                    Skip graph rendering\n";
  out << "  --benchmark                    Alias for --no-render\n";
  out << "  --cuda-host-memory <mode>      pageable, pinned, zero_copy\n";
  out << "  -h, --help                     Show this help\n";
  return out.str();
}

inline config_parse_result parse_app_config(int argc, char **argv) {
  config_parse_result result;
  result.config.cuda_host_memory = default_cuda_host_memory_mode();

  std::string graph_name = "test";
  auto random_config = mst::core::default_large_random_graph_config();
  int random_vertices = random_config.vertices().value();
  std::size_t random_extra_edges = random_config.extra_edges().value();
  std::uint64_t random_seed = random_config.seed().value();
  int random_max_weight = random_config.max_weight().value();

  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "-h" || argument == "--help") {
      result.success = true;
      result.help_requested = true;
      return result;
    }

    std::string error;
    if (is_option(argument, "--graph")) {
      const auto value = option_value(argument, "--graph", index, argc, argv,
                                      error);
      if (!value) {
        result.error = error;
        return result;
      }
      graph_name = std::string(*value);
      continue;
    }
    if (is_option(argument, "--random-vertices")) {
      const auto value = option_value(argument, "--random-vertices", index,
                                      argc, argv, error);
      const auto parsed = value ? parse_integer<int>(*value) : std::nullopt;
      if (!value || !parsed) {
        result.error = error.empty()
                           ? "invalid value for --random-vertices"
                           : error;
        return result;
      }
      random_vertices = *parsed;
      continue;
    }
    if (is_option(argument, "--random-extra-edges")) {
      const auto value = option_value(argument, "--random-extra-edges", index,
                                      argc, argv, error);
      const auto parsed =
          value ? parse_integer<std::size_t>(*value) : std::nullopt;
      if (!value || !parsed) {
        result.error = error.empty()
                           ? "invalid value for --random-extra-edges"
                           : error;
        return result;
      }
      random_extra_edges = *parsed;
      continue;
    }
    if (is_option(argument, "--random-seed")) {
      const auto value = option_value(argument, "--random-seed", index, argc,
                                      argv, error);
      const auto parsed =
          value ? parse_integer<std::uint64_t>(*value) : std::nullopt;
      if (!value || !parsed) {
        result.error =
            error.empty() ? "invalid value for --random-seed" : error;
        return result;
      }
      random_seed = *parsed;
      continue;
    }
    if (is_option(argument, "--random-max-weight")) {
      const auto value = option_value(argument, "--random-max-weight", index,
                                      argc, argv, error);
      const auto parsed = value ? parse_integer<int>(*value) : std::nullopt;
      if (!value || !parsed) {
        result.error = error.empty()
                           ? "invalid value for --random-max-weight"
                           : error;
        return result;
      }
      random_max_weight = *parsed;
      continue;
    }
    if (is_option(argument, "--report")) {
      const auto value = option_value(argument, "--report", index, argc, argv,
                                      error);
      if (!value) {
        result.error = error;
        return result;
      }
      result.config.report_path = std::filesystem::path(std::string(*value));
      continue;
    }
    if (argument == "--render") {
      result.config.render_graph = true;
      continue;
    }
    if (argument == "--no-render" || argument == "--benchmark") {
      result.config.render_graph = false;
      continue;
    }
    if (is_option(argument, "--cuda-host-memory")) {
      const auto value = option_value(argument, "--cuda-host-memory", index,
                                      argc, argv, error);
      const auto parsed = value ? parse_cuda_host_memory_mode(*value)
                                : std::nullopt;
      if (!value || !parsed) {
        result.error = error.empty()
                           ? "invalid value for --cuda-host-memory"
                           : error;
        return result;
      }
      result.config.cuda_host_memory = *parsed;
      continue;
    }

    result.error = std::string("unknown option ") + std::string(argument);
    return result;
  }

  if (!is_known_graph_name(graph_name)) {
    result.error = "unknown graph '" + graph_name + "'; expected one of " +
                   graph_name_list();
    return result;
  }

  if (graph_name == "random") {
    const auto config = mst::core::random_connected_graph_config::create(
        mst::core::make_graph_vertex_count(random_vertices),
        mst::core::make_edge_count(random_extra_edges),
        mst::core::make_random_seed(random_seed),
        mst::core::make_edge_weight(random_max_weight));
    if (!config) {
      result.error = "invalid random graph configuration";
      return result;
    }
    result.config.graph = graph_config{
        graph_source_kind::random_generated,
        graph_name,
        *config,
    };
  } else {
    result.config.graph = graph_config{
        graph_source_kind::named_example,
        graph_name,
        std::nullopt,
    };
  }

  result.success = true;
  return result;
}

inline bool should_render_graph(const app_config &config) noexcept {
  return compile_time_rendering_enabled() &&
         config.render_graph.value_or(true);
}

} // namespace mst::app
