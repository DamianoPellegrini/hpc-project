#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mst/app/graph_selection.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/summary.hpp"
#include "mst/memory/buffer.hpp"
#include "mst/reporting/json_report.hpp"
#include "mst/visualization/render_graph.hpp"
#include "cuda/boruvka_kernels.cuh"

namespace mst::backend::cuda_backend {

namespace {

struct cuda_profile {
  double host_to_device_seconds = 0.0;
  double scan_seconds = 0.0;
  double contract_seconds = 0.0;
  double compress_seconds = 0.0;
  double device_to_host_seconds = 0.0;
};

struct cuda_result {
  std::vector<mst::core::mst_edge> edges;
  int total_weight = 0;
  int rounds = 0;
};

inline void check_cuda(cudaError_t status, const char *message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " +
                             cudaGetErrorString(status));
  }
}

template <class value_t>
class device_allocation {
public:
  explicit device_allocation(std::size_t size) : size_(size) {
    if (size_ == 0) {
      return;
    }
    check_cuda(cudaMalloc(&data_, sizeof(value_t) * size_),
               "cudaMalloc failed");
  }

  device_allocation(const device_allocation &) = delete;
  device_allocation &operator=(const device_allocation &) = delete;

  device_allocation(device_allocation &&other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  device_allocation &operator=(device_allocation &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    release();
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
    return *this;
  }

  ~device_allocation() { release(); }

  mst::memory::device_buffer<value_t> buffer() noexcept {
    return mst::memory::device_buffer<value_t>{data_, size_};
  }

private:
  void release() noexcept {
    if (data_ != nullptr) {
      cudaFree(data_);
      data_ = nullptr;
    }
  }

  value_t *data_ = nullptr;
  std::size_t size_ = 0;
};

mst::memory::host_buffer<device_edge, mst::memory::host_memory>
make_device_edges(const mst::core::validated_graph &graph) {
  std::vector<device_edge> edges;
  edges.reserve(graph.edges().size());
  for (const mst::core::edge &edge : graph.edges()) {
    edges.push_back({edge.u.value(), edge.v.value(), edge.weight.value()});
  }
  return mst::memory::host_buffer<device_edge, mst::memory::host_memory>{
      std::move(edges)};
}

cuda_result run_boruvka_on_device(const mst::core::validated_graph &graph,
                                  cuda_profile &profile) {
  using clock = std::chrono::steady_clock;

  auto host_edges = make_device_edges(graph);
  device_allocation<device_edge> device_edges(host_edges.size());
  device_allocation<int> device_parent(
      static_cast<std::size_t>(graph.vertex_count()));
  device_allocation<std::uint64_t> device_best(
      static_cast<std::size_t>(graph.vertex_count()));
  device_allocation<int> device_admitted_edge_indices(
      static_cast<std::size_t>(std::max(1, graph.vertex_count() - 1)));
  device_allocation<int> device_admitted_count(1);
  device_allocation<int> device_changed(1);

  const int block_size = 256;
  const int vertex_grid = (graph.vertex_count() + block_size - 1) / block_size;
  const int edge_grid =
      (static_cast<int>(graph.edges().size()) + block_size - 1) / block_size;

  const auto h2d_start = clock::now();
  check_cuda(cudaMemcpy(device_edges.buffer().data(), host_edges.span().data(),
                        sizeof(device_edge) * host_edges.size(),
                        cudaMemcpyHostToDevice),
             "copying edges to device failed");
  check_cuda(cudaMemset(device_admitted_count.buffer().data(), 0, sizeof(int)),
             "resetting admitted edge count failed");
  initialize_parent_kernel<<<vertex_grid, block_size>>>(
      device_parent.buffer().data(), graph.vertex_count());
  check_cuda(cudaGetLastError(), "initializing device parents failed");
  check_cuda(cudaDeviceSynchronize(), "synchronizing CUDA setup failed");
  profile.host_to_device_seconds +=
      std::chrono::duration<double>(clock::now() - h2d_start).count();

  int host_changed = 0;
  int host_admitted_count = 0;
  int rounds = 0;
  while (true) {
    ++rounds;
    const auto scan_start = clock::now();
    reset_round_state_kernel<<<1, 1>>>(device_changed.buffer().data());
    initialize_best_kernel<<<vertex_grid, block_size>>>(
        device_best.buffer().data(), graph.vertex_count());
    scan_edges_kernel<<<edge_grid, block_size>>>(
        device_edges.buffer().data(), static_cast<int>(graph.edges().size()),
        device_parent.buffer().data(), device_best.buffer().data());
    check_cuda(cudaGetLastError(), "scanning edges on device failed");
    check_cuda(cudaDeviceSynchronize(), "synchronizing CUDA scan failed");
    profile.scan_seconds +=
        std::chrono::duration<double>(clock::now() - scan_start).count();

    const auto contract_start = clock::now();
    contract_candidates_kernel<<<vertex_grid, block_size>>>(
        device_best.buffer().data(), graph.vertex_count(),
        device_edges.buffer().data(), device_parent.buffer().data(),
        device_admitted_edge_indices.buffer().data(),
        device_admitted_count.buffer().data(), device_changed.buffer().data());
    check_cuda(cudaGetLastError(), "contracting candidates on device failed");
    check_cuda(cudaMemcpy(&host_changed, device_changed.buffer().data(),
                          sizeof(int), cudaMemcpyDeviceToHost),
               "copying CUDA changed flag failed");
    check_cuda(cudaMemcpy(&host_admitted_count,
                          device_admitted_count.buffer().data(), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "copying CUDA admitted count failed");
    profile.contract_seconds +=
        std::chrono::duration<double>(clock::now() - contract_start).count();

    const auto compress_start = clock::now();
    compress_all_kernel<<<vertex_grid, block_size>>>(device_parent.buffer().data(),
                                                     graph.vertex_count());
    check_cuda(cudaGetLastError(), "compressing device parents failed");
    check_cuda(cudaDeviceSynchronize(), "synchronizing CUDA compression failed");
    profile.compress_seconds +=
        std::chrono::duration<double>(clock::now() - compress_start).count();

    if (host_changed == 0 || host_admitted_count >= graph.vertex_count() - 1) {
      break;
    }
  }

  const auto d2h_start = clock::now();
  std::vector<int> admitted_indices(static_cast<std::size_t>(
      std::max(0, host_admitted_count)));
  if (!admitted_indices.empty()) {
    check_cuda(cudaMemcpy(admitted_indices.data(),
                          device_admitted_edge_indices.buffer().data(),
                          sizeof(int) * admitted_indices.size(),
                          cudaMemcpyDeviceToHost),
               "copying admitted CUDA edge indices failed");
  }
  profile.device_to_host_seconds +=
      std::chrono::duration<double>(clock::now() - d2h_start).count();

  cuda_result result;
  result.rounds = rounds;
  result.edges.reserve(admitted_indices.size());
  for (const int edge_index : admitted_indices) {
    const mst::core::edge edge =
        graph.edges()[static_cast<std::size_t>(edge_index)];
    result.edges.push_back(mst::core::mst_edge{edge});
    result.total_weight += edge.weight.value();
  }
  return result;
}

} // namespace

} // namespace mst::backend::cuda_backend

int main() {
  using namespace mst::backend::cuda_backend;
  using clock = std::chrono::steady_clock;

  const auto total_start = clock::now();

  const mst::app::selected_graph selected = mst::app::select_graph_from_env();
  const mst::core::validated_graph graph =
      mst::core::validate(selected.graph);
  cuda_profile profile;

  const auto mst_start = clock::now();
  const cuda_result result = run_boruvka_on_device(graph, profile);
  const auto mst_end = clock::now();
  const auto total_end = clock::now();

  std::cout << "CUDA Boruvka MST\n";
  std::cout << mst::core::mst_summary(result.edges, result.total_weight);
  mst::visualization::render_graph_with_mst(graph, result.edges,
                                            result.total_weight);
  const mst::boruvka::verification_result verification =
      mst::boruvka::verify_against_sequential_cpu(graph, result.edges,
                                                  result.total_weight);
  if (verification.success) {
    std::cout << "Sequential CPU verification: passed\n";
  } else {
    std::cerr << "Sequential CPU verification failed: expected weight "
              << verification.expected_total_weight << " with "
              << verification.expected_edge_count << " edges, got weight "
              << verification.actual_total_weight << " with "
              << verification.actual_edge_count << " edges\n";
  }

  int device_count = 0;
  int active_device = 0;
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceCount(&device_count), "querying CUDA device count failed");
  check_cuda(cudaGetDevice(&active_device), "querying active CUDA device failed");
  check_cuda(cudaGetDeviceProperties(&properties, active_device),
             "querying CUDA device properties failed");

  std::ostringstream report;
  report << "{\n";
  report << mst::reporting::common_metadata_json("cuda",
                                                 verification.success)
         << ",\n";
  report << mst::app::graph_metadata_json(selected) << ",\n";
  report << "  \"timings\": {\n";
  report << "    \"total_seconds\": "
         << std::chrono::duration<double>(total_end - total_start).count()
         << ",\n";
  report << "    \"mst_loop_seconds\": "
         << std::chrono::duration<double>(mst_end - mst_start).count() << ",\n";
  report << "    \"host_to_device_seconds\": " << profile.host_to_device_seconds
         << ",\n";
  report << "    \"scan_seconds\": " << profile.scan_seconds << ",\n";
  report << "    \"contract_seconds\": " << profile.contract_seconds << ",\n";
  report << "    \"compress_seconds\": " << profile.compress_seconds << ",\n";
  report << "    \"device_to_host_seconds\": " << profile.device_to_host_seconds
         << "\n";
  report << "  },\n";
  report << "  \"capabilities\": {\n";
  report << "    \"device_count\": " << device_count << ",\n";
  report << "    \"active_device\": " << active_device << ",\n";
  report << "    \"device_name\": \""
         << mst::reporting::json_escape(properties.name) << "\",\n";
  report << "    \"compute_capability_major\": " << properties.major << ",\n";
  report << "    \"compute_capability_minor\": " << properties.minor << ",\n";
  report << "    \"global_memory_bytes\": " << properties.totalGlobalMem << ",\n";
  report << "    \"multiprocessor_count\": "
         << properties.multiProcessorCount << ",\n";
  report << "    \"max_threads_per_block\": "
         << properties.maxThreadsPerBlock << ",\n";
  report << "    \"warp_size\": " << properties.warpSize << "\n";
  report << "  },\n";
  report << "  \"mst\": {\n";
  report << "    \"vertex_count\": " << graph.vertex_count() << ",\n";
  report << "    \"input_edge_count\": " << graph.edges().size() << ",\n";
  report << "    \"selected_edge_count\": " << result.edges.size() << ",\n";
  report << "    \"rounds\": " << result.rounds << ",\n";
  report << "    \"total_weight\": " << result.total_weight << "\n";
  report << "  },\n";
  mst::boruvka::write_verification_json(report, verification);
  report << "\n";
  report << "}\n";
  mst::reporting::write_report_from_env(report.str());
  return verification.success ? EXIT_SUCCESS : EXIT_FAILURE;
}
