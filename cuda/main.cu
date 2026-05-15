#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/core/summary.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/memory/buffer.hpp"
#include "mst/visualization/render_graph.hpp"

namespace mst::backend::cuda_backend {

namespace {

struct device_edge {
  int u;
  int v;
  int weight;
};

constexpr std::uint64_t empty_candidate_key =
    std::numeric_limits<std::uint64_t>::max();

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

__device__ int find_root_device(const int *parent, int vertex) {
  int current = vertex;
  while (parent[current] != current) {
    current = parent[current];
  }
  return current;
}

__device__ std::uint64_t pack_candidate(int weight, int u, int v) {
  const int left = u < v ? u : v;
  const int right = u < v ? v : u;
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(weight)) << 32) |
         (static_cast<std::uint64_t>(static_cast<std::uint16_t>(left)) << 16) |
         static_cast<std::uint64_t>(static_cast<std::uint16_t>(right));
}

__global__ void initialize_best_kernel(std::uint64_t *best, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    best[index] = empty_candidate_key;
  }
}

__global__ void scan_edges_kernel(const device_edge *edges, int edge_count,
                                  const int *parent, std::uint64_t *best) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= edge_count) {
    return;
  }

  const device_edge edge = edges[index];
  const int left_root = find_root_device(parent, edge.u);
  const int right_root = find_root_device(parent, edge.v);
  if (left_root == right_root) {
    return;
  }

  const std::uint64_t packed = pack_candidate(edge.weight, edge.u, edge.v);
  atomicMin(reinterpret_cast<unsigned long long *>(&best[left_root]),
            static_cast<unsigned long long>(packed));
  atomicMin(reinterpret_cast<unsigned long long *>(&best[right_root]),
            static_cast<unsigned long long>(packed));
}

mst::memory::host_buffer<device_edge, mst::memory::host_memory>
make_device_edges(const mst::core::validated_graph &graph) {
  std::vector<device_edge> edges;
  edges.reserve(graph.edges().size());
  for (const mst::core::edge &edge : graph.edges()) {
    edges.push_back({mst::core::as_index(edge.u), mst::core::as_index(edge.v),
                     mst::core::as_value(edge.weight)});
  }
  return mst::memory::host_buffer<device_edge, mst::memory::host_memory>{
      std::move(edges)};
}

std::vector<int> pack_snapshot(const mst::dsu::parent_snapshot &snapshot) {
  std::vector<int> packed;
  packed.reserve(snapshot.parent().size());
  for (const mst::core::vertex_id vertex : snapshot.parent()) {
    packed.push_back(mst::core::as_index(vertex));
  }
  return packed;
}

std::vector<mst::core::maybe_candidate_edge>
unpack_candidates(const std::vector<std::uint64_t> &packed) {
  std::vector<mst::core::maybe_candidate_edge> candidates(packed.size());
  for (std::size_t index = 0; index < packed.size(); ++index) {
    if (packed[index] == empty_candidate_key) {
      continue;
    }
    const int weight = static_cast<int>(packed[index] >> 32);
    const int u = static_cast<int>((packed[index] >> 16) & 0xffffULL);
    const int v = static_cast<int>(packed[index] & 0xffffULL);
    candidates[index] = mst::core::candidate_edge{
        mst::core::edge{mst::core::make_vertex_id(u),
                        mst::core::make_vertex_id(v),
                        mst::core::make_edge_weight(weight)}};
  }
  return candidates;
}

std::vector<mst::core::maybe_candidate_edge>
scan_candidates_on_device(const mst::core::validated_graph &graph,
                          const mst::dsu::parent_snapshot &snapshot) {
  auto host_edges = make_device_edges(graph);
  const std::vector<int> host_parent = pack_snapshot(snapshot);
  std::vector<std::uint64_t> host_best(static_cast<std::size_t>(graph.vertex_count()),
                                       empty_candidate_key);

  device_allocation<device_edge> device_edges(host_edges.size());
  device_allocation<int> device_parent(host_parent.size());
  device_allocation<std::uint64_t> device_best(host_best.size());

  check_cuda(cudaMemcpy(device_edges.buffer().data(), host_edges.span().data(),
                        sizeof(device_edge) * host_edges.size(),
                        cudaMemcpyHostToDevice),
             "copying edges to device failed");
  check_cuda(cudaMemcpy(device_parent.buffer().data(), host_parent.data(),
                        sizeof(int) * host_parent.size(),
                        cudaMemcpyHostToDevice),
             "copying parent snapshot to device failed");

  const int block_size = 256;
  const int best_grid =
      (graph.vertex_count() + block_size - 1) / block_size;
  const int edge_grid =
      (static_cast<int>(graph.edges().size()) + block_size - 1) / block_size;

  initialize_best_kernel<<<best_grid, block_size>>>(device_best.buffer().data(),
                                                    graph.vertex_count());
  check_cuda(cudaGetLastError(), "initializing best candidates failed");

  scan_edges_kernel<<<edge_grid, block_size>>>(
      device_edges.buffer().data(), static_cast<int>(graph.edges().size()),
      device_parent.buffer().data(), device_best.buffer().data());
  check_cuda(cudaGetLastError(), "scanning edges on device failed");
  check_cuda(cudaDeviceSynchronize(), "synchronizing CUDA scan failed");

  check_cuda(cudaMemcpy(host_best.data(), device_best.buffer().data(),
                        sizeof(std::uint64_t) * host_best.size(),
                        cudaMemcpyDeviceToHost),
             "copying candidate minima to host failed");
  return unpack_candidates(host_best);
}

} // namespace

} // namespace mst::backend::cuda_backend

int main() {
  using namespace mst::backend::cuda_backend;

  const mst::core::validated_graph graph =
      mst::core::validate(mst::core::make_test_graph());
  if (graph.vertex_count() > 65535) {
    std::cerr << "CUDA backend currently requires vertex_count <= 65535 for "
                 "deterministic packed candidate encoding.\n";
    return EXIT_FAILURE;
  }

  mst::dsu::disjoint_set<mst::core::uncompressed_parents> dsu(
      graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;

  while (dsu.component_count() > 1) {
    const mst::dsu::parent_snapshot snapshot = dsu.compressed_snapshot();
    const auto best = scan_candidates_on_device(graph, snapshot);

    bool changed = false;
    for (const auto &candidate : best) {
      if (!candidate) {
        continue;
      }
      if (auto admitted = dsu.unite(*candidate)) {
        mst_edges.push_back(*admitted);
        total_weight += mst::core::as_value(admitted->value.weight);
        changed = true;
      }
    }

    if (!changed) {
      break;
    }
  }

  std::cout << "CUDA Boruvka MST\n";
  std::cout << mst::core::mst_summary(mst_edges, total_weight);
  mst::visualization::render_graph_with_mst(graph, mst_edges, total_weight);
  return EXIT_SUCCESS;
}
