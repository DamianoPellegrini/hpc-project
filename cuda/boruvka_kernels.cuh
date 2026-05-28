#pragma once

#include <cstdint>

#include "cuda/device_dsu.cuh"

namespace mst::backend::cuda_backend {

struct device_edge {
  int u;
  int v;
  int weight;
};

inline constexpr std::uint64_t cuda_empty_candidate_key =
    0xffffffffffffffffULL;

__device__ std::uint64_t pack_candidate_key_device(int weight, int edge_index) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(weight)) << 32) |
         static_cast<std::uint64_t>(static_cast<std::uint32_t>(edge_index));
}

__device__ int edge_index_from_candidate_key_device(std::uint64_t key) {
  return static_cast<int>(key & 0xffffffffULL);
}

__global__ void initialize_parent_kernel(int *parent, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    parent[index] = index;
  }
}

__global__ void initialize_best_kernel(std::uint64_t *best, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    best[index] = cuda_empty_candidate_key;
  }
}

__global__ void reset_round_state_kernel(int *changed) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *changed = 0;
  }
}

__global__ void scan_edges_kernel(const device_edge *edges, int edge_count,
                                  int *parent, std::uint64_t *best) {
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

  const std::uint64_t packed = pack_candidate_key_device(edge.weight, index);
  atomicMin(reinterpret_cast<unsigned long long *>(&best[left_root]),
            static_cast<unsigned long long>(packed));
  atomicMin(reinterpret_cast<unsigned long long *>(&best[right_root]),
            static_cast<unsigned long long>(packed));
}

__global__ void contract_candidates_kernel(
    const std::uint64_t *best, int vertex_count, const device_edge *edges,
    int *parent, int *admitted_edge_indices, int *admitted_count,
    int *changed) {
  const int component = blockIdx.x * blockDim.x + threadIdx.x;
  if (component >= vertex_count) {
    return;
  }

  const std::uint64_t key = best[component];
  if (key == cuda_empty_candidate_key) {
    return;
  }

  const int edge_index = edge_index_from_candidate_key_device(key);
  const device_edge edge = edges[edge_index];
  if (unite_device(parent, edge.u, edge.v)) {
    const int out = atomicAdd(admitted_count, 1);
    admitted_edge_indices[out] = edge_index;
    atomicExch(changed, 1);
  }
}

__global__ void compress_all_kernel(int *parent, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    parent[index] = find_root_device(parent, index);
  }
}

} // namespace mst::backend::cuda_backend
