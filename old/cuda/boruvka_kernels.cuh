#pragma once

#include <cstdint>

#include "cuda/device_dsu.cuh"

namespace mst::backend::cuda_backend {

/// Versione "piatta" di un arco per il device: `int` grezzi al posto dei
/// tipi forti dell'host, che `cudaMemcpy` e il codice `__device__` non capirebbero.
struct device_edge {
  int u;
  int v;
  int weight;
};

/// "Vuota" come `mst::core::empty_candidate_key`: tutti i bit a uno, sempre
/// più grande di qualsiasi chiave reale.
inline constexpr std::uint64_t cuda_empty_candidate_key = 0xffffffffffffffffULL;

/// Stessa codifica di `mst::core::make_candidate_key` (peso alto, indice
/// basso): così l'`atomicMin` a 64 bit sul device produce lo stesso minimo
/// "logico" (peso, poi indice) degli altri backend, bit per bit.
__device__ std::uint64_t pack_candidate_key_device(int weight, int edge_index) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(weight)) << 32) |
         static_cast<std::uint64_t>(static_cast<std::uint32_t>(edge_index));
}

/// Inverso di `pack_candidate_key_device`: tira fuori l'indice arco dai 32 bit bassi.
__device__ int edge_index_from_candidate_key_device(std::uint64_t key) {
  return static_cast<int>(key & 0xffffffffULL);
}

/// Un thread per vertice: ognuno parte come radice di sé stesso. Gira una sola volta, prima del primo round.
__global__ void initialize_parent_kernel(int* parent, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    parent[index] = index;
  }
}

/// Resetta `changed` (lo fa un solo thread) e riporta i minimi per componente
/// al valore "vuoto", pronti per i prossimi `atomicMin`.
__global__ void initialize_round_kernel(std::uint64_t* best, int count, int* changed) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index == 0) {
    *changed = 0;
  }
  if (index < count) {
    best[index] = cuda_empty_candidate_key;
  }
}

/// Fasi 1+2 insieme, un thread per arco: trova le radici (sola lettura) e se
/// diverse fa `atomicMin` su `best[componente]` — riduzione gratis via
/// hardware. `collision_count` è solo telemetria sugli overwrite.
__global__ void scan_edges_kernel(const device_edge* edges, int edge_count,
                                  const int* parent, std::uint64_t* best,
                                  unsigned long long* collision_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= edge_count) {
    return;
  }

  const device_edge edge = edges[index];
  const int left_root = find_root_device_read_only(parent, edge.u);
  const int right_root = find_root_device_read_only(parent, edge.v);
  if (left_root == right_root) {
    return;
  }

  const std::uint64_t packed = pack_candidate_key_device(edge.weight, index);
  const unsigned long long old_left =
      atomicMin(reinterpret_cast<unsigned long long*>(&best[left_root]),
                static_cast<unsigned long long>(packed));
  if (old_left != cuda_empty_candidate_key) {
    atomicAdd(collision_count, 1ULL);
  }
  const unsigned long long old_right =
      atomicMin(reinterpret_cast<unsigned long long*>(&best[right_root]),
                static_cast<unsigned long long>(packed));
  if (old_right != cuda_empty_candidate_key) {
    atomicAdd(collision_count, 1ULL);
  }
}

/// Fase 3, un thread per componente: se ha un candidato prova a fonderlo con
/// `unite_device`. Solo chi riesce davvero registra l'arco (`atomicAdd` su
/// `admitted_count`) e segnala il cambiamento all'host con `atomicExch`.
__global__ void contract_candidates_kernel(const std::uint64_t* best, int vertex_count,
                                           const device_edge* edges, int* parent,
                                           int* admitted_edge_indices,
                                           int* admitted_count, int* changed) {
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

/// Fase 4: un thread per vertice si aggancia alla radice corrente — l'analogo
/// device di `compress_all_parallel` nell'OpenMP.
__global__ void compress_all_kernel(int* parent, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    parent[index] = find_root_device(parent, index);
  }
}

} // namespace mst::backend::cuda_backend
