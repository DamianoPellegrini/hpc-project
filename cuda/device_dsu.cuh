#pragma once

namespace mst::backend::cuda_backend {

// DSU lock-free sul device: l'analogo CUDA di `parallel_disjoint_set`, ma
// con `int*` grezzo e atomics di CUDA al posto degli `std::atomic`.

/// Path-halving con `atomicCAS`: se fallisce va bene lo stesso, vuol dire che
/// qualcun altro l'ha già fatto. Scrive `parent`, quindi solo dove è sicuro in concorrenza.
__device__ int find_root_device(int* parent, int vertex) {
  int current = vertex;
  while (true) {
    const int parent_value = parent[current];
    const int grandparent = parent[parent_value];
    if (parent_value == grandparent) {
      return parent_value;
    }
    atomicCAS(&parent[current], parent_value, grandparent);
    current = parent_value;
  }
}

/// Come sopra ma di sola lettura, senza scrivere `parent`: usata dalla
/// scansione degli archi per non avere corse con la contrazione dello
/// stesso round (l'equivalente CUDA dello snapshot immutabile OpenMP).
__device__ int find_root_device_read_only(const int* parent, int vertex) {
  int current = vertex;
  while (true) {
    const int parent_value = parent[current];
    const int grandparent = parent[parent_value];
    if (parent_value == grandparent) {
      return parent_value;
    }
    current = parent_value;
  }
}

/// Come la `unite` lock-free OpenMP: radice minore come genitore (regola
/// uguale per tutti, niente cicli), `atomicCAS` e ritenta se fallisce.
__device__ bool unite_device(int* parent, int left_vertex, int right_vertex) {
  while (true) {
    const int left_root = find_root_device(parent, left_vertex);
    const int right_root = find_root_device(parent, right_vertex);
    if (left_root == right_root) {
      return false;
    }

    const int parent_root = left_root < right_root ? left_root : right_root;
    const int child_root = left_root < right_root ? right_root : left_root;
    if (atomicCAS(&parent[child_root], child_root, parent_root) == child_root) {
      return true;
    }
  }
}

} // namespace mst::backend::cuda_backend
