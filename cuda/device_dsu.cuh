#pragma once

namespace mst::backend::cuda_backend {

__device__ int find_root_device(int *parent, int vertex) {
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

__device__ bool unite_device(int *parent, int left_vertex, int right_vertex) {
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
