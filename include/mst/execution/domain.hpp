#pragma once

namespace mst::execution {

/// Shared-memory CPU thread execution domain.
struct cpu_thread_domain {};
/// Distributed MPI process execution domain.
struct mpi_process_domain {};
/// CUDA block-level execution domain.
struct gpu_block_domain {};
/// CUDA warp-level execution domain.
struct gpu_warp_domain {};

/// MPI round after parent broadcast.
struct parents_broadcasted {};
/// MPI round after local minima computation.
struct local_minima_computed {};
/// MPI round after minima reduction to the root.
struct minima_reduced {};
/// MPI round after parent synchronization.
struct parents_synchronized {};

template <class phase_t>
struct mpi_round {
  int value = 0;
};

} // namespace mst::execution
