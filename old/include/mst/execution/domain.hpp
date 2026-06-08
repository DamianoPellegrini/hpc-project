#pragma once

namespace mst::execution {

/// Dominio di esecuzione a thread su CPU a memoria condivisa (OpenMP).
struct cpu_thread_domain {};
/// Dominio di esecuzione a processi MPI distribuiti.
struct mpi_process_domain {};
/// Dominio di esecuzione a livello di block CUDA.
struct gpu_block_domain {};
/// Dominio di esecuzione a livello di warp CUDA.
struct gpu_warp_domain {};

/// Tag di fase per `mpi_round`: dopo il broadcast dei genitori del DSU.
struct parents_broadcasted {};
/// Tag di fase: dopo il calcolo dei minimi locali su ogni rank.
struct local_minima_computed {};
/// Tag di fase: dopo la riduzione (Allreduce) in minimi globali.
struct minima_reduced {};
/// Tag di fase: dopo la sincronizzazione dei genitori fra rank.
struct parents_synchronized {};

/// Contatore di round MPI: il tag `phase_t` segna a livello di tipo il punto della pipeline in cui ci si trova.
template <class phase_t> struct mpi_round {
  int value = 0;
};

} // namespace mst::execution
