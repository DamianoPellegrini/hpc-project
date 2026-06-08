#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cuda/boruvka_kernels.cuh"
#include "mst/app/backend_app.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/memory/buffer.hpp"
#include "mst/reporting/json_report.hpp"

namespace mst::backend::cuda_backend {

namespace {

/// Tempi per fase più contatori CUDA (collisioni, kernel lanciati...): per il report.
struct cuda_profile {
  double setup_seconds = 0.0;
  double host_to_device_seconds = 0.0;
  double initialization_seconds = 0.0;
  double round_prepare_seconds = 0.0;
  double scan_seconds = 0.0;
  double contract_kernel_seconds = 0.0;
  double contract_copy_seconds = 0.0;
  double contract_seconds = 0.0;
  double compress_seconds = 0.0;
  double device_to_host_seconds = 0.0;
  double kernel_launch_overhead_estimated_seconds = 0.0;
  double unattributed_residual_seconds = 0.0;
  std::uint64_t cuda_atomic_min_collision_count = 0;
  int kernel_launch_count = 0;
  std::string host_edge_memory_mode = "unknown";

  double device_algorithm_seconds() const noexcept {
    return round_prepare_seconds + scan_seconds + contract_kernel_seconds +
           compress_seconds;
  }

  double profiled_backend_seconds() const noexcept {
    return setup_seconds + host_to_device_seconds + initialization_seconds +
           round_prepare_seconds + scan_seconds + contract_kernel_seconds +
           contract_copy_seconds + compress_seconds + device_to_host_seconds;
  }

  double allocation_and_overhead_seconds() const noexcept {
    return setup_seconds + host_to_device_seconds + initialization_seconds +
           device_to_host_seconds + contract_copy_seconds +
           kernel_launch_overhead_estimated_seconds;
  }

  double atomic_min_collision_rate(std::size_t edge_count, int rounds) const noexcept {
    const double attempted_updates =
        static_cast<double>(std::max<std::size_t>(1, edge_count)) *
        static_cast<double>(std::max(1, rounds)) * 2.0;
    return static_cast<double>(cuda_atomic_min_collision_count) / attempted_updates;
  }
};

/// Risultato del Boruvka su device: archi ammessi, peso totale, round eseguiti.
struct cuda_result {
  std::vector<mst::core::mst_edge> edges;
  int total_weight = 0;
  int rounds = 0;
};

/// Trasforma un errore CUDA in un'eccezione C++ col messaggio giusto.
inline void check_cuda(cudaError_t status, const char* message) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
  }
}

/// Wrapper RAII su `cudaEvent_t`: crea alla costruzione, distrugge nel distruttore.
class cuda_event {
public:
  cuda_event() {
    check_cuda(cudaEventCreate(&event_), "creating CUDA event failed");
  }

  cuda_event(const cuda_event&) = delete;
  cuda_event& operator=(const cuda_event&) = delete;

  ~cuda_event() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  cudaEvent_t get() const noexcept {
    return event_;
  }

private:
  cudaEvent_t event_ = nullptr;
};

#ifndef MST_DEFAULT_CUDA_HOST_MEMORY
#define MST_DEFAULT_CUDA_HOST_MEMORY "pinned"
#endif

/// Preferenza su come allocare la memoria host per gli archi, fissata a
/// compile time dalla macro `MST_DEFAULT_CUDA_HOST_MEMORY` (vedi CMakeLists.txt).
enum class cuda_host_edge_memory_preference {
  pageable,
  pinned,
  mapped_zero_copy,
};

constexpr cuda_host_edge_memory_preference
compile_time_host_edge_memory_preference() noexcept {
  constexpr std::string_view mode = MST_DEFAULT_CUDA_HOST_MEMORY;
  if (mode == "pageable") {
    return cuda_host_edge_memory_preference::pageable;
  }
  if (mode == "zero_copy" || mode == "mapped_zero_copy") {
    return cuda_host_edge_memory_preference::mapped_zero_copy;
  }
  return cuda_host_edge_memory_preference::pinned;
}

/// Modalità ottenuta davvero (può differire dalla preferenza, se è scattato un
/// fallback).
enum class cuda_host_edge_memory_mode {
  pageable,
  pinned,
  mapped_zero_copy,
};

const char* cuda_host_edge_memory_mode_name(cuda_host_edge_memory_mode mode) noexcept {
  switch (mode) {
  case cuda_host_edge_memory_mode::pageable:
    return "pageable";
  case cuda_host_edge_memory_mode::pinned:
    return "pinned";
  case cuda_host_edge_memory_mode::mapped_zero_copy:
    return "mapped_zero_copy";
  }
  return "unknown";
}

void enable_mapped_host_memory_if_requested(
    cuda_host_edge_memory_preference preference) {
  if (preference != cuda_host_edge_memory_preference::mapped_zero_copy) {
    return;
  }

  const cudaError_t status = cudaSetDeviceFlags(cudaDeviceMapHost);
  if (status == cudaSuccess || status == cudaErrorSetOnActiveProcess) {
    if (status == cudaErrorSetOnActiveProcess) {
      cudaGetLastError();
    }
    return;
  }
  cudaGetLastError();
}

double elapsed_seconds(cudaEvent_t start, cudaEvent_t stop) {
  float milliseconds = 0.0f;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop),
             "measuring CUDA event elapsed time failed");
  return static_cast<double>(milliseconds) / 1000.0;
}

/// Buffer host per gli archi, RAII: `make` prova zero-copy, poi pinned, poi
/// pageable, ricordando in `mode_` cosa ha funzionato. In zero-copy il device
/// legge direttamente il puntatore mappato, niente `device_allocation` a parte.
class host_edge_storage {
public:
  host_edge_storage() = default;

  host_edge_storage(const host_edge_storage&) = delete;
  host_edge_storage& operator=(const host_edge_storage&) = delete;

  host_edge_storage(host_edge_storage&& other) noexcept
      : pageable_edges_(std::move(other.pageable_edges_)),
        host_edges_(other.host_edges_),
        mapped_device_edges_(other.mapped_device_edges_), size_(other.size_),
        mode_(other.mode_) {
    other.host_edges_ = nullptr;
    other.mapped_device_edges_ = nullptr;
    other.size_ = 0;
  }

  ~host_edge_storage() {
    release();
  }

  static host_edge_storage make(const mst::core::validated_graph& graph,
                                cuda_host_edge_memory_preference preference) {
    host_edge_storage storage;
    storage.size_ = graph.edges().size();

    if (preference == cuda_host_edge_memory_preference::mapped_zero_copy &&
        storage.try_allocate_cuda_host(cudaHostAllocMapped,
                                       cuda_host_edge_memory_mode::mapped_zero_copy)) {
      storage.fill(graph);
      return storage;
    }

    if (preference != cuda_host_edge_memory_preference::pageable &&
        storage.try_allocate_cuda_host(cudaHostAllocDefault,
                                       cuda_host_edge_memory_mode::pinned)) {
      storage.fill(graph);
      return storage;
    }

    storage.pageable_edges_.resize(storage.size_);
    storage.mode_ = cuda_host_edge_memory_mode::pageable;
    storage.fill(graph);
    return storage;
  }

  device_edge* host_data() noexcept {
    if (host_edges_ != nullptr) {
      return host_edges_;
    }
    return pageable_edges_.data();
  }

  const device_edge* mapped_device_data() const noexcept {
    return mapped_device_edges_;
  }

  std::size_t size() const noexcept {
    return size_;
  }

  bool uses_mapped_zero_copy() const noexcept {
    return mode_ == cuda_host_edge_memory_mode::mapped_zero_copy;
  }

  const char* mode_name() const noexcept {
    return cuda_host_edge_memory_mode_name(mode_);
  }

private:
  bool try_allocate_cuda_host(unsigned int flags, cuda_host_edge_memory_mode mode) {
    if (size_ == 0) {
      mode_ = mode;
      return true;
    }

    void* raw = nullptr;
    const cudaError_t status = cudaHostAlloc(&raw, sizeof(device_edge) * size_, flags);
    if (status != cudaSuccess) {
      cudaGetLastError();
      return false;
    }

    host_edges_ = static_cast<device_edge*>(raw);
    mode_ = mode;

    if (mode_ == cuda_host_edge_memory_mode::mapped_zero_copy) {
      device_edge* device_pointer = nullptr;
      const cudaError_t map_status =
          cudaHostGetDevicePointer(&device_pointer, host_edges_, 0);
      if (map_status != cudaSuccess) {
        cudaGetLastError();
        release();
        return false;
      }
      mapped_device_edges_ = device_pointer;
    }

    return true;
  }

  void fill(const mst::core::validated_graph& graph) {
    device_edge* target = host_data();
    for (std::size_t index = 0; index < graph.edges().size(); ++index) {
      const mst::core::edge& edge = graph.edges()[index];
      target[index] = {edge.u.value(), edge.v.value(), edge.weight.value()};
    }
  }

  void release() noexcept {
    if (host_edges_ != nullptr) {
      cudaFreeHost(host_edges_);
      host_edges_ = nullptr;
      mapped_device_edges_ = nullptr;
    }
  }

  std::vector<device_edge> pageable_edges_;
  device_edge* host_edges_ = nullptr;
  device_edge* mapped_device_edges_ = nullptr;
  std::size_t size_ = 0;
  cuda_host_edge_memory_mode mode_ = cuda_host_edge_memory_mode::pageable;
};

/// RAII su `cudaMalloc`/`cudaFree`: alloca alla costruzione, libera alla
/// distruzione (move-only), ed espone un `mst::memory::device_buffer` —
/// il tipo leggero che i kernel si aspettano.
template <class value_t> class device_allocation {
public:
  explicit device_allocation(std::size_t size) : size_(size) {
    if (size_ == 0) {
      return;
    }
    check_cuda(cudaMalloc(&data_, sizeof(value_t) * size_), "cudaMalloc failed");
  }

  device_allocation(const device_allocation&) = delete;
  device_allocation& operator=(const device_allocation&) = delete;

  device_allocation(device_allocation&& other) noexcept
      : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }

  device_allocation& operator=(device_allocation&& other) noexcept {
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

  ~device_allocation() {
    release();
  }

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

  value_t* data_ = nullptr;
  std::size_t size_ = 0;
};

/// Esegue Boruvka sulla GPU. A differenza di OpenMP/MPI il ciclo vive quasi
/// tutto sul device: ogni round lancia quattro kernel (prepara, scansiona,
/// contrai, comprimi) e l'host legge solo `changed` e `admitted_count` —
/// il minimo per non far dominare la latenza del bus PCIe sul calcolo.
cuda_result run_boruvka_on_device(const mst::core::validated_graph& graph,
                                  cuda_profile& profile) {
  using clock = std::chrono::steady_clock;

  constexpr cuda_host_edge_memory_preference host_memory_preference =
      compile_time_host_edge_memory_preference();
  enable_mapped_host_memory_if_requested(host_memory_preference);

  const auto setup_start = clock::now();
  host_edge_storage host_edges = host_edge_storage::make(graph, host_memory_preference);
  profile.host_edge_memory_mode = host_edges.mode_name();
  device_allocation<device_edge> device_edges(
      host_edges.uses_mapped_zero_copy() ? 0 : host_edges.size());
  device_allocation<int> device_parent(static_cast<std::size_t>(graph.vertex_count()));
  device_allocation<std::uint64_t> device_best(
      static_cast<std::size_t>(graph.vertex_count()));
  device_allocation<int> device_admitted_edge_indices(
      static_cast<std::size_t>(std::max(1, graph.vertex_count() - 1)));
  device_allocation<int> device_admitted_count(1);
  device_allocation<int> device_changed(1);
  device_allocation<unsigned long long> device_atomic_min_collision_count(1);

  const int block_size = 256;
  const int vertex_grid = (graph.vertex_count() + block_size - 1) / block_size;
  const int edge_grid =
      (static_cast<int>(graph.edges().size()) + block_size - 1) / block_size;
  profile.setup_seconds +=
      std::chrono::duration<double>(clock::now() - setup_start).count();

  // Gli archi non cambiano mai: si copiano una volta sola, prima del primo
  // round (in zero-copy nemmeno questo serve, il device legge da host via PCIe).
  const device_edge* device_edge_data = nullptr;
  const auto h2d_start = clock::now();
  if (host_edges.uses_mapped_zero_copy()) {
    device_edge_data = host_edges.mapped_device_data();
  } else {
    device_edge_data = device_edges.buffer().data();
    if (host_edges.size() > 0) {
      check_cuda(cudaMemcpy(device_edges.buffer().data(), host_edges.host_data(),
                            sizeof(device_edge) * host_edges.size(),
                            cudaMemcpyHostToDevice),
                 "copying edges to device failed");
      check_cuda(cudaDeviceSynchronize(), "synchronizing CUDA host-device copy failed");
    }
  }
  profile.host_to_device_seconds +=
      std::chrono::duration<double>(clock::now() - h2d_start).count();

  // Una tantum, prima del ciclo: DSU e contatori azzerati (come la
  // costruzione del DSU singoletto negli altri backend).
  const auto initialization_start = clock::now();
  check_cuda(cudaMemset(device_admitted_count.buffer().data(), 0, sizeof(int)),
             "resetting admitted edge count failed");
  check_cuda(cudaMemset(device_atomic_min_collision_count.buffer().data(), 0,
                        sizeof(unsigned long long)),
             "resetting atomicMin collision counter failed");
  initialize_parent_kernel<<<vertex_grid, block_size>>>(device_parent.buffer().data(),
                                                        graph.vertex_count());
  profile.kernel_launch_count += 1;
  check_cuda(cudaGetLastError(), "initializing device parents failed");
  check_cuda(cudaDeviceSynchronize(), "synchronizing CUDA setup failed");
  profile.initialization_seconds +=
      std::chrono::duration<double>(clock::now() - initialization_start).count();

  int host_changed = 0;
  int host_admitted_count = 0;
  int rounds = 0;
  cuda_event round_start;
  cuda_event prepare_done;
  cuda_event scan_done;
  cuda_event contract_start;
  cuda_event contract_done;
  cuda_event compress_start;
  cuda_event compress_done;
  // Quattro kernel in sequenza sullo stream di default (quindi già ordinati
  // fra loro, niente sync intermedie): prepara, scansiona+riduci, contrai,
  // comprimi. Un evento CUDA fra l'uno e l'altro misura i tempi; si
  // sincronizza una sola volta, alla fine, quando serve davvero il risultato.
  while (true) {
    ++rounds;
    check_cuda(cudaEventRecord(round_start.get()), "recording CUDA round start failed");
    // Fase 0: azzera `changed` e i minimi per componente.
    initialize_round_kernel<<<vertex_grid, block_size>>>(
        device_best.buffer().data(), graph.vertex_count(),
        device_changed.buffer().data());
    profile.kernel_launch_count += 1;
    check_cuda(cudaGetLastError(), "preparing CUDA round state failed");
    check_cuda(cudaEventRecord(prepare_done.get()),
               "recording CUDA round preparation completion failed");
    // Fasi 1+2: un thread per arco propone il candidato con `atomicMin`
    // (dettagli sulla codifica della chiave nel commento di `scan_edges_kernel`).
    scan_edges_kernel<<<edge_grid, block_size>>>(
        device_edge_data, static_cast<int>(graph.edges().size()),
        device_parent.buffer().data(), device_best.buffer().data(),
        device_atomic_min_collision_count.buffer().data());
    profile.kernel_launch_count += 1;
    check_cuda(cudaGetLastError(), "scanning edges on device failed");
    check_cuda(cudaEventRecord(scan_done.get()),
               "recording CUDA scan completion failed");

    // Fase 3: ogni componente prova ad ammettere il proprio candidato via DSU
    // lock-free.
    check_cuda(cudaEventRecord(contract_start.get()),
               "recording CUDA contract start failed");
    contract_candidates_kernel<<<vertex_grid, block_size>>>(
        device_best.buffer().data(), graph.vertex_count(), device_edge_data,
        device_parent.buffer().data(), device_admitted_edge_indices.buffer().data(),
        device_admitted_count.buffer().data(), device_changed.buffer().data());
    profile.kernel_launch_count += 1;
    check_cuda(cudaGetLastError(), "contracting candidates on device failed");
    check_cuda(cudaEventRecord(contract_done.get()),
               "recording CUDA contract completion failed");

    // Fase 4: appiattisce i cammini, pronti per il round successivo.
    check_cuda(cudaEventRecord(compress_start.get()),
               "recording CUDA compression start failed");
    compress_all_kernel<<<vertex_grid, block_size>>>(device_parent.buffer().data(),
                                                     graph.vertex_count());
    profile.kernel_launch_count += 1;
    check_cuda(cudaGetLastError(), "compressing device parents failed");
    check_cuda(cudaEventRecord(compress_done.get()),
               "recording CUDA compression completion failed");
    // Unica sincronizzazione del round: si aspetta la GPU prima di leggere tempi e
    // contatori.
    check_cuda(cudaEventSynchronize(compress_done.get()),
               "synchronizing CUDA compression failed");
    profile.round_prepare_seconds +=
        elapsed_seconds(round_start.get(), prepare_done.get());
    profile.scan_seconds += elapsed_seconds(prepare_done.get(), scan_done.get());
    const double contract_kernel_seconds =
        elapsed_seconds(contract_start.get(), contract_done.get());
    profile.contract_kernel_seconds += contract_kernel_seconds;
    profile.compress_seconds +=
        elapsed_seconds(compress_start.get(), compress_done.get());

    // Unico trasferimento device→host nel ciclo caldo: due interi, il minimo
    // per decidere se continuare (il resto si legge solo a fine esecuzione).
    const auto contract_copy_start = clock::now();
    check_cuda(cudaMemcpy(&host_changed, device_changed.buffer().data(), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "copying CUDA changed flag failed");
    check_cuda(cudaMemcpy(&host_admitted_count, device_admitted_count.buffer().data(),
                          sizeof(int), cudaMemcpyDeviceToHost),
               "copying CUDA admitted count failed");
    const double contract_copy_seconds =
        std::chrono::duration<double>(clock::now() - contract_copy_start).count();
    profile.contract_copy_seconds += contract_copy_seconds;
    profile.contract_seconds += contract_kernel_seconds + contract_copy_seconds;

    // Ci si ferma se non si è ammesso nulla (punto fisso, grafo non connesso)
    // o se l'MST ha già V-1 archi: non può servire altro.
    if (host_changed == 0 || host_admitted_count >= graph.vertex_count() - 1) {
      break;
    }
  }

  const auto d2h_start = clock::now();
  std::vector<int> admitted_indices(
      static_cast<std::size_t>(std::max(0, host_admitted_count)));
  if (!admitted_indices.empty()) {
    check_cuda(cudaMemcpy(admitted_indices.data(),
                          device_admitted_edge_indices.buffer().data(),
                          sizeof(int) * admitted_indices.size(),
                          cudaMemcpyDeviceToHost),
               "copying admitted CUDA edge indices failed");
  }
  unsigned long long host_collision_count = 0;
  check_cuda(cudaMemcpy(&host_collision_count,
                        device_atomic_min_collision_count.buffer().data(),
                        sizeof(unsigned long long), cudaMemcpyDeviceToHost),
             "copying CUDA atomicMin collision count failed");
  profile.cuda_atomic_min_collision_count =
      static_cast<std::uint64_t>(host_collision_count);
  profile.device_to_host_seconds +=
      std::chrono::duration<double>(clock::now() - d2h_start).count();

  cuda_result result;
  result.rounds = rounds;
  result.edges.reserve(admitted_indices.size());
  for (const int edge_index : admitted_indices) {
    const mst::core::edge edge = graph.edges()[static_cast<std::size_t>(edge_index)];
    result.edges.push_back(mst::core::mst_edge{edge});
    result.total_weight += edge.weight.value();
  }
  return result;
}

} // namespace

} // namespace mst::backend::cuda_backend

int main(int argc, char** argv) {
  using namespace mst::backend::cuda_backend;
  using clock = std::chrono::steady_clock;

  const mst::app::config_parse_result parsed = mst::app::parse_app_config(argc, argv);
  int config_exit_code = EXIT_FAILURE;
  if (!mst::app::handle_config_parse_result(parsed, argv[0], config_exit_code)) {
    return config_exit_code;
  }
  const mst::app::app_config& config = parsed.config;

  const auto total_start = clock::now();

  const mst::app::loaded_graph loaded = mst::app::load_graph(config);
  const mst::app::selected_graph& selected = loaded.selected;
  const mst::core::validated_graph& graph = loaded.graph;
  cuda_profile profile;

  const auto mst_start = clock::now();
  const cuda_result result = run_boruvka_on_device(graph, profile);
  const auto mst_end = clock::now();
  const auto total_end = clock::now();
  const double mst_loop_seconds =
      std::chrono::duration<double>(mst_end - mst_start).count();
  const double total_seconds =
      std::chrono::duration<double>(total_end - total_start).count();
  profile.unattributed_residual_seconds =
      std::max(0.0, mst_loop_seconds - profile.profiled_backend_seconds());
  profile.kernel_launch_overhead_estimated_seconds =
      profile.unattributed_residual_seconds;

  mst::app::print_result("CUDA Boruvka MST", config, graph, result.edges,
                         result.total_weight);
  const auto verification_start = clock::now();
  const mst::boruvka::verification_result verification =
      mst::app::verify_and_print(graph, result.edges, result.total_weight);
  const double verification_seconds =
      std::chrono::duration<double>(clock::now() - verification_start).count();

  int device_count = 0;
  int active_device = 0;
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceCount(&device_count), "querying CUDA device count failed");
  check_cuda(cudaGetDevice(&active_device), "querying active CUDA device failed");
  check_cuda(cudaGetDeviceProperties(&properties, active_device),
             "querying CUDA device properties failed");

  std::ostringstream report;
  report << "{\n";
  report << mst::reporting::common_metadata_json("cuda", verification.success) << ",\n";
  report << mst::app::graph_metadata_json(selected) << ",\n";
  report << mst::app::configuration_metadata_json(config) << ",\n";
  std::ostringstream backend_timing_fields;
  backend_timing_fields << "    \"backend\": {\n";
  backend_timing_fields << "      \"setup_seconds\": " << profile.setup_seconds
                        << ",\n";
  backend_timing_fields << "      \"host_to_device_seconds\": "
                        << profile.host_to_device_seconds << ",\n";
  backend_timing_fields << "      \"initialization_seconds\": "
                        << profile.initialization_seconds << ",\n";
  backend_timing_fields << "      \"round_prepare_seconds\": "
                        << profile.round_prepare_seconds << ",\n";
  backend_timing_fields << "      \"contract_kernel_seconds\": "
                        << profile.contract_kernel_seconds << ",\n";
  backend_timing_fields << "      \"contract_copy_seconds\": "
                        << profile.contract_copy_seconds << ",\n";
  backend_timing_fields << "      \"device_to_host_seconds\": "
                        << profile.device_to_host_seconds << ",\n";
  backend_timing_fields << "      \"device_algorithm_seconds\": "
                        << profile.device_algorithm_seconds() << ",\n";
  backend_timing_fields << "      \"profiled_backend_seconds\": "
                        << profile.profiled_backend_seconds() << ",\n";
  backend_timing_fields << "      \"unattributed_residual_seconds\": "
                        << profile.unattributed_residual_seconds << ",\n";
  backend_timing_fields << "      \"kernel_launch_overhead_estimated_seconds\": "
                        << profile.kernel_launch_overhead_estimated_seconds << ",\n";
  backend_timing_fields << "      \"kernel_launch_count\": "
                        << profile.kernel_launch_count << ",\n";
  backend_timing_fields << "      \"cuda_atomic_min_collision_count\": "
                        << profile.cuda_atomic_min_collision_count << ",\n";
  backend_timing_fields << "      \"cuda_atomic_min_collision_rate\": "
                        << profile.atomic_min_collision_rate(graph.edges().size(),
                                                             result.rounds)
                        << "\n";
  backend_timing_fields << "    }\n";
  mst::reporting::write_phase_timings_json(report,
                                           mst::reporting::phase_timing_profile{
                                               total_seconds,
                                               mst_loop_seconds,
                                               verification_seconds,
                                               profile.scan_seconds,
                                               0.0,
                                               profile.contract_seconds,
                                               profile.compress_seconds,
                                               profile.device_algorithm_seconds(),
                                           },
                                           backend_timing_fields.str());
  report << ",\n";
  mst::reporting::write_telemetry_details_json(
      report,
      mst::reporting::telemetry_details_profile{
          profile.scan_seconds,
          0.0,
          profile.contract_kernel_seconds,
          profile.allocation_and_overhead_seconds(),
          0,
          profile.kernel_launch_overhead_estimated_seconds,
          profile.cuda_atomic_min_collision_count,
          profile.atomic_min_collision_rate(graph.edges().size(), result.rounds),
          profile.unattributed_residual_seconds,
      });
  report << ",\n";
  report << "  \"capabilities\": {\n";
  report << "    \"device_count\": " << device_count << ",\n";
  report << "    \"active_device\": " << active_device << ",\n";
  report << "    \"host_edge_memory\": \""
         << mst::reporting::json_escape(profile.host_edge_memory_mode) << "\",\n";
  report << "    \"device_name\": \"" << mst::reporting::json_escape(properties.name)
         << "\",\n";
  report << "    \"compute_capability_major\": " << properties.major << ",\n";
  report << "    \"compute_capability_minor\": " << properties.minor << ",\n";
  report << "    \"global_memory_bytes\": " << properties.totalGlobalMem << ",\n";
  report << "    \"multiprocessor_count\": " << properties.multiProcessorCount << ",\n";
  report << "    \"max_threads_per_block\": " << properties.maxThreadsPerBlock << ",\n";
  report << "    \"warp_size\": " << properties.warpSize << "\n";
  report << "  },\n";
  report << mst::app::mst_metadata_json(graph, result.edges.size(), result.rounds,
                                        result.total_weight)
         << ",\n";
  mst::boruvka::write_verification_json(report, verification);
  report << "\n";
  report << "}\n";
  mst::app::write_report_if_requested(config, report.str());
  return verification.success ? EXIT_SUCCESS : EXIT_FAILURE;
}
