#include <mpi.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <optional>
#include <sstream>
#include <vector>

#include "mst/app/backend_app.hpp"
#include "mst/boruvka/sequential_verifier.hpp"
#include "mst/core/edge.hpp"
#include "mst/core/graph.hpp"
#include "mst/dsu/disjoint_set.hpp"
#include "mst/execution/domain.hpp"
#include "mst/reporting/json_report.hpp"

namespace mst::backend::mpi {

constexpr int root_rank = 0;

/// Buffer riusati ad ogni round: `local_keys` per i minimi sulla fetta di
/// archi di questo rank, `reduced_keys` per il risultato dell'`MPI_Allreduce`.
struct mpi_workspace {
  explicit mpi_workspace(int vertex_count)
      : local_keys(static_cast<std::size_t>(vertex_count),
                   mst::core::empty_candidate_key.value()),
        reduced_keys(static_cast<std::size_t>(vertex_count),
                     mst::core::empty_candidate_key.value()) {}

  void reset_local() {
    std::fill(local_keys.begin(), local_keys.end(),
              mst::core::empty_candidate_key.value());
  }

  std::vector<std::uint64_t> local_keys;
  std::vector<std::uint64_t> reduced_keys;
};

/// Inizio (incluso) della fetta di archi `[begin, end)` di un rank: il
/// rank radice carica il grafo e lo trasmette con `broadcast_graph`, così
/// ogni processo ne ha una copia completa ma scansiona solo la sua
/// porzione (`edge_count * rank / size`), dividendo il calcolo senza
/// dover ripartire i dati arco per arco.
int edge_begin_for_rank(int edge_count, int rank, int size) {
  return (edge_count * rank) / size;
}

/// Fine (esclusa) della stessa fetta — l'inizio del blocco successivo.
int edge_end_for_rank(int edge_count, int rank, int size) {
  return (edge_count * (rank + 1)) / size;
}

/// Spacchetta gli archi in interi grezzi per la trasmissione: ogni arco
/// diventa una tripletta (estremo, estremo, peso) in un buffer piatto — MPI
/// parla solo di tipi primitivi, non dei nostri tipi forti.
std::vector<int> pack_edges(const std::vector<mst::core::edge> &edges) {
  std::vector<int> packed;
  packed.reserve(edges.size() * 3);
  for (const mst::core::edge &edge_value : edges) {
    packed.push_back(edge_value.u.value());
    packed.push_back(edge_value.v.value());
    packed.push_back(edge_value.weight.value());
  }
  return packed;
}

/// Inverso di `pack_edges`: ricostruisce gli archi tipati dalle triplette ricevute.
std::vector<mst::core::edge> unpack_edges(const std::vector<int> &packed) {
  std::vector<mst::core::edge> edges;
  edges.reserve(packed.size() / 3);
  for (std::size_t index = 0; index + 2 < packed.size(); index += 3) {
    edges.push_back({mst::core::make_vertex_id(packed[index]),
                     mst::core::make_vertex_id(packed[index + 1]),
                     mst::core::make_edge_weight(packed[index + 2])});
  }
  return edges;
}

/// Distribuisce il grafo dal rank radice a tutti gli altri con due
/// `MPI_Bcast`: prima le dimensioni (vertici, archi), poi gli archi
/// impacchettati come triplette di interi. Il radice (che ha già caricato
/// `graph_on_root`) restituisce la propria copia; gli altri rank ricevono
/// il buffer, lo sciolgono in archi tipati e validano il proprio grafo —
/// niente più generazione indipendente, una sola sorgente di verità.
mst::core::validated_graph broadcast_graph(
    const mst::core::validated_graph *graph_on_root, int rank, MPI_Comm comm) {
  int dimensions[2] = {0, 0}; // {vertex_count, edge_count}
  std::vector<int> packed;

  if (rank == root_rank) {
    dimensions[0] = graph_on_root->vertex_count();
    dimensions[1] = static_cast<int>(graph_on_root->edges().size());
    packed = pack_edges(graph_on_root->edges());
  }
  MPI_Bcast(dimensions, 2, MPI_INT, root_rank, comm);

  packed.resize(static_cast<std::size_t>(dimensions[1]) * 3);
  MPI_Bcast(packed.data(), static_cast<int>(packed.size()), MPI_INT, root_rank,
            comm);

  if (rank == root_rank) {
    return *graph_on_root;
  }
  return mst::core::validate(
      mst::core::raw_graph{dimensions[0], unpack_edges(packed)});
}

/// Scansiona la fetta `[begin, end)` di archi di questo rank e aggiorna
/// `best_keys` con il minimo per componente, con la stessa `candidate_key`
/// a 64 bit usata dagli altri backend (confronto intero, pareggi risolti
/// per indice più basso). Il DSU è già completo e sincronizzato in locale,
/// quindi non serve comunicazione per trovare le radici.
void local_best_candidate_keys(
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu, int begin,
    int end, std::vector<std::uint64_t> &best_keys) {
  for (int index = begin; index < end; ++index) {
    const mst::core::edge &edge =
        graph.edges()[static_cast<std::size_t>(index)];
    const mst::core::component_id left_root = dsu.find(edge.u);
    const mst::core::component_id right_root = dsu.find(edge.v);
    if (left_root == right_root) {
      continue;
    }

    const mst::core::edge_index edge_index =
        mst::core::make_edge_index(static_cast<std::size_t>(index));
    const std::uint64_t key =
        mst::core::make_candidate_key(edge.weight, edge_index).value();
    auto &left_candidate = best_keys[left_root.index()];
    auto &right_candidate = best_keys[right_root.index()];
    left_candidate = std::min(left_candidate, key);
    right_candidate = std::min(right_candidate, key);
  }
}

/// Fase 1 (locale): calcola i minimi sulla fetta di archi di questo rank.
/// Il parametro `mpi_round<parents_broadcasted>` è solo un token di tipo:
/// segnala che il DSU è già sincronizzato fra i rank quando si entra in
/// questa fase (lo è per costruzione: stesso `MPI_Allreduce` e stesse
/// contrazioni in ordine identico, vedi `apply_contractions`), nello
/// stesso spirito dei concept in `contracts.hpp`.
const std::vector<std::uint64_t> &compute_local_minima(
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu,
    int edge_begin, int edge_end, mpi_workspace &workspace,
    mst::execution::mpi_round<mst::execution::parents_broadcasted>) {
  workspace.reset_local();
  local_best_candidate_keys(graph, dsu, edge_begin, edge_end,
                            workspace.local_keys);
  return workspace.local_keys;
}

/// Fase 2 (collettiva): un solo `MPI_Allreduce` con `MPI_MIN` su
/// `MPI_UINT64_T` riduce e ridistribuisce i minimi globali in un colpo
/// solo — il minimo bit a bit sulla `candidate_key` coincide col minimo
/// "logico" (peso, indice), niente da fare di custom.
const std::vector<std::uint64_t> &reduce_minima(
    const std::vector<std::uint64_t> &local_keys, mpi_workspace &workspace,
    const mst::core::validated_graph &graph, MPI_Comm comm,
    mst::execution::mpi_round<mst::execution::local_minima_computed>) {
  const int vertex_count = graph.vertex_count();
  MPI_Allreduce(local_keys.data(), workspace.reduced_keys.data(), vertex_count,
                MPI_UINT64_T, MPI_MIN, comm);

  return workspace.reduced_keys;
}

/// Fase 3: ogni rank applica le stesse contrazioni in locale, in modo
/// indipendente ma deterministico — partendo dallo stesso stato e dallo
/// stesso `best_keys`, tutti arrivano allo stesso risultato senza
/// scambiarsi nulla ("embarrassingly replicated": si ripete un lavoro O(V)
/// pur di non dover distribuire/raccogliere gli archi ammessi).
int apply_contractions(
    const std::vector<std::uint64_t> &best_keys,
    const mst::core::validated_graph &graph,
    mst::dsu::disjoint_set<mst::core::uncompressed_parents> &dsu,
    std::vector<mst::core::mst_edge> &mst_edges, int &total_weight) {
  int admitted_count = 0;
  for (const std::uint64_t key_value : best_keys) {
    if (key_value == mst::core::empty_candidate_key.value()) {
      continue;
    }
    const mst::core::candidate_key key{key_value};
    const mst::core::edge_index edge_index =
        mst::core::edge_index_from_candidate_key(key);
    const mst::core::candidate_edge candidate{
        graph.edges()[static_cast<std::size_t>(edge_index.value())],
        edge_index};
    if (auto admitted = dsu.unite(candidate)) {
      mst_edges.push_back(*admitted);
      total_weight += admitted->value.weight.value();
      ++admitted_count;
    }
  }
  return admitted_count;
}

} // namespace mst::backend::mpi

int main(std::int32_t argc, char **argv) {
  using namespace mst::backend::mpi;

  MPI_Init(&argc, &argv);

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const mst::app::config_parse_result parsed =
      mst::app::parse_app_config(argc, argv);
  if (!parsed.success || parsed.help_requested) {
    int config_exit_code = EXIT_FAILURE;
    if (rank == root_rank) {
      mst::app::handle_config_parse_result(parsed, argv[0],
                                           config_exit_code);
    } else {
      config_exit_code = parsed.help_requested ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    MPI_Finalize();
    return config_exit_code;
  }
  const mst::app::app_config &config = parsed.config;

  const double total_start = MPI_Wtime();
  // Solo il rank radice carica (o genera) il grafo: gli altri lo ricevono
  // via `broadcast_graph`, così la sorgente di verità è una sola e i dati
  // viaggiano sul canale MPI invece di essere ricostruiti N volte.
  std::optional<mst::app::loaded_graph> loaded_on_root;
  if (rank == root_rank) {
    loaded_on_root = mst::app::load_graph(config);
  }
  const mst::core::validated_graph graph = broadcast_graph(
      rank == root_rank ? &loaded_on_root->graph : nullptr, rank,
      MPI_COMM_WORLD);
  mst::dsu::disjoint_set<mst::core::uncompressed_parents> dsu(
      graph.vertex_count());
  std::vector<mst::core::mst_edge> mst_edges;
  int total_weight = 0;
  int rounds = 0;
  int remaining_components = graph.vertex_count();
  mpi_workspace workspace(graph.vertex_count());
  double local_compute_seconds = 0.0;
  double reduce_seconds = 0.0;
  double contract_seconds = 0.0;
  const double mst_start = MPI_Wtime();

  // Ogni rank ha l'intero grafo e un proprio DSU, ma scansiona solo la sua
  // fetta di archi; un solo MPI_Allreduce combina i minimi e tutti i rank
  // ne escono già allineati (vedi `apply_contractions`). Anche qui le
  // componenti dimezzano ad ogni round buono: O(log V) round.
  while (remaining_components > 1) {
    ++rounds;
    const int begin =
        edge_begin_for_rank(static_cast<int>(graph.edges().size()), rank, size);
    const int end =
        edge_end_for_rank(static_cast<int>(graph.edges().size()), rank, size);
    const double local_compute_start = MPI_Wtime();
    const std::vector<std::uint64_t> &local =
        compute_local_minima(graph, dsu, begin, end, workspace, {});
    local_compute_seconds += MPI_Wtime() - local_compute_start;

    const double reduce_start = MPI_Wtime();
    const std::vector<std::uint64_t> &reduced =
        reduce_minima(local, workspace, graph, MPI_COMM_WORLD, {});
    reduce_seconds += MPI_Wtime() - reduce_start;

    const double contract_start = MPI_Wtime();
    const int admitted_count =
        apply_contractions(reduced, graph, dsu, mst_edges, total_weight);
    contract_seconds += MPI_Wtime() - contract_start;
    remaining_components -= admitted_count;

    if (admitted_count == 0) {
      break;
    }
  }
  const double mst_end = MPI_Wtime();
  const double total_end = MPI_Wtime();

  double max_local_compute_seconds = 0.0;
  double avg_local_compute_seconds = 0.0;
  double max_reduce_seconds = 0.0;
  double max_contract_seconds = 0.0;
  MPI_Reduce(&local_compute_seconds, &max_local_compute_seconds, 1, MPI_DOUBLE,
             MPI_MAX, root_rank, MPI_COMM_WORLD);
  MPI_Reduce(&local_compute_seconds, &avg_local_compute_seconds, 1, MPI_DOUBLE,
             MPI_SUM, root_rank, MPI_COMM_WORLD);
  MPI_Reduce(&reduce_seconds, &max_reduce_seconds, 1, MPI_DOUBLE, MPI_MAX,
             root_rank, MPI_COMM_WORLD);
  MPI_Reduce(&contract_seconds, &max_contract_seconds, 1, MPI_DOUBLE, MPI_MAX,
             root_rank, MPI_COMM_WORLD);

  const double verification_start = MPI_Wtime();
  const mst::boruvka::verification_result verification =
      mst::boruvka::verify_against_sequential_cpu(graph, mst_edges,
                                                  total_weight);
  const double verification_seconds = MPI_Wtime() - verification_start;
  double max_verification_seconds = 0.0;
  MPI_Reduce(&verification_seconds, &max_verification_seconds, 1, MPI_DOUBLE,
             MPI_MAX, root_rank, MPI_COMM_WORLD);
  const int local_verification_success = verification.success ? 1 : 0;
  int all_verification_success = 0;
  MPI_Allreduce(&local_verification_success, &all_verification_success, 1,
                MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  int version = 0;
  int subversion = 0;
  MPI_Get_version(&version, &subversion);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int processor_name_length = 0;
  MPI_Get_processor_name(processor_name, &processor_name_length);

  if (rank == root_rank) {
    mst::app::print_result("MPI Boruvka MST", config, graph, mst_edges,
                           total_weight);
    if (all_verification_success != 0) {
      std::cout << "Sequential CPU verification: passed\n";
    } else {
      std::cerr << "Sequential CPU verification failed: expected weight "
                << verification.expected_total_weight << " with "
                << verification.expected_edge_count << " edges, got weight "
                << verification.actual_total_weight << " with "
                << verification.actual_edge_count << " edges\n";
    }

    avg_local_compute_seconds /= static_cast<double>(size);

    std::ostringstream report;
    report << "{\n";
    report << mst::reporting::common_metadata_json(
                  "mpi", all_verification_success != 0)
           << ",\n";
    report << mst::app::graph_metadata_json(loaded_on_root->selected) << ",\n";
    report << mst::app::configuration_metadata_json(config) << ",\n";
    std::ostringstream backend_timing_fields;
    backend_timing_fields << "    \"backend\": {\n";
    backend_timing_fields << "      \"max_scan_seconds\": "
                          << max_local_compute_seconds << ",\n";
    backend_timing_fields << "      \"avg_scan_seconds\": "
                          << avg_local_compute_seconds << "\n";
    backend_timing_fields << "    }\n";
    mst::reporting::write_phase_timings_json(
        report, mst::reporting::phase_timing_profile{
                    total_end - total_start,
                    mst_end - mst_start,
                    max_verification_seconds,
                    max_local_compute_seconds,
                    max_reduce_seconds,
                    max_contract_seconds,
                    0.0,
                },
        backend_timing_fields.str());
    report << ",\n";
    mst::reporting::write_telemetry_details_json(
        report, mst::reporting::telemetry_details_profile{
                    max_local_compute_seconds,
                    max_reduce_seconds,
                    max_contract_seconds,
                    0.0,
                    0,
                    0.0,
                    0,
                    0.0,
                    0.0,
                });
    report << ",\n";
    report << "  \"capabilities\": {\n";
    report << "    \"world_size\": " << size << ",\n";
    report << "    \"mpi_version_major\": " << version << ",\n";
    report << "    \"mpi_version_minor\": " << subversion << ",\n";
    report << "    \"processor_name\": \""
           << mst::reporting::json_escape(
                  std::string(processor_name,
                              static_cast<std::size_t>(processor_name_length)))
           << "\"\n";
    report << "  },\n";
    report << mst::app::mst_metadata_json(graph, mst_edges.size(), rounds,
                                          total_weight)
           << ",\n";
    mst::boruvka::write_verification_json(report, verification);
    report << "\n";
    report << "}\n";
    mst::app::write_report_if_requested(config, report.str());
  }

  MPI_Finalize();
  return all_verification_success != 0 ? 0 : 1;
}
