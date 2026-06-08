#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>

#include <unistd.h>

namespace mst::reporting {

/// Tempi di fase comuni a tutti i backend, per il report JSON.
/// `algorithm_seconds` è opzionale (-1.0 = non misurato): in quel caso
/// `write_phase_timings_json` ripiega su `mst_loop_seconds`.
struct phase_timing_profile {
  double total_seconds = 0.0;
  double mst_loop_seconds = 0.0;
  double sequential_cpu_verification_seconds = 0.0;
  double scan_seconds = 0.0;
  double reduce_seconds = 0.0;
  double contract_seconds = 0.0;
  double compress_seconds = 0.0;
  double algorithm_seconds = -1.0;
};

/// Telemetria opzionale specifica del backend (retry sul DSU, collisioni `atomicMin`, overhead kernel...) per analisi più fini dei tempi di fase.
struct telemetry_details_profile {
  double scan_time_pure_seconds = 0.0;
  double reduce_time_pure_seconds = 0.0;
  double contract_time_pure_seconds = 0.0;
  double allocation_and_overhead_seconds = 0.0;
  std::uint64_t dsu_contention_retries = 0;
  double kernel_launch_overhead_estimated_seconds = 0.0;
  std::uint64_t cuda_atomic_min_collision_count = 0;
  double cuda_atomic_min_collision_rate = 0.0;
  double unattributed_residual_seconds = 0.0;
};

/// Escaping minimo per infilare una stringa in un valore JSON fra virgolette (backslash, virgolette, newline/CR/tab).
inline std::string json_escape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());

  for (const char ch : value) {
    switch (ch) {
    case '\\':
      escaped += "\\\\";
      break;
    case '"':
      escaped += "\\\"";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      escaped.push_back(ch);
      break;
    }
  }

  return escaped;
}

/// Scrive il report su disco (creando le directory mancanti); `false` su qualsiasi errore.
inline bool write_report(const std::filesystem::path &path,
                         std::string_view report) {
  std::error_code error;
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
      return false;
    }
  }

  std::ofstream stream(path);
  if (!stream) {
    return false;
  }

  stream << report;
  return static_cast<bool>(stream);
}

/// Variabile d'ambiente o stringa vuota se non definita (es. `SLURM_JOB_ID` fuori da SLURM) — niente puntatori nulli nel report.
inline std::string env_or_empty(const char *name) {
  if (const char *value = std::getenv(name)) {
    return value;
  }
  return {};
}

/// Nome host della macchina (per distinguere i nodi nei benchmark distribuiti); `"unknown"` se la syscall fallisce.
inline std::string hostname() {
  std::array<char, 256> buffer{};
  if (gethostname(buffer.data(), buffer.size()) == 0) {
    buffer.back() = '\0';
    return std::string(buffer.data());
  }
  return "unknown";
}

/// Timestamp ISO 8601 / UTC, per datare i report in modo confrontabile fra esecuzioni e nodi.
inline std::string utc_timestamp() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t time = std::chrono::system_clock::to_time_t(now);
  std::tm parts{};
#if defined(_WIN32)
  gmtime_s(&parts, &time);
#else
  gmtime_r(&time, &parts);
#endif

  std::ostringstream out;
  out << std::put_time(&parts, "%Y-%m-%dT%H:%M:%SZ");
  return out.str();
}

/// Frammento JSON coi metadati comuni a tutti i report: backend, esito, timestamp, hostname, ID job SLURM.
inline std::string common_metadata_json(std::string_view backend,
                                        bool success) {
  std::ostringstream out;
  out << "  \"backend\": \"" << json_escape(backend) << "\",\n";
  out << "  \"success\": " << (success ? "true" : "false") << ",\n";
  out << "  \"timestamp\": \"" << json_escape(utc_timestamp()) << "\",\n";
  out << "  \"hostname\": \"" << json_escape(hostname()) << "\",\n";
  out << "  \"slurm_job_id\": \"" << json_escape(env_or_empty("SLURM_JOB_ID"))
      << "\"";
  return out.str();
}

/// Blocco JSON `"timings"`: `algorithm_seconds` ripiega su `mst_loop_seconds`
/// se non misurato a parte; `backend_fields` accoda campi specifici del backend allo stesso oggetto.
inline void write_phase_timings_json(std::ostream &out,
                                     const phase_timing_profile &timings,
                                     std::string_view backend_fields = {}) {
  const double algorithm_seconds = timings.algorithm_seconds >= 0.0
                                       ? timings.algorithm_seconds
                                       : timings.mst_loop_seconds;
  out << "  \"timings\": {\n";
  out << "    \"total_seconds\": " << timings.total_seconds << ",\n";
  out << "    \"mst_loop_seconds\": " << timings.mst_loop_seconds << ",\n";
  out << "    \"algorithm_seconds\": " << algorithm_seconds << ",\n";
  out << "    \"sequential_cpu_verification_seconds\": "
      << timings.sequential_cpu_verification_seconds << ",\n";
  out << "    \"scan_seconds\": " << timings.scan_seconds << ",\n";
  out << "    \"reduce_seconds\": " << timings.reduce_seconds << ",\n";
  out << "    \"contract_seconds\": " << timings.contract_seconds << ",\n";
  out << "    \"compress_seconds\": " << timings.compress_seconds;
  if (!backend_fields.empty()) {
    out << ",\n";
    out << backend_fields;
  } else {
    out << "\n";
  }
  out << "  }";
}

/// I report vogliono la telemetria in millisecondi, internamente è in secondi.
inline double seconds_to_milliseconds(double seconds) {
  return seconds * 1000.0;
}

/// Blocco JSON opzionale `"telemetry_details"`, tempi convertiti in millisecondi per leggibilità.
inline void
write_telemetry_details_json(std::ostream &out,
                             const telemetry_details_profile &telemetry) {
  out << "  \"telemetry_details\": {\n";
  out << "    \"scan_time_pure_ms\": "
      << seconds_to_milliseconds(telemetry.scan_time_pure_seconds) << ",\n";
  out << "    \"reduce_time_pure_ms\": "
      << seconds_to_milliseconds(telemetry.reduce_time_pure_seconds) << ",\n";
  out << "    \"contract_time_pure_ms\": "
      << seconds_to_milliseconds(telemetry.contract_time_pure_seconds)
      << ",\n";
  out << "    \"allocation_and_overhead_ms\": "
      << seconds_to_milliseconds(telemetry.allocation_and_overhead_seconds)
      << ",\n";
  out << "    \"dsu_contention_retries\": "
      << telemetry.dsu_contention_retries << ",\n";
  out << "    \"kernel_launch_overhead_estimated_ms\": "
      << seconds_to_milliseconds(
             telemetry.kernel_launch_overhead_estimated_seconds)
      << ",\n";
  out << "    \"cuda_atomic_min_collision_count\": "
      << telemetry.cuda_atomic_min_collision_count << ",\n";
  out << "    \"cuda_atomic_min_collision_rate\": "
      << telemetry.cuda_atomic_min_collision_rate << ",\n";
  out << "    \"unattributed_residual_ms\": "
      << seconds_to_milliseconds(telemetry.unattributed_residual_seconds)
      << "\n";
  out << "  }";
}

} // namespace mst::reporting
