#pragma once

#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <string_view>

#include <unistd.h>

namespace mst::reporting {

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

inline std::string env_or_empty(const char *name) {
  if (const char *value = std::getenv(name)) {
    return value;
  }
  return {};
}

inline std::string hostname() {
  std::array<char, 256> buffer{};
  if (gethostname(buffer.data(), buffer.size()) == 0) {
    buffer.back() = '\0';
    return std::string(buffer.data());
  }
  return "unknown";
}

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

inline std::string common_metadata_json(std::string_view backend,
                                        bool success) {
  std::ostringstream out;
  out << "  \"backend\": \"" << json_escape(backend) << "\",\n";
  out << "  \"success\": " << (success ? "true" : "false") << ",\n";
  out << "  \"timestamp\": \"" << json_escape(utc_timestamp()) << "\",\n";
  out << "  \"hostname\": \"" << json_escape(hostname()) << "\",\n";
  out << "  \"slurm_job_id\": \""
      << json_escape(env_or_empty("SLURM_JOB_ID")) << "\"";
  return out.str();
}

inline std::string report_path_or_empty() {
  return env_or_empty("MST_REPORT_PATH");
}

inline bool write_report_from_env(std::string_view report) {
  const std::string path = report_path_or_empty();
  if (path.empty()) {
    return false;
  }
  return write_report(path, report);
}

} // namespace mst::reporting
