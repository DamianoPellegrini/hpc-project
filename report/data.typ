// Data loading and normalization for benchmark reports.
//
// Compile from the repository root with:
//   typst compile --root . report/main.typ report/main.pdf
//
// The leading slash in each path is project-root-relative, not filesystem-root.

#let report-files = (
  "/results/cuda_dense16_27015.json",
  "/results/cuda_random_27015.json",
  "/results/cuda_square_27015.json",
  "/results/cuda_test_27015.json",
  "/results/cuda_tie_27015.json",
  "/results/cuda_triangle_27015.json",
  "/results/mpi_dense16_27014.json",
  "/results/mpi_random_27014.json",
  "/results/mpi_square_27014.json",
  "/results/mpi_test_27014.json",
  "/results/mpi_tie_27014.json",
  "/results/mpi_triangle_27014.json",
  "/results/openmp_dense16_27013.json",
  "/results/openmp_random_27013.json",
  "/results/openmp_square_27013.json",
  "/results/openmp_test_27013.json",
  "/results/openmp_tie_27013.json",
  "/results/openmp_triangle_27013.json",
)

#let backends = ("mpi", "openmp", "cuda")
#let graphs = ("triangle", "square", "tie", "test", "dense16", "random")

#let reports = report-files.map(path => {
  let raw = json(path)
  (
    path: path,
    backend: raw.backend,
    graph: raw.graph.name,
    vertices: raw.graph.vertex_count,
    edges: raw.graph.edge_count,
    total: raw.timings.total_seconds,
    loop: raw.timings.mst_loop_seconds,
    rounds: raw.mst.rounds,
    mst_edges: raw.mst.selected_edge_count,
    weight: raw.mst.total_weight,
    success: raw.success,
    verified: raw.verification.sequential_cpu_success,
    raw: raw,
  )
})

#let by-backend(name) = reports.filter(run => run.backend == name)
#let by-graph(name) = reports.filter(run => run.graph == name)
#let run(backend, graph) = reports.filter(item => item.backend == backend and item.graph == graph).at(0)

#let random-runs = backends.map(backend => run(backend, "random"))

#let duration(seconds) = {
  if seconds < 0.001 {
    str(calc.round(seconds * 1000000, digits: 2)) + " us"
  } else if seconds < 1 {
    str(calc.round(seconds * 1000, digits: 3)) + " ms"
  } else {
    str(calc.round(seconds, digits: 3)) + " s"
  }
}

#let ratio(value) = str(calc.round(value, digits: 2)) + "x"

#let graph-label(name) = (
  triangle: "triangle",
  square: "square",
  tie: "tie",
  test: "test",
  dense16: "dense16",
  random: "random",
).at(name)

#let backend-label(name) = (
  mpi: "MPI",
  openmp: "OpenMP",
  cuda: "CUDA",
).at(name)

#let backend-color(name) = (
  mpi: rgb("#2f6f9f"),
  openmp: rgb("#8a5a16"),
  cuda: rgb("#2f7d4a"),
).at(name)

#let workers(item) = {
  if item.backend == "mpi" {
    str(item.raw.capabilities.world_size) + " processi"
  } else if item.backend == "openmp" {
    str(item.raw.capabilities.max_threads) + " thread"
  } else {
    str(item.raw.capabilities.multiprocessor_count) + " SM, " + str(item.raw.capabilities.max_threads_per_block) + " thread/blocco"
  }
}

#let worker-count(item) = {
  if item.backend == "mpi" {
    item.raw.capabilities.world_size
  } else if item.backend == "openmp" {
    item.raw.capabilities.max_threads
  } else {
    item.raw.capabilities.multiprocessor_count
  }
}

#let cpu-cost-threshold(item) = 2 * worker-count(item) * item.vertices
#let cpu-cost-ratio(item) = item.edges / cpu-cost-threshold(item)
#let per-worker-edge-count(item) = item.edges / worker-count(item)
#let per-worker-candidate-threshold(item) = 2 * item.vertices

#let platform(item) = {
  if item.backend == "cuda" {
    item.raw.capabilities.device_name
  } else if item.backend == "mpi" {
    item.raw.capabilities.processor_name
  } else {
    item.raw.hostname
  }
}

#let timing-breakdown(item) = {
  if item.backend == "mpi" {
    (
      (phase: "compute locale", seconds: item.raw.timings.max_local_compute_seconds),
      (phase: "reduce", seconds: item.raw.timings.max_reduce_seconds),
      (phase: "contract", seconds: item.raw.timings.max_contract_seconds),
    )
  } else if item.backend == "openmp" {
    (
      (phase: "scan", seconds: item.raw.timings.scan_seconds),
      (phase: "reduce", seconds: item.raw.timings.reduce_seconds),
      (phase: "contract", seconds: item.raw.timings.contract_seconds),
      (phase: "compress", seconds: item.raw.timings.compress_seconds),
    )
  } else {
    (
      (phase: "host-device", seconds: item.raw.timings.host_to_device_seconds),
      (phase: "scan", seconds: item.raw.timings.scan_seconds),
      (phase: "contract", seconds: item.raw.timings.contract_seconds),
      (phase: "compress", seconds: item.raw.timings.compress_seconds),
      (phase: "device-host", seconds: item.raw.timings.device_to_host_seconds),
    )
  }
}

#let profiled-seconds(item) = timing-breakdown(item).map(phase => phase.seconds).sum()
#let unprofiled-mst-seconds(item) = item.loop - profiled-seconds(item)
#let setup-before-loop-seconds(item) = item.total - item.loop
