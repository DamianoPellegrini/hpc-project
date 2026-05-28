// Data loading and normalization for benchmark reports.
//
// Compile from the repository root with:
//   typst compile --root . report/main.typ report/main.pdf
//
// The leading slash in each path is project-root-relative, not filesystem-root.

#let report-files = (
  "/results/cuda_dense16_27058.json",
  "/results/cuda_random_27058.json",
  "/results/cuda_square_27058.json",
  "/results/cuda_test_27058.json",
  "/results/cuda_tie_27058.json",
  "/results/cuda_triangle_27058.json",
  "/results/mpi_dense16_27057.json",
  "/results/mpi_random_27057.json",
  "/results/mpi_square_27057.json",
  "/results/mpi_test_27057.json",
  "/results/mpi_tie_27057.json",
  "/results/mpi_triangle_27057.json",
  "/results/openmp_dense16_27056.json",
  "/results/openmp_random_27056.json",
  "/results/openmp_square_27056.json",
  "/results/openmp_test_27056.json",
  "/results/openmp_tie_27056.json",
  "/results/openmp_triangle_27056.json",
)

#let cuda-sweep-files = (
  "/results/cuda_random_v32768_e32768_27059.json",
  "/results/cuda_random_v32768_e65536_27059.json",
  "/results/cuda_random_v32768_e131072_27059.json",
  "/results/cuda_random_v32768_e196608_27059.json",
  "/results/cuda_random_v32768_e393216_27059.json",
  "/results/cuda_random_v32768_e786432_27059.json",
  "/results/cuda_random_v32768_e1572864_27059.json",
  "/results/cuda_random_v32768_e2064385_27061.json",
  "/results/cuda_random_v32768_e2588673_27061.json",
  "/results/cuda_random_v32768_e2850817_27061.json",
  "/results/cuda_random_v32768_e3112961_27061.json",
  "/results/cuda_random_v32768_e3637249_27061.json",
  "/results/cuda_random_v32768_e4161537_27061.json",
)

#let backends = ("mpi", "openmp", "cuda")
#let graphs = ("triangle", "square", "tie", "test", "dense16", "random")

#let load-report(path) = {
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
}

#let reports = report-files.map(load-report)
#let cuda-sweep-runs = cuda-sweep-files.map(load-report)
#let cuda-sweep-max = cuda-sweep-runs.at(cuda-sweep-runs.len() - 1)

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

#let percent(part, total) = str(calc.round(100 * part / total, digits: 1)) + "%"

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
  mpi: oklch(56%, 0.09, 238deg),
  openmp: oklch(59%, 0.10, 72deg),
  cuda: oklch(56%, 0.10, 150deg),
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

#let edge-density(item) = item.edges / item.vertices

#let platform(item) = {
  if item.backend == "cuda" {
    item.raw.capabilities.device_name
  } else if item.backend == "mpi" {
    item.raw.capabilities.processor_name
  } else {
    item.raw.hostname
  }
}

#let timing-value(item, key, fallback: 0.0) = item.raw.timings.at(key, default: fallback)
#let sequential-cpu-seconds(item) = timing-value(item, "sequential_cpu_verification_seconds")
#let empirical-speedup(item) = sequential-cpu-seconds(item) / item.loop
#let cuda-sweep-crossover-runs = cuda-sweep-runs.filter(item => empirical-speedup(item) >= 1)
#let cuda-sweep-first-crossover = {
  if cuda-sweep-crossover-runs.len() == 0 {
    none
  } else {
    cuda-sweep-crossover-runs.at(0)
  }
}

#let timing-breakdown(item) = {
  let phases = if item.backend == "mpi" {
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
      (phase: "setup", seconds: timing-value(item, "setup_seconds")),
      (phase: "host-device", seconds: timing-value(item, "host_to_device_seconds")),
      (phase: "init parent", seconds: timing-value(item, "initialization_seconds")),
      (phase: "reset", seconds: timing-value(item, "round_reset_seconds")),
      (phase: "init best", seconds: timing-value(item, "initialize_best_seconds")),
      (phase: "scan", seconds: timing-value(item, "scan_seconds")),
      (phase: "contract", seconds: timing-value(item, "contract_kernel_seconds", fallback: timing-value(item, "contract_seconds"))),
      (phase: "copy flags", seconds: timing-value(item, "contract_copy_seconds")),
      (phase: "compress", seconds: timing-value(item, "compress_seconds")),
      (phase: "device-host", seconds: timing-value(item, "device_to_host_seconds")),
    )
  }
  phases.filter(phase => phase.seconds > 0)
}

#let profiled-seconds(item) = timing-breakdown(item).map(phase => phase.seconds).sum()
#let unprofiled-mst-seconds(item) = item.loop - profiled-seconds(item)
#let setup-before-loop-seconds(item) = item.total - item.loop

#let linear-fit(items, value) = {
  let count = items.len()
  let sx = items.map(item => item.edges).sum()
  let sy = items.map(value).sum()
  let sxx = items.map(item => item.edges * item.edges).sum()
  let sxy = items.map(item => item.edges * value(item)).sum()
  let denominator = count * sxx - sx * sx
  let slope = (count * sxy - sx * sy) / denominator
  let intercept = (sy - slope * sx) / count
  (intercept: intercept, slope: slope)
}

#let cuda-loop-fit = linear-fit(cuda-sweep-runs, item => item.loop)
#let cuda-sequential-fit = linear-fit(cuda-sweep-runs, sequential-cpu-seconds)

#let cuda-threshold-edges = {
  if cuda-sequential-fit.slope > cuda-loop-fit.slope {
    let numerator = cuda-loop-fit.intercept - cuda-sequential-fit.intercept
    let denominator = cuda-sequential-fit.slope - cuda-loop-fit.slope
    numerator / denominator
  } else {
    none
  }
}

#let cuda-threshold-density = {
  if cuda-threshold-edges == none {
    none
  } else {
    cuda-threshold-edges / cuda-sweep-runs.at(0).vertices
  }
}

#let seconds-per-million-edges(slope) = slope * 1000000
