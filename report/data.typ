// Data loading and normalization for benchmark reports.
//
// Compile from the repository root with:
//   typst compile --root . report/main.typ report/main.pdf
//
// The leading slash in each path is project-root-relative, not filesystem-root.

#let report-files = (
  "/results/cuda_dense16_hmcompile_default_27786.json",
  "/results/cuda_random_v32768_e196608_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_square_hmcompile_default_27786.json",
  "/results/cuda_test_hmcompile_default_27786.json",
  "/results/cuda_tie_hmcompile_default_27786.json",
  "/results/cuda_triangle_hmcompile_default_27786.json",
  "/results/mpi_dense16_np2_27785.json",
  "/results/mpi_random_v32768_e196608_s886261_w10000_np2_27788.json",
  "/results/mpi_square_np2_27785.json",
  "/results/mpi_test_np2_27785.json",
  "/results/mpi_tie_np2_27785.json",
  "/results/mpi_triangle_np2_27785.json",
  "/results/openmp_dense16_t4_27784.json",
  "/results/openmp_random_v32768_e196608_s886261_w10000_t4_27787.json",
  "/results/openmp_square_t4_27784.json",
  "/results/openmp_test_t4_27784.json",
  "/results/openmp_tie_t4_27784.json",
  "/results/openmp_triangle_t4_27784.json",
)

#let random-sweep-files = (
  "/results/mpi_random_v32768_e32768_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e65536_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e131072_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e196608_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e393216_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e786432_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e1572864_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e3145728_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e6291456_s886261_w10000_np2_27788.json",
  "/results/mpi_random_v32768_e12582912_s886261_w10000_np2_27788.json",
  "/results/openmp_random_v32768_e32768_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e65536_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e131072_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e196608_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e393216_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e786432_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e1572864_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e3145728_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e6291456_s886261_w10000_t4_27787.json",
  "/results/openmp_random_v32768_e12582912_s886261_w10000_t4_27787.json",
  "/results/cuda_random_v32768_e32768_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e65536_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e131072_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e196608_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e393216_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e786432_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e1572864_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e3145728_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e6291456_s886261_w10000_hmcompile_default_27789.json",
  "/results/cuda_random_v32768_e12582912_s886261_w10000_hmcompile_default_27789.json",
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
    algorithm: raw.timings.at("algorithm_seconds", default: raw.timings.mst_loop_seconds),
    rounds: raw.mst.rounds,
    mst_edges: raw.mst.selected_edge_count,
    weight: raw.mst.total_weight,
    success: raw.success,
    verified: raw.verification.sequential_cpu_success,
    raw: raw,
  )
}

#let reports = report-files.map(load-report)
#let random-sweep-runs = random-sweep-files.map(load-report)
#let random-sweep-projection-densities = (2, 3, 5, 7, 13, 25, 49, 97, 193, 385)

#let by-backend(name) = reports.filter(run => run.backend == name)
#let by-graph(name) = reports.filter(run => run.graph == name)
#let run(backend, graph) = reports.filter(item => item.backend == backend and item.graph == graph).at(0)
#let random-sweep-by-backend(name) = random-sweep-runs.filter(run => run.backend == name)

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
  mpi: oklch(66%, 0.09, 238deg),
  openmp: oklch(69%, 0.10, 72deg),
  cuda: oklch(66%, 0.10, 150deg),
).at(name)

#let workers(item) = {
  if item.backend == "mpi" {
    str(item.raw.capabilities.world_size) + " processi"
  } else if item.backend == "openmp" {
    str(item.raw.capabilities.max_threads) + " thread"
  } else {
    (
      str(item.raw.capabilities.multiprocessor_count)
        + " SM, "
        + str(item.raw.capabilities.max_threads_per_block)
        + " thread/blocco"
    )
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

#let timing-value(item, key, fallback: 0.0) = {
  let backend-timings = item.raw.timings.at("backend", default: (:))
  item.raw.timings.at(key, default: backend-timings.at(key, default: fallback))
}
#let sequential-cpu-seconds(item) = timing-value(item, "sequential_cpu_verification_seconds")
#let algorithm-seconds(item) = item.algorithm
#let backend-loop-speedup(item) = sequential-cpu-seconds(item) / item.loop
#let empirical-speedup(item) = sequential-cpu-seconds(item) / algorithm-seconds(item)

#let timing-breakdown(item) = {
  let phases = if item.backend == "mpi" {
    (
      (
        phase: "scan",
        seconds: timing-value(item, "max_local_compute_seconds", fallback: timing-value(item, "scan_seconds")),
      ),
      (
        phase: "reduce",
        seconds: timing-value(item, "max_reduce_seconds", fallback: timing-value(item, "reduce_seconds")),
      ),
      (
        phase: "contract",
        seconds: timing-value(item, "max_contract_seconds", fallback: timing-value(item, "contract_seconds")),
      ),
    )
  } else if item.backend == "openmp" {
    (
      (phase: "scan", seconds: item.raw.timings.scan_seconds),
      (phase: "reduce", seconds: item.raw.timings.reduce_seconds),
      (phase: "contract", seconds: item.raw.timings.contract_seconds),
      (phase: "compress", seconds: item.raw.timings.compress_seconds),
      (phase: "buffer/overhead", seconds: timing-value(item, "allocation_and_overhead_seconds")),
    )
  } else {
    let round-prepare = timing-value(
      item,
      "round_prepare_seconds",
      fallback: timing-value(item, "round_reset_seconds") + timing-value(item, "initialize_best_seconds"),
    )
    (
      (phase: "setup", seconds: timing-value(item, "setup_seconds")),
      (phase: "host-device", seconds: timing-value(item, "host_to_device_seconds")),
      (phase: "init parent", seconds: timing-value(item, "initialization_seconds")),
      (phase: "round prep", seconds: round-prepare),
      (phase: "scan", seconds: timing-value(item, "scan_seconds")),
      (
        phase: "contract",
        seconds: timing-value(item, "contract_kernel_seconds", fallback: timing-value(item, "contract_seconds")),
      ),
      (phase: "copy flags", seconds: timing-value(item, "contract_copy_seconds")),
      (phase: "compress", seconds: timing-value(item, "compress_seconds")),
      (phase: "device-host", seconds: timing-value(item, "device_to_host_seconds")),
      (phase: "launch/residuo", seconds: timing-value(item, "kernel_launch_overhead_estimated_seconds")),
    )
  }
  phases.filter(phase => phase.seconds > 0)
}

#let profiled-seconds(item) = timing-breakdown(item).map(phase => phase.seconds).sum()
#let unprofiled-mst-seconds(item) = item.loop - profiled-seconds(item)
#let setup-before-loop-seconds(item) = item.total - item.loop

#let log2-worker(item) = {
  let p = worker-count(item)
  if p <= 1 {
    0
  } else if p <= 2 {
    1
  } else if p <= 4 {
    2
  } else if p <= 8 {
    3
  } else if p <= 16 {
    4
  } else if p <= 32 {
    5
  } else if p <= 64 {
    6
  } else if p <= 128 {
    7
  } else {
    8
  }
}

#let cuda-sm-parallelism(item) = item.raw.capabilities.multiprocessor_count

#let random-sweep-reference-seconds(item) = {
  let matches = random-sweep-runs.filter(run => run.edges == item.edges)
  matches.map(sequential-cpu-seconds).sum() / matches.len()
}

#let random-sweep-speedup(item) = random-sweep-reference-seconds(item) / algorithm-seconds(item)
#let random-sweep-backend-loop-speedup(item) = random-sweep-reference-seconds(item) / item.loop
#let random-sweep-efficiency(item) = random-sweep-speedup(item) / worker-count(item)

#let random-sweep-theoretical-speedup(item) = {
  let n = item.vertices
  let m = item.edges
  let p = worker-count(item)
  if item.backend == "mpi" {
    m / (m / p + n * log2-worker(item))
  } else if item.backend == "openmp" {
    (m + n) / (m / p + n)
  } else {
    let q = cuda-sm-parallelism(item)
    q
  }
}

#let random-sweep-theoretical-speedup-at-density(backend, density) = {
  let reference = random-sweep-by-backend(backend).at(0)
  let n = reference.vertices
  let m = density * n
  let p = worker-count(reference)
  if backend == "mpi" {
    m / (m / p + n * log2-worker(reference))
  } else if backend == "openmp" {
    (m + n) / (m / p + n)
  } else {
    let q = cuda-sm-parallelism(reference)
    q
  }
}

#let random-sweep-half-efficiency-speedup(item) = worker-count(item) / 2

#let random-sweep-first(backend, predicate) = {
  let matches = random-sweep-by-backend(backend).filter(predicate)
  if matches.len() == 0 {
    none
  } else {
    matches.at(0)
  }
}

#let random-sweep-first-speedup-crossover(backend) = random-sweep-first(
  backend,
  item => random-sweep-speedup(item) >= 1,
)

#let random-sweep-first-half-efficiency(backend) = {
  random-sweep-first(
    backend,
    item => random-sweep-speedup(item) >= random-sweep-half-efficiency-speedup(item),
  )
}

#let isoefficiency-density-threshold(backend) = {
  let reference = random-sweep-by-backend(backend).at(0)
  if backend == "mpi" {
    worker-count(reference) * log2-worker(reference)
  } else if backend == "openmp" {
    worker-count(reference)
  } else {
    none
  }
}

#let isoefficiency-threshold-label(backend) = {
  let threshold = isoefficiency-density-threshold(backend)
  if threshold == none {
    [nessuna soglia asintotica]
  } else {
    [$frac(|E|, |V|) >= #threshold$]
  }
}

#let linear-fit(items, x-value, y-value) = {
  let count = items.len()
  if count == 0 {
    return (intercept: 0.0, slope: 0.0)
  }
  let sx = items.map(x-value).sum()
  let sy = items.map(y-value).sum()
  let sxx = items.map(item => x-value(item) * x-value(item)).sum()
  let sxy = items.map(item => x-value(item) * y-value(item)).sum()
  let denominator = count * sxx - sx * sx
  if denominator == 0 {
    (intercept: sy / count, slope: 0.0)
  } else {
    let slope = (count * sxy - sx * sy) / denominator
    let intercept = (sy - slope * sx) / count
    (intercept: intercept, slope: slope)
  }
}

#let fit-seconds(fit, density) = fit.intercept + fit.slope * density

#let random-sweep-cpu-fit = linear-fit(
  random-sweep-by-backend("mpi"),
  edge-density,
  random-sweep-reference-seconds,
)

#let random-sweep-backend-fit(backend) = linear-fit(
  random-sweep-by-backend(backend),
  edge-density,
  algorithm-seconds,
)

#let random-sweep-cpu-fit-seconds(item) = fit-seconds(random-sweep-cpu-fit, edge-density(item))

#let random-sweep-backend-fit-seconds(item) = {
  fit-seconds(random-sweep-backend-fit(item.backend), edge-density(item))
}

#let random-sweep-fit-crossover-density(backend) = {
  let backend-fit = random-sweep-backend-fit(backend)
  let denominator = backend-fit.slope - random-sweep-cpu-fit.slope
  if denominator == 0 {
    none
  } else {
    (random-sweep-cpu-fit.intercept - backend-fit.intercept) / denominator
  }
}

#let random-sweep-crossover-density(backend) = {
  let fitted = random-sweep-fit-crossover-density(backend)
  let first-measured = random-sweep-first-speedup-crossover(backend)
  if fitted != none and fitted > 0 {
    fitted
  } else if first-measured != none {
    edge-density(first-measured)
  } else {
    none
  }
}

#let random-sweep-crossover-label(backend) = {
  let fitted = random-sweep-fit-crossover-density(backend)
  if fitted != none and fitted > 0 {
    [soglia fitted]
  } else {
    [primo $S >= 1$]
  }
}
