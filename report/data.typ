// Data loading and normalization for benchmark reports.
//
// Compile from the repository root with:
//   typst compile --root . report/main.typ report/main.pdf
//
// The leading slash in each path is project-root-relative, not filesystem-root.
// Ogni file è il CSV prodotto da uno script Slurm (vedi scripts/slurm/),
// schema: backend,vertices,edges,density,seed,resources,overhead_seconds,
// exec_seconds,total_seconds,verified.

#let result-files = (
  "/results/mpi_28093.csv",
  "/results/openmp_28092.csv",
  "/results/cuda_28094.csv",
  "/results/sequential_28140.csv",
)

#let load-csv(path) = csv(path, row-type: dictionary).map(row => (
  backend: row.backend,
  vertices: int(row.vertices),
  edges: int(row.edges),
  density: float(row.density),
  seed: int(row.seed),
  resources: int(row.resources),
  overhead: float(row.overhead_seconds),
  exec: float(row.exec_seconds),
  total: float(row.total_seconds),
  verified: row.verified,
))

#let runs = result-files.map(load-csv).flatten()

#let backends = ("mpi", "openmp", "cuda")

#let by-backend(name) = runs.filter(item => item.backend == name)

#let run-at-density(backend, density) = by-backend(backend).filter(item => item.density == density).at(0)

// Densità di riferimento usata per tabelle/grafici a singolo punto: stesso
// valore (|E|=196608, |V|=32768) della run "random" della versione precedente.
#let reference-density = 6.0

#let reference-runs = backends.map(backend => run-at-density(backend, reference-density))

// Densità della sweep (uguali per i tre backend, vedi RANDOM_EDGES_LIST in
// scripts/slurm/*.sh): 1, 2, 4, 6, 12, 24, 48, 96, 192, 384.
#let swept-densities = by-backend("mpi").map(item => item.density)

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

#let backend-label(name) = (
  mpi: "MPI",
  openmp: "OpenMP",
  cuda: "CUDA",
).at(name)

#let backend-color(name) = (
  mpi: oklch(60%, 0.16, 255deg),
  openmp: oklch(68%, 0.16, 55deg),
  cuda: oklch(62%, 0.16, 145deg),
).at(name)

// Grado di parallelismo: processi MPI, thread OpenMP, GPU CUDA (vedi colonna
// `resources` dei CSV e scripts/slurm/*.sh).
#let worker-count(item) = item.resources

#let workers(item) = (
  if item.backend == "mpi" {
    str(item.resources) + " processi"
  } else if item.backend == "openmp" {
    str(item.resources) + " thread"
  } else {
    str(item.resources) + " GPU"
  }
)

#let log2(x) = calc.log(x, base: 2)

// Numero di SM della GPU usata per le run CUDA (NVIDIA L40S).
#let cuda-sm-count = 142

// Per CUDA, E_p è definita rispetto al proprio grado di parallelismo q (numero
// di SM, @eq:cuda-sm-speedup): E_p^CUDA = S_p^CUDA / q = (d+1)/(d+log2|V|),
// dove d = |E|/|V|. Questa forma normalizzata non dipende da q.
#let cuda-normalized-efficiency(item) = {
  let d = item.density
  (d + 1) / (d + log2(item.vertices))
}

// Speedup teorico ideale S_p = T_s / T_p, calcolato dalle formule del
// Capitolo 2 (@eq:mpi-speedup, @eq:omp-speedup, @eq:cuda-sm-speedup) usando
// solo |E|, |V| e p (per CUDA, q = cuda-sm-count): non richiede una baseline
// sequenziale misurata.
#let theoretical-speedup(item) = {
  let n = item.vertices
  let m = item.edges
  let p = worker-count(item)
  if item.backend == "mpi" {
    m / (m / p + n * log2(p))
  } else if item.backend == "openmp" {
    (m + n) / (m / p + n)
  } else {
    cuda-sm-count * cuda-normalized-efficiency(item)
  }
}

#let theoretical-efficiency(item) = {
  if item.backend == "cuda" {
    cuda-normalized-efficiency(item)
  } else {
    theoretical-speedup(item) / worker-count(item)
  }
}

// Soglia operativa di isoefficienza @eq:mpi-threshold / @eq:omp-threshold /
// @eq:cuda-threshold, valutata al grado di parallelismo p (o |V|) usato dalle run.
#let isoefficiency-threshold(backend) = {
  let reference = by-backend(backend).at(0)
  let p = worker-count(reference)
  if backend == "mpi" {
    p * log2(p)
  } else if backend == "openmp" {
    p
  } else {
    log2(reference.vertices)
  }
}

#let isoefficiency-threshold-label(backend) = {
  let threshold = isoefficiency-threshold(backend)
  [$frac(|E|, |V|) >= #calc.round(threshold, digits: 2)$]
}

// Tempo sequenziale T_s misurato da src/sequential.cpp (stesso grafo, stesso
// seed, stessa densità) -- baseline per lo speedup misurato.
#let sequential-exec-at-density(density) = by-backend("sequential").filter(item => item.density == density).at(0).exec

// Speedup misurato S_p = T_s / T_p, con T_s dalla run sequential.cpp alla
// stessa densità e T_p = item.exec.
#let measured-speedup(item) = sequential-exec-at-density(item.density) / item.exec

// Efficienza misurata E_p = S_p / p (per CUDA, p = cuda-sm-count = 142 SM).
#let measured-efficiency(item) = {
  if item.backend == "cuda" {
    measured-speedup(item) / cuda-sm-count
  } else {
    measured-speedup(item) / worker-count(item)
  }
}

// Regressione lineare ai minimi quadrati: usata per leggere la pendenza del
// tempo misurato in funzione della densità e confrontarla con la forma
// prevista da T_p(d) nel Capitolo 2.
#let linear-fit(items, x-value, y-value) = {
  let count = items.len()
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

#let backend-exec-fit(backend) = linear-fit(by-backend(backend), item => item.density, item => item.exec)

#let backend-exec-fit-seconds(item) = fit-seconds(backend-exec-fit(item.backend), item.density)
