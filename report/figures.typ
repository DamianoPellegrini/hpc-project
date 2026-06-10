#import "_prelude.typ": cetz, chart, plot
#import "data.typ": (
  backend-color, backend-label, backends, by-backend, duration, isoefficiency-threshold-label,
  measured-speedup, reference-density, reference-runs, swept-densities, theoretical-efficiency,
  theoretical-speedup, worker-count,
)

#let hbar-chart(entries, width: 12.0, height: auto, row-height: 1.1) = {
  let values = entries.map(item => item.value)
  let max = calc.max(..values)

  cetz.canvas({
    chart.barchart(
      entries.map(item => ([#item.label], item.value)),
      size: (width, if height == auto { calc.max(2.8, entries.len() * row-height) } else { height }),
      bar-style: index => (fill: entries.at(index).fill, stroke: none),
      x-label: [tempo (ms)],
      x-format: value => [#(str(calc.round(value * 1000, digits: 3)) + " ms")],
      x-min: 0,
      x-max: max * 1.15,
      y-label: none,
    )
  })
}

// Grafico a barre del tempo totale al punto di riferimento (densità
// `reference-density`), uno per backend.
#let reference-total-chart = {
  let entries = reference-runs.map(item => (
    label: backend-label(item.backend),
    value: item.total,
    fill: backend-color(item.backend),
  ))

  figure(
    hbar-chart(entries, width: 12.0, height: 6, row-height: 1.4),
    caption: [Tempo totale (overhead + esecuzione) alla densità di riferimento $|E| slash |V| = #calc.round(reference-density, digits: 0)$, per backend.],
  )
}

// Barre impilate overhead + esecuzione per ciascun backend, alla densità di
// riferimento. Sostituisce il vecchio breakdown per fasi (scan/reduce/
// contract/compress), non più disponibile nei nuovi CSV.
#let reference-breakdown-stacked-chart = {
  let legend-entry(label, fill) = stack(
    dir: ltr,
    spacing: .28em,
    rect(width: .62em, height: .62em, fill: fill, stroke: none),
    [#label],
  )
  let stack-legend = align(
    center,
    text(
      size: 8pt,
      stack(
        dir: ltr,
        spacing: 1.0em,
        legend-entry([overhead], oklch(48%, 0.02, 260deg)),
        legend-entry([esecuzione], oklch(70%, 0.02, 260deg)),
      ),
    ),
  )
  let rows = reference-runs.map(item => {
    let total = item.overhead + item.exec
    (
      backend-label(item.backend),
      100 * item.overhead / total,
      100 * item.exec / total,
    )
  })
  let value-label(value) = [#(str(calc.round(value, digits: 0)) + "%")]

  figure(
    stack(
      dir: ttb,
      spacing: .45em,
      cetz.canvas({
        import cetz.draw: *
        plot.plot(
          size: (12, 6),
          axis-style: "left",
          x-min: -0.6,
          x-max: 2.6,
          x-tick-step: none,
          x-ticks: reference-runs.enumerate().map(((idx, item)) => (idx, backend-label(item.backend))),
          x-label: none,
          y-label: [quota del tempo totale (%)],
          y-min: 0,
          y-max: 100,
          y-format: value => [#(str(calc.round(value, digits: 0)) + "%")],
          y-grid: true,
          {
            plot.annotate({
              for (idx, row) in rows.enumerate() {
                let overhead-pct = row.at(1)
                let exec-pct = row.at(2)
                rect(
                  (idx - .32, 0),
                  (idx + .32, overhead-pct),
                  fill: oklch(48%, 0.02, 260deg),
                  stroke: white + .6pt,
                )
                rect(
                  (idx - .32, overhead-pct),
                  (idx + .32, 100),
                  fill: oklch(70%, 0.02, 260deg),
                  stroke: white + .6pt,
                )
                content((idx, overhead-pct / 2), text(fill: white, value-label(overhead-pct)), anchor: "center")
                content((idx, overhead-pct + exec-pct / 2), value-label(exec-pct), anchor: "center")
              }
            }, resize: false)
          },
        )
      }),
      stack-legend,
    ),
    caption: [Quota di overhead ed esecuzione pura sul tempo totale, alla densità di riferimento $|E| slash |V| = #calc.round(reference-density, digits: 0)$, per backend.],
  )
}

#let log-ticks(values, y-min, y-max) = values.filter(value => value >= y-min and value <= y-max)

// Tempo totale misurato in funzione della densità, un punto per ogni run
// della sweep, un colore per backend (assi log-log).
#let total-vs-density-plot = {
  let values = backends.map(backend => by-backend(backend).map(item => item.total)).flatten()
  let y-min = calc.min(..values) / 1.5
  let y-max = calc.max(..values) * 1.5
  let x-min = calc.min(..swept-densities) / 1.2
  let x-max = calc.max(..swept-densities) * 2

  cetz.canvas({
    plot.plot(
      size: (12, 6.5),
      axis-style: "left",
      legend: "north",
      x-label: [densità $|E| slash |V|$],
      y-label: [tempo totale],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: swept-densities,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: log-ticks((0.001, 0.01, 0.1, 1), y-min, y-max),
      y-format: value => [#duration(value)],
      x-grid: true,
      y-grid: true,
      {
        for backend in backends {
          plot.add(
            by-backend(backend).map(item => (item.density, item.total)),
            mark: "o",
            mark-size: .105,
            mark-style: (stroke: backend-color(backend), fill: backend-color(backend)),
            line: "linear",
            label: [#backend-label(backend)],
            style: (stroke: backend-color(backend) + 1.2pt),
          )
        }
      },
    )
  })
}

#let total-vs-density-chart = figure(
  total-vs-density-plot,
  caption: [Tempo totale (overhead + esecuzione) misurato sulla sweep `random`, in funzione della densità $|E| slash |V|$, per i tre backend.],
)

// Speedup teorico ideale S_p (@eq:mpi-speedup, @eq:omp-speedup,
// @eq:cuda-sm-speedup), funzione solo di |E|, |V| e p: nessuna baseline
// sequenziale misurata necessaria. Per CUDA, p = q = cuda-sm-count (142 SM,
// NVIDIA L40S).
#let theoretical-speedup-plot = {
  let theory-backends = backends
  let series = theory-backends.map(backend => (
    backend: backend,
    points: by-backend(backend).map(item => (item.density, theoretical-speedup(item))),
  ))
  let values = series.map(s => s.points.map(p => p.at(1))).flatten() + (1,)
  let y-min = calc.min(..values) / 1.5
  let y-max = calc.max(..values) * 1.5
  let x-min = calc.min(..swept-densities) / 1.2
  let x-max = calc.max(..swept-densities) * 2

  cetz.canvas({
    plot.plot(
      size: (12, 6.5),
      axis-style: "left",
      legend: "north",
      x-label: [densità $|E| slash |V|$],
      y-label: [speedup teorico $S_p$],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: swept-densities,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: log-ticks((0.5, 1, 2, 4, 8, 16, 32, 64, 128), y-min, y-max),
      y-format: value => [#(str(calc.round(value, digits: 2)) + "x")],
      x-grid: true,
      y-grid: true,
      {
        for s in series {
          plot.add(
            s.points,
            mark: "o",
            mark-size: .105,
            mark-style: (stroke: backend-color(s.backend), fill: backend-color(s.backend)),
            line: "linear",
            label: [#backend-label(s.backend) teorico],
            style: (stroke: backend-color(s.backend) + 1.2pt),
          )
        }
        plot.add(
          ((x-min, 1), (x-max, 1)),
          mark: "none",
          line: "linear",
          label: [pareggio $S_p = 1$],
          style: (stroke: (paint: black, thickness: .8pt, dash: "dashed")),
        )
      },
    )
  })
}

#let theoretical-speedup-chart = figure(
  theoretical-speedup-plot,
  caption: [Speedup teorico ideale $S_p = T_s slash T_p$ (@eq:mpi-speedup, @eq:omp-speedup, @eq:cuda-sm-speedup) in funzione della densità, calcolato da $|E|$, $|V|$ e $p$ -- nessuna baseline sequenziale misurata è richiesta. Per CUDA, $p=q=142$ (numero di SM della NVIDIA L40S usata per le run): la curva satura rapidamente vicino a $q$ perché $E_p^("CUDA")$ tende a $1$ per densità crescenti.],
)

// Efficienza teorica E_p = S_p / p, con la soglia operativa E_min = 1/2
// (@eq:half-efficiency) come riferimento.
#let theoretical-efficiency-plot = {
  let theory-backends = backends
  let series = theory-backends.map(backend => (
    backend: backend,
    points: by-backend(backend).map(item => (item.density, theoretical-efficiency(item))),
  ))
  let values = series.map(s => s.points.map(p => p.at(1))).flatten() + (0.5,)
  let y-min = calc.min(..values) / 1.5
  let y-max = calc.max(..values) * 1.5
  let x-min = calc.min(..swept-densities) / 1.2
  let x-max = calc.max(..swept-densities) * 2

  cetz.canvas({
    plot.plot(
      size: (12, 6.5),
      axis-style: "left",
      legend: "north",
      x-label: [densità $|E| slash |V|$],
      y-label: [efficienza teorica $E_p = S_p slash p$],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: swept-densities,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: log-ticks((0.01, 0.03, 0.1, 0.3, 1), y-min, y-max),
      y-format: value => [#(str(calc.round(value * 100, digits: 0)) + "%")],
      x-grid: true,
      y-grid: true,
      {
        for s in series {
          plot.add(
            s.points,
            mark: "o",
            mark-size: .105,
            mark-style: (stroke: backend-color(s.backend), fill: backend-color(s.backend)),
            line: "linear",
            label: [#backend-label(s.backend) teorico],
            style: (stroke: backend-color(s.backend) + 1.2pt),
          )
        }
        plot.add(
          ((x-min, 0.5), (x-max, 0.5)),
          mark: "none",
          line: "linear",
          label: [soglia $E_("min") = 1/2$],
          style: (stroke: (paint: black, thickness: .8pt, dash: "dashed")),
        )
      },
    )
  })
}

#let theoretical-efficiency-chart = figure(
  theoretical-efficiency-plot,
  caption: [Efficienza teorica $E_p = S_p slash p$ in funzione della densità, per i tre backend. Per CUDA $p=q$ (numero di SM) e $E_p$ si riduce a $(d+1)slash(d+log_2|V|)$ (@eq:cuda-sm-speedup), indipendente da $q$. La linea tratteggiata è la soglia operativa $E_("min") = 1/2$ (@eq:half-efficiency): MPI la raggiunge per #isoefficiency-threshold-label("mpi"), OpenMP per #isoefficiency-threshold-label("openmp"), CUDA per #isoefficiency-threshold-label("cuda").],
)

// Speedup misurato S_p = T_s / T_p (linea continua, T_s da src/sequential.cpp
// alla stessa densità) confrontato con lo speedup teorico (linea tratteggiata,
// stesso colore) per ciascun backend.
#let measured-vs-theoretical-speedup-plot = {
  let series = backends.map(backend => (
    backend: backend,
    measured: by-backend(backend).map(item => (item.density, measured-speedup(item))),
    theoretical: by-backend(backend).map(item => (item.density, theoretical-speedup(item))),
  ))
  let values = series.map(s => (s.measured + s.theoretical).map(p => p.at(1))).flatten() + (1,)
  let y-min = calc.min(..values) / 1.5
  let y-max = calc.max(..values) * 1.5
  let x-min = calc.min(..swept-densities) / 1.2
  let x-max = calc.max(..swept-densities) * 2

  cetz.canvas({
    plot.plot(
      size: (12, 7),
      axis-style: "left",
      legend: "north",
      legend-style: (item-spacing: (.6, .4)),
      x-label: [densità $|E| slash |V|$],
      y-label: [speedup $S_p$],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: swept-densities,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: log-ticks((0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100), y-min, y-max),
      y-format: value => [#(str(calc.round(value, digits: 2)) + "x")],
      x-grid: true,
      y-grid: true,
      {
        for s in series {
          plot.add(
            s.measured,
            mark: "o",
            mark-size: .105,
            mark-style: (stroke: backend-color(s.backend), fill: backend-color(s.backend)),
            line: "linear",
            label: [#backend-label(s.backend) misurato],
            style: (stroke: backend-color(s.backend) + 1.2pt),
          )
          plot.add(
            s.theoretical,
            mark: "none",
            line: "linear",
            label: [#backend-label(s.backend) teorico],
            style: (stroke: (paint: backend-color(s.backend), thickness: 1.8pt, dash: "dashed")),
          )
        }
        plot.add(
          ((x-min, 1), (x-max, 1)),
          mark: "none",
          line: "linear",
          label: [pareggio $S_p = 1$],
          style: (stroke: (paint: black, thickness: 1.2pt, dash: "dotted")),
        )
      },
    )
  })
}

#let measured-vs-theoretical-speedup-chart = figure(
  measured-vs-theoretical-speedup-plot,
  caption: [Speedup misurato $S_p = T_s slash T_p$ (linea continua) confrontato con lo speedup teorico (linea tratteggiata, @eq:mpi-speedup, @eq:omp-speedup, @eq:cuda-sm-speedup) per i tre backend, in funzione della densità. $T_s$ è il tempo di Borůvka seriale misurato da `sequential_app` sullo stesso grafo (stesso seed, stessa densità) -- vedi Capitolo 1. La linea punteggiata è il pareggio $S_p=1$ con la versione sequenziale.],
)
