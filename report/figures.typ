#import "_prelude.typ": cetz, chart, plot
#import "data.typ": (
  algorithm-seconds, backend-color, backend-label, backends, duration, edge-density,
  empirical-speedup, random-sweep-efficiency,
  sequential-cpu-seconds, timing-value,
  random-runs, random-sweep-backend-fit-seconds, random-sweep-backend-loop-speedup,
  random-sweep-by-backend, random-sweep-crossover-density,
  random-sweep-crossover-label, random-sweep-cpu-fit-seconds,
  random-sweep-projection-densities, random-sweep-reference-seconds, random-sweep-speedup,
  random-sweep-theoretical-speedup, random-sweep-theoretical-speedup-at-density, timing-breakdown,
  worker-count,
)

#let hbar-chart(
  entries,
  max-value: auto,
  width: 12.0,
  height: auto,
  label-width: 2.2,
  log-scale: false,
  row-height: 1.1,
  unit: "auto",
) = {
  let values = entries.map(item => item.value)
  let min = calc.min(..values)
  let max = if max-value == auto { calc.max(..values) } else { max-value }
  let x-min = if log-scale { min / 2 } else { 0 }
  let x-max = if log-scale { max * 2 } else { max * 1.15 }
  let x-label-content = if unit == "ms" {
    [tempo (ms)]
  } else if unit == "ratio" {
    [densità $|E| slash |V|$]
  } else if unit == "speedup" {
    [speedup $S_p = T_s slash T_p$]
  } else {
    [tempo]
  }
  let x-format-fn = if unit == "ms" {
    value => [#(str(calc.round(value * 1000, digits: 3)) + " ms")]
  } else if unit == "ratio" {
    value => [#calc.round(value, digits: 0)]
  } else if unit == "speedup" {
    value => [#(str(calc.round(value, digits: 1)) + "x")]
  } else {
    value => [#duration(value)]
  }
  let log-ticks = (
    0.000001,
    0.00001,
    0.0001,
    0.001,
    0.01,
    0.1,
    1,
  ).filter(value => value >= x-min and value <= x-max)

  cetz.canvas({
    chart.barchart(
      entries.map(item => ([#item.label], item.value)),
      size: (width, if height == auto { calc.max(2.8, entries.len() * row-height) } else { height }),
      bar-style: index => (fill: entries.at(index).fill, stroke: none),
      x-label: x-label-content,
      x-format: x-format-fn,
      x-mode: if log-scale { "log" } else { "lin" },
      x-base: 10,
      x-tick-step: if log-scale { none } else { auto },
      x-ticks: if log-scale { log-ticks } else { auto },
      x-min: x-min,
      x-max: x-max,
      y-label: none,
    )
  })
}

#let random-total-chart = {
  let entries = random-runs.map(item => (
    label: backend-label(item.backend),
    value: item.total,
    text: duration(item.total),
    fill: backend-color(item.backend),
  ))

  figure(
    hbar-chart(entries, width: 12.0, height: 6, label-width: 1.2, unit: "ms"),
    caption: [Tempo totale sul grafo `random` per backend, espresso in millisecondi.],
  )
}

#let backend-breakdown-chart(item) = {
  let entries = timing-breakdown(item).map(phase => (
    label: phase.phase,
    value: phase.seconds,
    text: duration(phase.seconds),
    fill: backend-color(item.backend),
  ))

  figure(
    hbar-chart(
      entries,
      width: 12.0,
      height: 6,
      label-width: 2.1,
      log-scale: item.backend == "cuda",
      unit: "ms",
    ),
    caption: [Breakdown dei tempi per #backend-label(item.backend) sul grafo #raw(item.graph), con asse in millisecondi.],
  )
}

#let random-breakdown-bucket(item, bucket) = {
  if bucket == "scan" {
    if item.backend == "mpi" {
      timing-value(item, "max_local_compute_seconds", fallback: timing-value(item, "scan_seconds"))
    } else {
      timing-value(item, "scan_seconds")
    }
  } else if bucket == "reduce" {
    if item.backend == "mpi" {
      timing-value(item, "max_reduce_seconds", fallback: timing-value(item, "reduce_seconds"))
    } else if item.backend == "openmp" {
      timing-value(item, "reduce_seconds")
    } else {
      0
    }
  } else if bucket == "contract" {
    if item.backend == "mpi" {
      timing-value(item, "max_contract_seconds", fallback: timing-value(item, "contract_seconds"))
    } else if item.backend == "cuda" {
      timing-value(item, "contract_kernel_seconds", fallback: timing-value(item, "contract_seconds"))
    } else {
      timing-value(item, "contract_seconds")
    }
  } else {
    let known = (
      random-breakdown-bucket(item, "scan")
      + random-breakdown-bucket(item, "reduce")
      + random-breakdown-bucket(item, "contract")
    )
    let total = if item.backend == "cuda" { algorithm-seconds(item) } else { item.loop }
    calc.max(0, total - known)
  }
}

#let random-breakdown-stacked-chart = {
  let phase-color(backend, index) = {
    let base = backend-color(backend)
    if index == 0 {
      base
    } else if index == 1 {
      base.lighten(28%)
    } else if index == 2 {
      base.darken(12%)
    } else {
      base.lighten(48%)
    }
  }
  let legend-entry(label, note, fill) = {
    stack(
      dir: ltr,
      spacing: .28em,
      rect(width: .62em, height: .62em, fill: fill, stroke: none),
      [#label#note],
    )
  }
  let stack-legend = align(
    center,
    text(
      size: 8pt,
      stack(
        dir: ltr,
        spacing: 1.0em,
        [Ordine dal basso verso l'alto:],
        legend-entry([scan], [], phase-color("cuda", 0)),
        legend-entry([riduzione], [], phase-color("cuda", 1)),
        legend-entry([contrazione], [], phase-color("cuda", 2)),
        legend-entry([residuo/overhead], [ (in cima)], phase-color("cuda", 3)),
      ),
    ),
  )
  let rows = random-runs.map(item => {
    let scan = random-breakdown-bucket(item, "scan")
    let reduce = random-breakdown-bucket(item, "reduce")
    let contract = random-breakdown-bucket(item, "contract")
    let overhead = random-breakdown-bucket(item, "overhead")
    let total = scan + reduce + contract + overhead
    (
      backend-label(item.backend),
      100 * scan / total,
      100 * reduce / total,
      100 * contract / total,
      100 * overhead / total,
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
          x-ticks: random-runs.enumerate().map(((idx, item)) => (idx, backend-label(item.backend))),
          x-label: none,
          y-label: [quota del tempo (%)],
          y-min: 0,
          y-max: 100,
          y-format: value => [#(str(calc.round(value, digits: 0)) + "%")],
          y-grid: true,
          {
            plot.annotate({
              for (idx, item) in random-runs.enumerate() {
                let row = rows.at(idx)
                let values = (row.at(1), row.at(2), row.at(3), row.at(4))
                let starts = (
                  0,
                  values.at(0),
                  values.at(0) + values.at(1),
                  values.at(0) + values.at(1) + values.at(2),
                )
                for (phase-idx, value) in values.enumerate() {
                  let y0 = starts.at(phase-idx)
                  let y1 = y0 + value
                  rect(
                    (idx - .32, y0),
                    (idx + .32, y1),
                    fill: phase-color(item.backend, phase-idx),
                    stroke: white + .6pt,
                  )
                  if value >= 3 {
                    content(
                      (idx, (y0 + y1) / 2),
                      if phase-idx == 0 or phase-idx == 2 {
                        text(fill: white, value-label(value))
                      } else {
                        value-label(value)
                      },
                      anchor: "center",
                    )
                  }
                }
              }
            }, resize: false)
          },
        )
      }),
      stack-legend,
    ),
    caption: [Breakdown percentuale in pila sul grafo `random`: ogni barra usa il colore del backend e tonalità derivate per le fasi. MPI e OpenMP usano il loop MST, mentre CUDA usa il tempo algoritmo device; i segmenti, dal basso verso l'alto, rappresentano scan, riduzione, contrazione e residuo/overhead.],
  )
}

#let random-sweep-x-values = random-sweep-by-backend("mpi").map(edge-density)

#let sweep-y-ticks(y-min, y-max) = (
  0.00001,
  0.0001,
  0.001,
  0.01,
  0.1,
  1,
).filter(value => value >= y-min and value <= y-max)

#let comparison-colors(backend) = (
  oklch(48%, 0.02, 260deg),
  backend-color(backend),
  oklch(48%, 0.02, 260deg),
  backend-color(backend),
  oklch(62%, 0.04, 20deg),
)

#let comparison-palette(backend) = cetz.palette.new(
  base: (stroke: (paint: black, thickness: 1.05pt), fill: none),
  colors: comparison-colors(backend),
)

#let comparison-line-style(backend, index) = {
  let style = comparison-palette(backend)(index, stroke: true, fill: false)
  if index >= 2 {
    style.stroke.thickness = .85pt
    style.stroke.dash = "dashed"
  }
  style
}

#let comparison-mark-style(backend, index) = {
  let style = comparison-palette(backend)(index, stroke: true, fill: true)
  style.stroke.thickness = .75pt
  style
}

#let backend-cpu-comparison-plot(backend) = {
  let runs = random-sweep-by-backend(backend)
  let threshold = random-sweep-crossover-density(backend)
  let cpu-values = runs.map(random-sweep-reference-seconds)
  let measured-values = runs.map(algorithm-seconds)
  let cpu-fit-values = runs.map(random-sweep-cpu-fit-seconds)
  let backend-fit-values = runs.map(random-sweep-backend-fit-seconds)
  let values = measured-values + cpu-values + cpu-fit-values + backend-fit-values
  let positive-values = values.filter(value => value > 0)
  let y-min = calc.min(..positive-values) / 2
  let y-max = calc.max(..positive-values) * 2
  let x-values = random-sweep-x-values
  let x-min = calc.min(..x-values) / 1.2
  let x-max = calc.max(calc.max(..x-values) * 1.2, if threshold == none { 0 } else { threshold * 1.05 })
  let line-style = index => comparison-line-style(backend, index)
  let mark-style = index => comparison-mark-style(backend, index)

  cetz.canvas({
    plot.plot(
      size: (12, 6),
      axis-style: "left",
      plot-style: line-style,
      mark-style: mark-style,
      legend: "north",
      x-label: none,
      y-label: [tempo algoritmo],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: x-values,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: sweep-y-ticks(y-min, y-max),
      y-format: value => [#duration(value)],
      x-grid: true,
      y-grid: true,
      {
        plot.add(
          runs.map(item => (edge-density(item), random-sweep-reference-seconds(item))),
          mark: "diamond",
          mark-size: .105,
          line: "linear",
          label: [CPU misurato],
        )
        plot.add(
          runs.map(item => (edge-density(item), algorithm-seconds(item))),
          mark: "o",
          mark-size: .105,
          line: "linear",
          label: [#backend-label(backend) misurato],
        )
        plot.add(
          runs.map(item => (edge-density(item), random-sweep-cpu-fit-seconds(item))),
          mark: "none",
          line: "linear",
          label: [fit CPU],
        )
        plot.add(
          runs.map(item => (edge-density(item), random-sweep-backend-fit-seconds(item))),
          mark: "none",
          line: "linear",
          label: [fit #backend-label(backend)],
        )
        if threshold != none {
          plot.add(
            ((threshold, y-min), (threshold, y-max)),
            line: "linear",
            label: random-sweep-crossover-label(backend),
          )
        }
      },
    )
  })
}

#let backend-cpu-comparison-chart(backend) = {
  figure(
    backend-cpu-comparison-plot(backend),
    caption: [Sweep `random` #backend-label(backend): confronto con la baseline sequenziale CPU. Le linee fitted stimano la soglia in cui il tempo parallelo scende sotto quello sequenziale.],
  )
}

#let speedup-colors(backend) = (
  backend-color(backend),
  backend-color(backend),
  oklch(58%, 0.08, 305deg),
  oklch(62%, 0.04, 20deg),
  oklch(48%, 0.02, 260deg),
)

#let speedup-palette(backend) = cetz.palette.new(
  base: (stroke: (paint: black, thickness: 1.05pt), fill: none),
  colors: speedup-colors(backend),
)

#let speedup-line-style(backend, index) = {
  let style = speedup-palette(backend)(index, stroke: true, fill: false)
  if index == 1 {
    style.stroke.thickness = 1.35pt
    style.stroke.dash = "dashed"
  } else if index >= 2 {
    style.stroke.thickness = 1.15pt
    style.stroke.dash = "dashed"
  }
  style
}

#let speedup-mark-style(backend, index) = {
  let style = speedup-palette(backend)(index, stroke: true, fill: true)
  style.stroke.thickness = .75pt
  style
}

#let backend-speedup-theory-plot(backend) = {
  let runs = random-sweep-by-backend(backend)
  let measured-values = runs.map(random-sweep-speedup)
  let theory-values = random-sweep-projection-densities.map(density => random-sweep-theoretical-speedup-at-density(backend, density))
  let reference-values = (1,)
  let values = measured-values + theory-values + reference-values
  let positive-values = values.filter(value => value > 0)
  let y-min = calc.min(..positive-values) / 1.8
  let y-max = calc.max(..positive-values) * 1.8
  let x-values = random-sweep-projection-densities
  let x-min = calc.min(..x-values) / 1.2
  let x-max = calc.max(..x-values) * 1.2
  let y-ticks = (
    0.03,
    0.1,
    0.5,
    1,
    2,
    5,
    10,
    20,
    100,
    200,
    1000,
    10000,
    100000,
    200000,
  ).filter(value => value >= y-min and value <= y-max)
  let line-style = index => speedup-line-style(backend, index)
  let mark-style = index => speedup-mark-style(backend, index)

  cetz.canvas({
    plot.plot(
      size: (12, 6),
      axis-style: "left",
      plot-style: line-style,
      mark-style: mark-style,
      legend: "north",
      x-label: none,
      y-label: [speedup $T_s / T_p$],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: x-values,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: y-ticks,
      y-format: value => [#(str(calc.round(value, digits: 2)) + "x")],
      x-grid: true,
      y-grid: true,
      {
        plot.add(
          runs.map(item => (edge-density(item), random-sweep-speedup(item))),
          mark: "o",
          mark-size: .105,
          line: "linear",
          label: [#backend-label(backend) misurato],
        )
        plot.add(
          x-values.map(density => (density, random-sweep-theoretical-speedup-at-density(backend, density))),
          mark: "none",
          line: "linear",
          label: [#backend-label(backend) ideale],
        )
      },
    )
  })
}

#let backend-speedup-theory-chart(backend) = {
  figure(
    backend-speedup-theory-plot(backend),
    caption: if backend == "cuda" {
      [Sweep `random` CUDA: speedup misurato e limite ideale basato sugli SM. La distanza dal limite ideale evidenzia che il modello non include accessi globali, atomiche e sincronizzazioni tra kernel.]
    } else {
      [Sweep `random` #backend-label(backend): speedup misurato e modello teorico ideale per il backend.]
    },
  )
}

#let efficiency-chart-colors = (
  backend-color("mpi"),
  backend-color("openmp"),
  backend-color("cuda"),
  oklch(48%, 0.02, 260deg),
)

#let efficiency-chart-palette = cetz.palette.new(
  base: (stroke: (paint: black, thickness: 1.05pt), fill: none),
  colors: efficiency-chart-colors,
)

#let efficiency-line-style(index) = {
  let style = efficiency-chart-palette(index, stroke: true, fill: false)
  if index >= 3 {
    style.stroke.thickness = .85pt
    style.stroke.dash = "dashed"
  } else {
    style.stroke.thickness = 1.2pt
  }
  style
}

#let efficiency-mark-style(index) = {
  let style = efficiency-chart-palette(index, stroke: true, fill: true)
  style.stroke.thickness = .75pt
  style
}

#let efficiency-isoefficiency-plot = {
  let target-backends = backends
  let series = target-backends.map(backend => (
    backend: backend,
    points: random-sweep-by-backend(backend).map(item => (
      edge-density(item), random-sweep-efficiency(item),
    )),
  ))
  let x-values = random-sweep-projection-densities
  let x-min = calc.min(..x-values) / 1.2
  let x-max = calc.max(..x-values) * 1.2
  let all-values = series.map(s => s.points.map(p => p.at(1))).flatten().filter(v => v > 0)
  let y-min = calc.min(..all-values) / 1.5
  let y-max = calc.max(..all-values) * 1.5
  let y-ticks = (0.01, 0.03, 0.1, 0.3, 1, 1.5)
    .filter(value => value >= y-min and value <= y-max)

  cetz.canvas({
    plot.plot(
      size: (12, 6.5),
      axis-style: "left",
      plot-style: efficiency-line-style,
      mark-style: efficiency-mark-style,
      legend: "north",
      x-label: [densità $|E| slash |V|$],
      y-label: [efficienza $E_p = S_p slash p$],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: x-values,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: y-ticks,
      y-format: value => [#(str(calc.round(value * 100, digits: 0)) + "%")],
      x-grid: true,
      y-grid: true,
      {
        for s in series {
          plot.add(
            s.points,
            mark: "o",
            mark-size: .105,
            line: "linear",
            label: [#backend-label(s.backend) misurato],
          )
        }
      },
    )
  })
}

#let efficiency-isoefficiency-chart = figure(
  efficiency-isoefficiency-plot,
  caption: [Efficienza misurata $E_p = S_p slash p$ dei tre backend sulla sweep `random`, in
    funzione della densità $|E| slash |V|$. Asse verticale logaritmico: per CUDA $p = q = 142$,
    di ordini di grandezza superiore a MPI e OpenMP, quindi a parità di speedup l'efficienza
    in questo senso classico resta strutturalmente più bassa e in una banda più stretta.],
)

#let cross-backend-theory-colors = (
  backend-color("mpi"),
  backend-color("mpi").lighten(45%),
  backend-color("openmp"),
  backend-color("openmp").lighten(45%),
  backend-color("cuda"),
  backend-color("cuda").lighten(45%),
  oklch(42%, 0.02, 260deg),
)

#let cross-backend-theory-palette = cetz.palette.new(
  base: (stroke: (paint: black, thickness: 1.05pt), fill: none),
  colors: cross-backend-theory-colors,
)

#let cross-backend-theory-line-style(index) = {
  let style = cross-backend-theory-palette(index, stroke: true, fill: false)
  if index == 1 or index == 3 or index == 5 {
    style.stroke.thickness = 1.1pt
    style.stroke.dash = "dashed"
  } else if index == 6 {
    style.stroke.thickness = .9pt
    style.stroke.dash = "dotted"
  } else {
    style.stroke.thickness = 1.25pt
  }
  style
}

#let cross-backend-theory-mark-style(index) = {
  let style = cross-backend-theory-palette(index, stroke: true, fill: true)
  style.stroke.thickness = .75pt
  style
}

#let cross-backend-speedup-theory-plot = {
  let x-values = random-sweep-projection-densities
  let dense-random-density = calc.max(..x-values)
  let series = backends.map(backend => (
    backend: backend,
    measured: random-sweep-by-backend(backend).map(item => (
      edge-density(item), random-sweep-speedup(item),
    )),
    theoretical: x-values.map(density => (
      density, random-sweep-theoretical-speedup-at-density(backend, density),
    )),
  ))
  let x-min = calc.min(..x-values) / 1.2
  let x-max = calc.max(..x-values) * 1.2
  let all-values = series
    .map(s => s.measured.map(p => p.at(1)) + s.theoretical.map(p => p.at(1)))
    .flatten()
    .filter(v => v > 0)
  let y-min = calc.min(..all-values) / 1.8
  let y-max = calc.max(..all-values) * 1.8
  let y-ticks = (0.03, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500)
    .filter(value => value >= y-min and value <= y-max)

  cetz.canvas({
    plot.plot(
      size: (12, 7),
      axis-style: "left",
      plot-style: cross-backend-theory-line-style,
      mark-style: cross-backend-theory-mark-style,
      legend: "north",
      x-label: [densità $|E| slash |V|$],
      y-label: [speedup $S_p = T_s slash T_p$],
      x-mode: "log",
      x-base: 10,
      x-min: x-min,
      x-max: x-max,
      x-tick-step: none,
      x-ticks: x-values,
      x-format: value => [#calc.round(value, digits: 0)],
      y-mode: "log",
      y-base: 10,
      y-min: y-min,
      y-max: y-max,
      y-tick-step: none,
      y-ticks: y-ticks,
      y-format: value => [#(str(calc.round(value, digits: 2)) + "x")],
      x-grid: true,
      y-grid: true,
      {
        for s in series {
          plot.add(
            s.measured,
            mark: "o",
            mark-size: .105,
            line: "linear",
            label: [#backend-label(s.backend) misurato],
          )
          plot.add(
            s.theoretical,
            mark: "none",
            line: "linear",
            label: [#backend-label(s.backend) teorico],
          )
        }
        plot.add(
          ((dense-random-density, y-min), (dense-random-density, y-max)),
          mark: "none",
          line: "linear",
          label: [random più denso],
        )
      },
    )
  })
}

#let cross-backend-speedup-theory-chart = figure(
  cross-backend-speedup-theory-plot,
  caption: [Speedup misurato $S_p = T_s slash T_p$ contro lo speedup ideale previsto dal
    modello teorico (@eq:mpi-speedup, @eq:omp-speedup, @eq:cuda-sm-speedup), sulla sweep
    `random` e per i tre backend. Le linee continue con marker sono i valori misurati,
    le linee tratteggiate sono il modello valutato alle stesse densità e allo stesso grado
    di parallelismo di @tab:run-config; la linea verticale punteggiata identifica il punto
    `random` più denso disponibile nella sweep.],
)
