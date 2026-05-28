#import "_prelude.typ": cetz, chart, plot
#import "data.typ": (
  backend-color, backend-label, cuda-loop-fit, cuda-sequential-fit, cuda-sweep-runs, cuda-threshold-density, duration,
  edge-density, empirical-speedup, random-runs, ratio, sequential-cpu-seconds, timing-breakdown,
)

#let report-plot-colors = (
  oklch(66%, 0.10, 150deg),
  oklch(66%, 0.09, 238deg),
  oklch(73%, 0.10, 72deg),
  oklch(65%, 0.08, 305deg),
  oklch(58%, 0.03, 250deg),
)

#let report-plot-palette = cetz.palette.new(
  base: (stroke: (paint: black, thickness: 1.1pt), fill: none),
  colors: report-plot-colors,
)

#let report-line-style(index) = {
  report-plot-palette(index, stroke: true, fill: false)
}

#let report-mark-style(index) = {
  let style = report-plot-palette(index, stroke: true, fill: true)
  style.stroke.thickness = .75pt
  style
}

#let hbar-chart(
  entries,
  max-value: auto,
  width: 13.0,
  label-width: 2.2,
  log-scale: false,
) = {
  let values = entries.map(item => item.value)
  let min = calc.min(..values)
  let max = if max-value == auto { calc.max(..values) } else { max-value }
  let x-min = if log-scale { min / 2 } else { 0 }
  let x-max = if log-scale { max * 2 } else { max * 1.15 }
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
      size: (width, calc.max(2.8, entries.len() * .82)),
      bar-style: index => (fill: entries.at(index).fill, stroke: none),
      x-label: [tempo],
      x-format: value => text(size: 8.5pt)[#duration(value)],
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
    hbar-chart(entries, width: 12.8, label-width: 1.2),
    caption: [Tempo totale sul grafo `random` per backend.],
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
      width: 12.8,
      label-width: 2.1,
      log-scale: item.backend == "cuda",
    ),
    caption: [Breakdown dei tempi per #backend-label(item.backend) sul grafo #raw(item.graph).],
  )
}

#let fit-seconds(fit, density, vertices) = {
  fit.intercept + fit.slope * density * vertices
}

#let cuda-sweep-domain() = {
  let measured-max = calc.max(..cuda-sweep-runs.map(edge-density))
  let x-max = if cuda-threshold-density == none {
    measured-max * 1.15
  } else {
    calc.max(measured-max * 1.15, cuda-threshold-density * 1.08)
  }
  (
    x-max: x-max,
    points: (0, 10, 20, 30, 40, 50, 60, 70, 80, x-max).filter(value => value <= x-max),
  )
}

#let cuda-sweep-time-chart = {
  let vertices = cuda-sweep-runs.at(0).vertices
  let domain = cuda-sweep-domain()
  let y-min = 0.004
  let y-max = 0.25
  let y-ticks = (0.005, 0.01, 0.05, 0.1, 0.2)

  figure(
    cetz.canvas({
      plot.plot(
        size: (12.8, 5.6),
        axis-style: "left",
        plot-style: report-line-style,
        mark-style: report-mark-style,
        legend: "north",
        x-label: [densità $m / n$],
        y-label: [tempo],
        x-min: 0,
        x-max: domain.x-max,
        x-grid: true,
        y-grid: true,
        y-mode: "log",
        y-base: 10,
        y-min: y-min,
        y-max: y-max,
        y-tick-step: none,
        y-ticks: y-ticks,
        y-format: value => text(size: 8.5pt)[#duration(value)],
        {
          plot.add(
            cuda-sweep-runs.map(item => (edge-density(item), item.loop)),
            mark: "o",
            mark-size: .11,
            line: "linear",
            label: [CUDA loop],
          )
          plot.add(
            cuda-sweep-runs.map(item => (edge-density(item), sequential-cpu-seconds(item))),
            mark: "square",
            mark-size: .1,
            line: "linear",
            label: [CPU sequenziale],
          )
          plot.add(
            domain.points.map(density => (
              density,
              fit-seconds(cuda-loop-fit, density, vertices),
            )),
            line: "linear",
            label: [fit CUDA],
          )
          plot.add(
            domain.points.map(density => (
              density,
              fit-seconds(cuda-sequential-fit, density, vertices),
            )),
            line: "linear",
            label: [fit CPU],
          )
          if cuda-threshold-density != none {
            plot.add(
              ((cuda-threshold-density, y-min), (cuda-threshold-density, y-max)),
              line: "linear",
              label: [soglia stimata],
            )
          }
        },
      )
    }),
    caption: [Sweep CUDA `random`: confronto fra loop CUDA, baseline CPU sequenziale e regressioni lineari usate per stimare la soglia.],
  )
}

#let cuda-sweep-speedup-chart = {
  let vertices = cuda-sweep-runs.at(0).vertices
  let domain = cuda-sweep-domain()
  let y-max = 1.6

  figure(
    cetz.canvas({
      plot.plot(
        size: (12.8, 4.8),
        axis-style: "left",
        plot-style: report-line-style,
        mark-style: report-mark-style,
        legend: "north",
        x-label: [densità $m / n$],
        y-label: [speedup],
        x-min: 0,
        x-max: domain.x-max,
        y-min: 0,
        y-max: y-max,
        x-grid: true,
        y-grid: true,
        y-format: value => text(size: 8.5pt)[#ratio(value)],
        {
          plot.add(
            cuda-sweep-runs.map(item => (edge-density(item), empirical-speedup(item))),
            mark: "o",
            mark-size: .11,
            line: "linear",
            label: [misurato],
          )
          plot.add(
            domain.points.map(density => (
              density,
              fit-seconds(cuda-sequential-fit, density, vertices) / fit-seconds(cuda-loop-fit, density, vertices),
            )),
            line: "linear",
            label: [fit],
          )
          plot.add(
            ((0, 1), (domain.x-max, 1)),
            line: "linear",
            label: [pareggio],
          )
          if cuda-threshold-density != none {
            plot.add(
              ((cuda-threshold-density, 0), (cuda-threshold-density, y-max)),
              line: "linear",
              label: [soglia stimata],
            )
          }
        },
      )
    }),
    caption: [Speedup sperimentale della sweep CUDA rispetto alla baseline sequenziale CPU. La linea di pareggio è $T_s / T_("CUDA") = 1$.],
  )
}
