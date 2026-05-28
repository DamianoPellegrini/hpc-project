#import "_prelude.typ": cetz, chart
#import "data.typ": backend-color, backend-label, duration, random-runs, timing-breakdown

#let hbar-chart(entries, max-value: auto, width: 8.5, label-width: 2.2) = {
  let values = entries.map(item => item.value)
  let max = if max-value == auto { calc.max(..values) } else { max-value }

  cetz.canvas({
    chart.barchart(
      entries.map(item => ([#item.label], item.value)),
      size: (width, calc.max(1.8, entries.len() * .55)),
      bar-style: index => (fill: entries.at(index).fill, stroke: none),
      x-label: [tempo],
      x-format: value => text(size: 7pt)[#duration(value)],
      x-min: 0,
      x-max: max * 1.15,
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
    hbar-chart(entries, width: 8.0, label-width: 1.2),
    caption: [Tempo totale sul grafo `random` per backend.]
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
    hbar-chart(entries, width: 7.2, label-width: 2.1),
    caption: [Breakdown dei tempi per #backend-label(item.backend) sul grafo #raw(item.graph).]
  )
}
