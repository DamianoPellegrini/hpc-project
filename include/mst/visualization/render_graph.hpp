#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <sys/ioctl.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include "mst/core/graph.hpp"

namespace mst::visualization {

inline constexpr const char *clear_screen = "\x1b[2J";
inline constexpr const char *cursor_home = "\x1b[H";
inline constexpr const char *cursor_show = "\x1b[?25h";
inline constexpr const char *reset = "\x1b[0m";
inline constexpr const char *dim = "\x1b[2m";
inline constexpr const char *bold = "\x1b[1m";
inline constexpr const char *mst_green = "\x1b[38;5;46m";
inline constexpr const char *yellow = "\x1b[33m";
inline constexpr const char *white = "\x1b[37m";
inline constexpr const char *muted_gray = "\x1b[38;5;240m";
inline constexpr const char *muted_blue = "\x1b[38;5;67m";
inline constexpr const char *muted_teal = "\x1b[38;5;73m";
inline constexpr const char *muted_violet = "\x1b[38;5;103m";
inline constexpr const char *muted_sand = "\x1b[38;5;137m";
inline constexpr std::string_view edge_glyph = "•";

/// Coordinate continue del layout, prima della proiezione sulla griglia del terminale.
struct point {
  double x;
  double y;
};

/// Coordinate intere di una cella, dopo la proiezione di un `point`.
struct canvas_point {
  int x;
  int y;
};

/// Dimensioni rilevate del terminale (colonne x righe).
struct viewport_size {
  int width;
  int height;
};

/// Una cella della griglia: testo, colore ANSI, grassetto.
struct cell {
  std::string text = " ";
  const char *color = reset;
  bool is_bold = false;
};

/// Griglia di celle con colori ANSI: le primitive di disegno scrivono qui, `render` la emette su uno stream.
class canvas {
public:
  canvas(int width, int height)
      : width_(width), height_(height),
        cells_(static_cast<std::size_t>(width * height)) {}

  int width() const noexcept { return width_; }
  int height() const noexcept { return height_; }

  void set(int x, int y, std::string_view text = " ", const char *color = white,
           bool is_bold = false) {
    if (x < 0 || y < 0 || x >= width_ || y >= height_) {
      return;
    }

    cell &target = cells_[index(x, y)];
    target.text = std::string{text};
    target.color = color;
    target.is_bold = is_bold;
  }

  void set(int x, int y, char ch, const char *color, bool is_bold = false) {
    set(x, y, std::string_view(&ch, 1), color, is_bold);
  }

  void draw_text(int x, int y, const std::string &text,
                 const char *color = white, bool is_bold = false) {
    for (std::size_t offset = 0; offset < text.size(); ++offset) {
      set(x + static_cast<int>(offset), y, text[offset], color, is_bold);
    }
  }

  void fill_rect(int left, int top, int width, int height, char ch = ' ',
                 const char *color = reset, bool is_bold = false) {
    for (int y = top; y < top + height; ++y) {
      for (int x = left; x < left + width; ++x) {
        set(x, y, ch, color, is_bold);
      }
    }
  }

  /// Segmento fra due punti con Bresenham (aritmetica intera, nessuna trigonometria); `sx`/`sy` gestiscono qualunque direzione.
  void draw_line(int x0, int y0, int x1, int y1, std::string_view text,
                 const char *color, bool is_bold = false) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;

    while (true) {
      set(x0, y0, text, color, is_bold);
      if (x0 == x1 && y0 == y1) {
        break;
      }

      const int twice_error = 2 * error;
      if (twice_error >= dy) {
        error += dy;
        x0 += sx;
      }
      if (twice_error <= dx) {
        error += dx;
        y0 += sy;
      }
    }
  }

  /// Emette la griglia riga per riga: `emit_style` scrive l'escape ANSI solo
  /// quando lo stile cambia, per non ripeterlo a ogni carattere.
  void render(std::ostream &out) const {
    const char *current_color = reset;
    bool current_bold = false;

    auto emit_style = [&](const cell &value) {
      if (value.color == current_color && value.is_bold == current_bold) {
        return;
      }

      out << reset;
      if (value.color != reset) {
        out << value.color;
      }
      if (value.is_bold) {
        out << bold;
      }
      current_color = value.color;
      current_bold = value.is_bold;
    };

    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        const cell &value = cells_[index(x, y)];
        emit_style(value);
        out << value.text;
      }
      out << reset << '\n';
      current_color = reset;
      current_bold = false;
    }
    out << reset;
  }

private:
  int index(int x, int y) const noexcept { return y * width_ + x; }

  int width_;
  int height_;
  std::vector<cell> cells_;
};

/// Posizione di partenza: i vertici distribuiti su una circonferenza in base
/// al loro indice — punto di partenza deterministico per `make_force_layout`.
inline point make_initial_point(int vertex, int vertex_count) {
  constexpr double pi = 3.14159265358979323846;
  const double angle = (2.0 * pi * static_cast<double>(vertex)) /
                       static_cast<double>(vertex_count);
  return point{std::cos(angle), std::sin(angle)};
}

/// Layout "force-directed" alla Fruchterman-Reingold: parte dal cerchio di
/// `make_initial_point` e itera fra repulsione (allontana tutte le coppie) e
/// attrazione lungo gli archi, con una "temperatura" che si raffredda per farlo convergere.
inline std::vector<point>
make_force_layout(const mst::core::validated_graph &graph) {
  const int vertex_count = graph.vertex_count();
  std::vector<point> positions;
  positions.reserve(static_cast<std::size_t>(vertex_count));
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    positions.push_back(make_initial_point(vertex, vertex_count));
  }

  if (vertex_count <= 1) {
    return positions;
  }

  const double area = 4.0;
  const double ideal_length =
      std::sqrt(area / static_cast<double>(vertex_count));
  double temperature = 0.18;
  constexpr int iteration_count = 180;

  for (int iteration = 0; iteration < iteration_count; ++iteration) {
    std::vector<point> displacement(static_cast<std::size_t>(vertex_count),
                                    point{0.0, 0.0});

    for (int left = 0; left < vertex_count; ++left) {
      for (int right = left + 1; right < vertex_count; ++right) {
        double dx = positions[static_cast<std::size_t>(left)].x -
                    positions[static_cast<std::size_t>(right)].x;
        double dy = positions[static_cast<std::size_t>(left)].y -
                    positions[static_cast<std::size_t>(right)].y;
        double distance = std::sqrt(dx * dx + dy * dy);
        if (distance < 1e-6) {
          dx = 1e-3 * static_cast<double>(right - left);
          dy = 1e-3;
          distance = std::sqrt(dx * dx + dy * dy);
        }

        const double force = (ideal_length * ideal_length) / distance;
        const double unit_x = dx / distance;
        const double unit_y = dy / distance;

        displacement[static_cast<std::size_t>(left)].x += unit_x * force;
        displacement[static_cast<std::size_t>(left)].y += unit_y * force;
        displacement[static_cast<std::size_t>(right)].x -= unit_x * force;
        displacement[static_cast<std::size_t>(right)].y -= unit_y * force;
      }
    }

    for (const mst::core::edge &edge_value : graph.edges()) {
      const std::size_t left = edge_value.u.index();
      const std::size_t right = edge_value.v.index();
      double dx = positions[left].x - positions[right].x;
      double dy = positions[left].y - positions[right].y;
      double distance = std::sqrt(dx * dx + dy * dy);
      if (distance < 1e-6) {
        distance = 1e-6;
      }

      const double force = (distance * distance) / ideal_length;
      const double unit_x = dx / distance;
      const double unit_y = dy / distance;

      displacement[left].x -= unit_x * force;
      displacement[left].y -= unit_y * force;
      displacement[right].x += unit_x * force;
      displacement[right].y += unit_y * force;
    }

    for (int vertex = 0; vertex < vertex_count; ++vertex) {
      point &position = positions[static_cast<std::size_t>(vertex)];
      const point delta = displacement[static_cast<std::size_t>(vertex)];
      const double length = std::sqrt(delta.x * delta.x + delta.y * delta.y);
      if (length > 1e-9) {
        const double step = std::min(length, temperature);
        position.x += (delta.x / length) * step;
        position.y += (delta.y / length) * step;
      }
      position.x = std::clamp(position.x, -1.5, 1.5);
      position.y = std::clamp(position.y, -1.5, 1.5);
    }

    temperature *= 0.97;
  }

  return positions;
}

/// Layout continuo -> griglia: bounding box, normalizzazione in [0, 1] e riscalatura nell'area disegnabile (margini esclusi).
inline std::vector<canvas_point>
project_layout(const std::vector<point> &layout, int width, int height) {
  double min_x = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  for (const point &value : layout) {
    min_x = std::min(min_x, value.x);
    max_x = std::max(max_x, value.x);
    min_y = std::min(min_y, value.y);
    max_y = std::max(max_y, value.y);
  }

  const double span_x = std::max(max_x - min_x, 1e-6);
  const double span_y = std::max(max_y - min_y, 1e-6);
  const int margin_x = 6;
  const int margin_y = 3;
  const double drawable_width = static_cast<double>(width - 1 - 2 * margin_x);
  const double drawable_height = static_cast<double>(height - 1 - 2 * margin_y);

  std::vector<canvas_point> projected;
  projected.reserve(layout.size());
  for (const point &value : layout) {
    const double normalized_x = (value.x - min_x) / span_x;
    const double normalized_y = (value.y - min_y) / span_y;
    projected.push_back(canvas_point{
        margin_x + static_cast<int>(std::round(normalized_x * drawable_width)),
        margin_y +
            static_cast<int>(std::round(normalized_y * drawable_height))});
  }
  return projected;
}

/// Stessi due archi come non orientati (estremi normalizzati), anche se `u`/`v` sono scambiati.
inline bool same_undirected_edge(const mst::core::edge &left,
                                 const mst::core::edge &right) noexcept {
  return mst::core::normalized_endpoints(left) ==
         mst::core::normalized_endpoints(right);
}

/// Etichetta di un vertice: il suo indice, come stringa.
inline std::string node_label(int vertex) { return std::to_string(vertex); }

/// Colore "spento" per un arco fuori MST: hash deterministico degli estremi
/// normalizzati sulla tavolozza, così archi diversi tendono a colori diversi
/// e il risultato resta riproducibile.
inline const char *non_mst_edge_color(const mst::core::edge &edge_value) {
  constexpr const char *palette[] = {
      muted_gray, muted_blue, muted_teal, muted_violet, muted_sand,
  };
  const auto endpoints = mst::core::normalized_endpoints(edge_value);
  const std::size_t palette_index =
      static_cast<std::size_t>((endpoints.first * 31 + endpoints.second) %
                               static_cast<int>(std::size(palette)));
  return palette[palette_index];
}

/// Vero se l'arco è (come non orientato) uno di quelli ammessi nell'MST — decide se va in verde o nel colore "spento".
inline bool is_mst_edge(const mst::core::edge &edge_value,
                        const std::vector<mst::core::mst_edge> &mst_edges) {
  return std::any_of(mst_edges.begin(), mst_edges.end(),
                     [&](const mst::core::mst_edge &mst_edge_value) {
                       return same_undirected_edge(edge_value,
                                                   mst_edge_value.value);
                     });
}

/// Testo centrato su `center_x` con un riquadro di sfondo intorno: usato per
/// etichette e pesi, per restare leggibili anche sopra le linee disegnate.
inline void draw_padded_text(canvas &target, int center_x, int center_y,
                             std::string_view text, int horizontal_padding,
                             const char *color, bool is_bold) {
  const int width = static_cast<int>(text.size()) + (2 * horizontal_padding);
  const int start_x = center_x - (width / 2);
  target.fill_rect(start_x, center_y, width, 1, ' ');
  target.draw_text(start_x + horizontal_padding, center_y, std::string{text},
                   color, is_bold);
}

/// Peso di un arco nel punto medio fra i suoi estremi proiettati, via `draw_padded_text`.
inline void draw_edge_weight(canvas &target, const canvas_point &from,
                             const canvas_point &to, int weight,
                             int horizontal_padding, const char *color) {
  const double midpoint_x = (static_cast<double>(from.x) + to.x) * 0.5;
  const double midpoint_y = (static_cast<double>(from.y) + to.y) * 0.5;
  const std::string label = std::to_string(weight);
  const int label_x = static_cast<int>(std::round(midpoint_x));
  const int label_y = static_cast<int>(std::round(midpoint_y));
  draw_padded_text(target, label_x, label_y, label, horizontal_padding, color,
                   false);
}

/// Dimensioni del terminale: prima `ioctl(TIOCGWINSZ)` se interattivo, poi
/// `COLUMNS`/`LINES` (utili in batch con output rediretto), infine un default ragionevole.
inline viewport_size detect_viewport_size() {
  constexpr int default_width = 96;
  constexpr int default_height = 28;

  winsize terminal_size{};
  if (isatty(STDOUT_FILENO) != 0 &&
      ioctl(STDOUT_FILENO, TIOCGWINSZ, &terminal_size) == 0 &&
      terminal_size.ws_col > 0 && terminal_size.ws_row > 0) {
    return viewport_size{static_cast<int>(terminal_size.ws_col),
                         static_cast<int>(terminal_size.ws_row)};
  }

  const char *columns = std::getenv("COLUMNS");
  const char *lines = std::getenv("LINES");
  if (columns != nullptr && lines != nullptr) {
    const int width = std::atoi(columns);
    const int height = std::atoi(lines);
    if (width > 0 && height > 0) {
      return viewport_size{width, height};
    }
  }

  return viewport_size{default_width, default_height};
}

/// Disegna gli archi MST oppure non-MST (`draw_mst_edges`): chiamata due
/// volte, prima per lo sfondo poi per l'MST, così quest'ultimo finisce sopra (verde, grassetto).
inline void draw_graph_edges(canvas &target,
                             const mst::core::validated_graph &graph,
                             const std::vector<canvas_point> &points,
                             const std::vector<mst::core::mst_edge> &mst_edges,
                             bool draw_mst_edges) {
  for (const mst::core::edge &edge_value : graph.edges()) {
    const bool mst_edge = is_mst_edge(edge_value, mst_edges);
    if (mst_edge != draw_mst_edges) {
      continue;
    }

    const canvas_point &from = points[edge_value.u.index()];
    const canvas_point &to = points[edge_value.v.index()];
    const char *edge_color =
        mst_edge ? mst_green : non_mst_edge_color(edge_value);
    target.draw_line(from.x, from.y, to.x, to.y, edge_glyph, edge_color,
                     mst_edge);
  }
}

/// Etichetta di ogni vertice nella sua posizione, in giallo e grassetto, sopra agli archi.
inline void draw_graph_vertices(canvas &target,
                                const mst::core::validated_graph &graph,
                                const std::vector<canvas_point> &points) {
  for (int vertex = 0; vertex < graph.vertex_count(); ++vertex) {
    const canvas_point &location = points[static_cast<std::size_t>(vertex)];
    const std::string label = node_label(vertex);
    draw_padded_text(target, location.x, location.y, label, 2, yellow, true);
  }
}

/// Peso di ogni arco nel suo colore (verde se nell'MST, altrimenti "spento"), al centro dell'arco proiettato.
inline void
draw_graph_weights(canvas &target, const mst::core::validated_graph &graph,
                   const std::vector<canvas_point> &points,
                   const std::vector<mst::core::mst_edge> &mst_edges) {
  for (const mst::core::edge &edge_value : graph.edges()) {
    const canvas_point &from = points[edge_value.u.index()];
    const canvas_point &to = points[edge_value.v.index()];
    const bool mst_edge = is_mst_edge(edge_value, mst_edges);
    const char *edge_color =
        mst_edge ? mst_green : non_mst_edge_color(edge_value);
    draw_edge_weight(target, from, to, edge_value.weight.value(), 1,
                     edge_color);
  }
}

/// Ordine di disegno per la leggibilità: archi non-MST, poi MST, poi pesi, infine etichette in primo piano.
inline void
draw_graph_content(canvas &target, const mst::core::validated_graph &graph,
                   const std::vector<canvas_point> &points,
                   const std::vector<mst::core::mst_edge> &mst_edges) {
  draw_graph_edges(target, graph, points, mst_edges, false);
  draw_graph_edges(target, graph, points, mst_edges, true);
  draw_graph_weights(target, graph, points, mst_edges);
  draw_graph_vertices(target, graph, points);
}

/// Pulisce lo schermo e riporta il cursore in alto a sinistra, prima di ridisegnare.
inline void clear_terminal(std::ostream &out) {
  out << clear_screen << cursor_home;
}

/// Punto d'ingresso: se il grafo è abbastanza piccolo calcola layout,
/// proiezione e disegno con l'MST in evidenza; oltre `max_rendered_vertices`/`max_rendered_edges`
/// si limita a un riepilogo numerico (l'ASCII diventerebbe illeggibile e costoso).
inline void
render_graph_with_mst(const mst::core::validated_graph &graph,
                      const std::vector<mst::core::mst_edge> &mst_edges,
                      int total_weight, std::ostream &out = std::cout) {
  constexpr int max_rendered_vertices = 128;
  constexpr std::size_t max_rendered_edges = 1024;
  if (graph.vertex_count() > max_rendered_vertices ||
      graph.edges().size() > max_rendered_edges) {
    out << "Graph visualization skipped for large graph: vertices = "
        << graph.vertex_count() << ", edges = " << graph.edges().size()
        << ", MST weight = " << total_weight
        << ", MST edges = " << mst_edges.size() << '\n';
    return;
  }

  const viewport_size terminal = detect_viewport_size();
  const int width = std::clamp(terminal.width - 2, 96, 180);
  const int height = std::clamp(terminal.height - 8, 28, 60);
  const std::vector<point> layout = make_force_layout(graph);
  const std::vector<canvas_point> points =
      project_layout(layout, width, height);
  canvas target(width, height);

  clear_terminal(out);
  out << "Graph overlay with MST highlighted\n";
  out << "Legend: " << muted_blue << edge_glyph << " non-MST edges" << reset
      << ", " << mst_green << bold << edge_glyph << " MST edges" << reset
      << ", " << yellow << bold << "padded labels" << reset << ", "
      << muted_gray << "edge weights" << reset << '\n';
  out << "MST weight = " << total_weight << ", edges = " << mst_edges.size()
      << "\n\n";

  draw_graph_content(target, graph, points, mst_edges);
  target.render(out);
  out << "\nNodes are laid out with a deterministic force-directed layout.\n";
  out << reset << cursor_show;
}

} // namespace mst::visualization
