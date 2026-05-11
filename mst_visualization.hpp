#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mst_common.hpp"

namespace mst::viz {

// ANSI helpers keep the visualization self-contained and easy to reset after
// the canvas is printed.
inline constexpr const char *kClearScreen = "\x1b[2J";
inline constexpr const char *kCursorHome = "\x1b[H";
inline constexpr const char *kCursorShow = "\x1b[?25h";
inline constexpr const char *kReset = "\x1b[0m";
inline constexpr const char *kDim = "\x1b[2m";
inline constexpr const char *kBold = "\x1b[1m";
inline constexpr const char *kGreen = "\x1b[32m";
inline constexpr const char *kCyan = "\x1b[36m";
inline constexpr const char *kYellow = "\x1b[33m";
inline constexpr const char *kWhite = "\x1b[37m";

struct Point {
  int x;
  int y;
};

struct Cell {
  char ch = ' ';
  const char *color = kReset;
  bool bold = false;
};

class Canvas {
public:
  Canvas(int width, int height)
      : width_(width), height_(height), cells_(static_cast<std::size_t>(width * height)) {}

  int width() const { return width_; }
  int height() const { return height_; }

  void set(int x, int y, char ch, const char *color = kWhite, bool bold = false) {
    if (x < 0 || y < 0 || x >= width_ || y >= height_) {
      return;
    }

    Cell &cell = cells_[index(x, y)];
    cell.ch = ch;
    cell.color = color;
    cell.bold = bold;
  }

  void draw_text(int x, int y, const std::string &text, const char *color = kWhite,
                 bool bold = false) {
    for (std::size_t offset = 0; offset < text.size(); ++offset) {
      set(x + static_cast<int>(offset), y, text[offset], color, bold);
    }
  }

  void draw_line(int x0, int y0, int x1, int y1, char ch, const char *color, bool bold = false) {
    const int dx = std::abs(x1 - x0);
    const int sx = x0 < x1 ? 1 : -1;
    const int dy = -std::abs(y1 - y0);
    const int sy = y0 < y1 ? 1 : -1;
    int error = dx + dy;

    while (true) {
      set(x0, y0, ch, color, bold);
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

  void render(std::ostream &out) const {
    const char *current_color = kReset;
    bool current_bold = false;

    auto emit_style = [&](const Cell &cell) {
      if (cell.color == current_color && cell.bold == current_bold) {
        return;
      }

      out << kReset;
      if (cell.color != kReset) {
        out << cell.color;
      }
      if (cell.bold) {
        out << kBold;
      }
      current_color = cell.color;
      current_bold = cell.bold;
    };

    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        const Cell &cell = cells_[index(x, y)];
        emit_style(cell);
        out << cell.ch;
      }
      out << kReset << '\n';
      current_color = kReset;
      current_bold = false;
    }
    out << kReset;
  }

private:
  int index(int x, int y) const { return y * width_ + x; }

  int width_;
  int height_;
  std::vector<Cell> cells_;
};

inline std::vector<Point> make_circular_layout(int vertex_count, int width, int height) {
  const double pi = 3.14159265358979323846;
  const double center_x = static_cast<double>(width - 1) / 2.0;
  const double center_y = static_cast<double>(height - 1) / 2.0;
  const double radius_x = static_cast<double>(width) * 0.34;
  const double radius_y = static_cast<double>(height) * 0.34;

  std::vector<Point> points;
  points.reserve(static_cast<std::size_t>(vertex_count));
  for (int vertex = 0; vertex < vertex_count; ++vertex) {
    const double angle = (2.0 * pi * vertex) / static_cast<double>(vertex_count);
    const int x = static_cast<int>(std::round(center_x + radius_x * std::cos(angle)));
    const int y = static_cast<int>(std::round(center_y + radius_y * std::sin(angle)));
    points.push_back({x, y});
  }
  return points;
}

inline bool same_undirected_edge(const Edge &edge, int u, int v) {
  return (index_of(edge.u) == u && index_of(edge.v) == v) ||
         (index_of(edge.u) == v && index_of(edge.v) == u);
}

inline bool same_undirected_edge(const Edge &lhs, const Edge &rhs) {
  return same_undirected_edge(lhs, index_of(rhs.u), index_of(rhs.v));
}

inline bool same_undirected_edge(const Candidate &candidate, const Edge &edge) {
  return candidate && same_undirected_edge(edge, index_of(candidate->u), index_of(candidate->v));
}

inline std::string node_label(int vertex) {
  return std::to_string(vertex);
}

inline void draw_graph_overlay(Canvas &canvas, const Graph &graph, const std::vector<Point> &points,
                               const std::vector<Edge> &mst_edges) {
  // Draw the full graph first in a faint style, then overwrite the MST edges
  // so the selected tree is visible on top of the original graph.
  for (const Edge &edge : graph.edges) {
    const Point &from = points[static_cast<std::size_t>(index_of(edge.u))];
    const Point &to = points[static_cast<std::size_t>(index_of(edge.v))];
    canvas.draw_line(from.x, from.y, to.x, to.y, '.', kDim);
  }

  for (const Edge &edge : mst_edges) {
    const Point &from = points[static_cast<std::size_t>(index_of(edge.u))];
    const Point &to = points[static_cast<std::size_t>(index_of(edge.v))];
    canvas.draw_line(from.x, from.y, to.x, to.y, '#', kGreen, true);
  }

  for (int vertex = 0; vertex < graph.vertex_count; ++vertex) {
    const Point &point = points[static_cast<std::size_t>(vertex)];
    const std::string label = node_label(vertex);
    const int start_x = point.x - static_cast<int>(label.size() / 2);
    canvas.set(point.x, point.y, 'o', kWhite, true);
    canvas.draw_text(start_x, point.y, label, kYellow, true);
  }
}

inline void clear_terminal(std::ostream &out) {
  out << kClearScreen << kCursorHome;
}

inline void render_graph_with_mst(const Graph &graph, const std::vector<Edge> &mst_edges,
                                  int total_weight, std::ostream &out = std::cout) {
  const int width = 96;
  const int height = 28;
  const std::vector<Point> points = make_circular_layout(graph.vertex_count, width, height);
  Canvas canvas(width, height);

  clear_terminal(out);
  out << "Graph overlay with MST highlighted\n";
  out << "Legend: " << kDim << ". full graph" << kReset << ", " << kGreen << kBold
      << "# MST edges" << kReset << ", " << kYellow << kBold << "labels" << kReset << '\n';
  out << "MST weight = " << total_weight << ", edges = " << mst_edges.size() << "\n\n";

  draw_graph_overlay(canvas, graph, points, mst_edges);
  canvas.render(out);
  out << "\nNodes are laid out on a fixed circle so the same graph renders consistently.\n";
  out << kReset << kCursorShow;
}

} // namespace mst::viz
