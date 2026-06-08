// boruvka_mpi.cpp
//
// Distributed Borůvka Minimum Spanning Tree (MST) using MPI.
//
// Parallelization strategy (textbook distributed Borůvka):
//   * Edges are partitioned across processes in contiguous 1D blocks.
//   * The component-label array `comp[0..V)` is REPLICATED on every rank.
//   * Each Borůvka phase:
//       1. Every rank scans its LOCAL edge slice and computes, for each
//          component it touches, the lightest outgoing edge.            O(E/p)
//       2. An MPI_Allreduce with a custom min-operator combines these
//          per-component candidates into the GLOBAL lightest edge.      O(V log p)
//       3. Every rank applies the SAME deterministic union-find merge
//          and updates `comp[]`.                                        O(V α(V))
//   * Because step 2 is an Allreduce, every rank ends each phase with an
//     identical view of the chosen edges and the new components, so the
//     full MST is computed redundantly-consistently on all ranks (no gather).
//
// Number of phases is O(log V): the component count at least halves per phase.
// Total work:  O((E/p + V) log V)  with  O(V log p log V)  communication.
//
// Cycle-freeness: each component picks its UNIQUE lightest outgoing edge under
// a strict total order (weight, then global edge id). A standard exchange
// argument shows the selected edge set is then always a forest. Two distinct
// edges with equal weight connecting the same pair of components are still
// disambiguated by id; the redundant one is dropped by the union-find check.
//
// Build:  mpic++ -O3 -std=c++17 boruvka_mpi.cpp -o boruvka_mpi
// Run:    mpirun -np 4 ./boruvka_mpi [graph_file]
//
// Graph file format (1 header line + E edge lines):
//   V E
//   u v w        // 0-based vertex ids, w is the (real) weight
//
// With no file argument a random connected graph is generated for testing.

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// --------------------------------------------------------------------------
// Types
// --------------------------------------------------------------------------

using Weight = double;                       // change to int -> use MPI_INT below
static const Weight INF_W = std::numeric_limits<Weight>::infinity();

struct Edge {
    int    u, v;
    Weight w;
};

// Per-component candidate edge exchanged in the Allreduce.
// `gid` (global edge id) serves both as the strict-total-order tie-breaker
// and as the dedup key for edges chosen by both endpoints.
struct CandEdge {
    Weight weight;   // weight of the candidate edge
    int    gid;      // global edge id
    int    u, v;     // original endpoints (kept so MST output needs no lookup)
};

static const CandEdge EMPTY_CAND{INF_W, INT_MAX, -1, -1};

// Strict total order:  a "better" (smaller) than b ?
static inline bool better(const CandEdge& a, const CandEdge& b) {
    return a.weight < b.weight || (a.weight == b.weight && a.gid < b.gid);
}

// --------------------------------------------------------------------------
// MPI datatype / operator helpers
// --------------------------------------------------------------------------

static MPI_Datatype make_struct_type(int n, const MPI_Datatype* types,
                                     const MPI_Aint* disp, MPI_Aint extent) {
    std::vector<int> bl(n, 1);
    MPI_Datatype tmp, out;
    MPI_Type_create_struct(n, bl.data(), disp, types, &tmp);
    MPI_Type_create_resized(tmp, 0, extent, &out);
    MPI_Type_commit(&out);
    MPI_Type_free(&tmp);
    return out;
}

static MPI_Datatype make_edge_type() {
    Edge t;
    MPI_Aint base, d[3];
    MPI_Get_address(&t,   &base);
    MPI_Get_address(&t.u, &d[0]);
    MPI_Get_address(&t.v, &d[1]);
    MPI_Get_address(&t.w, &d[2]);
    for (auto& x : d) x -= base;
    MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};   // Weight == double
    return make_struct_type(3, types, d, sizeof(Edge));
}

static MPI_Datatype make_candedge_type() {
    CandEdge t;
    MPI_Aint base, d[4];
    MPI_Get_address(&t,        &base);
    MPI_Get_address(&t.weight, &d[0]);
    MPI_Get_address(&t.gid,    &d[1]);
    MPI_Get_address(&t.u,      &d[2]);
    MPI_Get_address(&t.v,      &d[3]);
    for (auto& x : d) x -= base;
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_INT, MPI_INT, MPI_INT}; // Weight==double
    return make_struct_type(4, types, d, sizeof(CandEdge));
}

// Custom reduction: element-wise min under `better`.
static void cand_min(void* in, void* inout, int* len, MPI_Datatype*) {
    const CandEdge* a = static_cast<const CandEdge*>(in);
    CandEdge*       b = static_cast<CandEdge*>(inout);
    for (int i = 0; i < *len; ++i)
        if (better(a[i], b[i])) b[i] = a[i];
}

// --------------------------------------------------------------------------
// Graph input
// --------------------------------------------------------------------------

static bool read_graph(const char* path, int& V, std::vector<Edge>& edges) {
    std::ifstream in(path);
    if (!in) return false;
    int E;
    in >> V >> E;
    edges.resize(E);
    for (int i = 0; i < E; ++i) in >> edges[i].u >> edges[i].v >> edges[i].w;
    return true;
}

static void generate_graph(int V, int extra, std::vector<Edge>& edges,
                           unsigned seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> wd(1.0, 100.0);
    // random spanning tree -> guarantees connectivity
    for (int i = 1; i < V; ++i) {
        std::uniform_int_distribution<int> pd(0, i - 1);
        edges.push_back({i, pd(rng), std::round(wd(rng))});
    }
    // extra random edges
    std::uniform_int_distribution<int> vd(0, V - 1);
    for (int k = 0; k < extra; ++k) {
        int a = vd(rng), b = vd(rng);
        if (a != b) edges.push_back({a, b, std::round(wd(rng))});
    }
}

// --------------------------------------------------------------------------
// Main
// --------------------------------------------------------------------------

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Datatype edge_type = make_edge_type();
    MPI_Datatype cand_type = make_candedge_type();
    MPI_Op       cand_op;
    MPI_Op_create(&cand_min, /*commute=*/1, &cand_op);

    // ---- rank 0 builds the graph -----------------------------------------
    int V = 0, E = 0;
    std::vector<Edge> edges;          // only meaningful on rank 0
    if (rank == 0) {
        if (argc > 1) {
            if (!read_graph(argv[1], V, edges)) {
                std::cerr << "Cannot read graph file: " << argv[1] << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        } else {
            V = 20000;
            generate_graph(V, 8 * V, edges);   // ~8 edges/vertex
        }
        E = static_cast<int>(edges.size());
        std::cout << "Graph: V=" << V << " E=" << E
                  << "  processes=" << size << "\n";
    }
    MPI_Bcast(&V, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // ---- distribute edges in contiguous blocks ---------------------------
    std::vector<int> counts(size), displs(size);
    int q = E / size, r = E % size, off = 0;
    for (int i = 0; i < size; ++i) {
        counts[i] = q + (i < r ? 1 : 0);
        displs[i] = off;
        off += counts[i];
    }
    int local_n   = counts[rank];
    int my_offset = displs[rank];                 // global id of local edge 0
    std::vector<Edge> local(local_n);
    MPI_Scatterv(rank == 0 ? edges.data() : nullptr,
                 counts.data(), displs.data(), edge_type,
                 local.data(), local_n, edge_type, 0, MPI_COMM_WORLD);

    // ---- Borůvka phases --------------------------------------------------
    std::vector<int> comp{V};                       // comp[v] = root of v
    std::iota(comp.begin(), comp.end(), 0);

    std::vector<CandEdge> best{V}, link_buf;
    std::vector<int>      link{V};
    std::vector<CandEdge> mst;                      // chosen MST edges
    double mst_weight = 0.0;
    int    phases     = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    while (true) {
        // 1. local lightest outgoing edge per component
        std::fill(best.begin(), best.end(), EMPTY_CAND);
        for (int i = 0; i < local_n; ++i) {
            const Edge& e = local[i];
            int cu = comp[e.u], cv = comp[e.v];
            if (cu == cv) continue;                 // internal edge
            CandEdge c{e.w, my_offset + i, e.u, e.v};
            if (better(c, best[cu])) best[cu] = c;
            if (better(c, best[cv])) best[cv] = c;
        }

        // 2. global lightest outgoing edge per component
        MPI_Allreduce(MPI_IN_PLACE, best.data(), V, cand_type,
                      cand_op, MPI_COMM_WORLD);

        // 3. deterministic union-find merge (identical on every rank)
        std::iota(link.begin(), link.end(), 0);
        auto find = [&](int x) {
            while (link[x] != x) { link[x] = link[link[x]]; x = link[x]; }
            return x;
        };

        // collect chosen edges (roots only) and dedup by global id
        std::vector<CandEdge> chosen;
        chosen.reserve(64);
        for (int c = 0; c < V; ++c)
            if (comp[c] == c && best[c].gid != -1) chosen.push_back(best[c]);
        std::sort(chosen.begin(), chosen.end(),
                  [](const CandEdge& a, const CandEdge& b) { return a.gid < b.gid; });
        chosen.erase(std::unique(chosen.begin(), chosen.end(),
                  [](const CandEdge& a, const CandEdge& b) { return a.gid == b.gid; }),
                  chosen.end());

        bool added = false;
        for (const CandEdge& ce : chosen) {
            int ru = find(comp[ce.u]), rv = find(comp[ce.v]);
            if (ru == rv) continue;                 // would close a cycle -> skip
            if (ru < rv) link[rv] = ru; else link[ru] = rv;   // union toward min
            mst.push_back(ce);
            mst_weight += ce.weight;
            added = true;
        }
        if (!added) break;                          // done (1 component or disconnected)

        // relabel every vertex to its new component root
        for (int v = 0; v < V; ++v) comp[v] = find(comp[v]);
        ++phases;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "MST edges : " << mst.size()  << "\n";
        std::cout << "MST weight: " << mst_weight   << "\n";
        std::cout << "Phases    : " << phases        << "\n";
        std::cout << "Time      : " << (t1 - t0)     << " s\n";
        // Uncomment to dump the tree:
        // for (const auto& e : mst)
        //     std::cout << e.u << ' ' << e.v << ' ' << e.weight << '\n';
    }

    MPI_Op_free(&cand_op);
    MPI_Type_free(&edge_type);
    MPI_Type_free(&cand_type);
    MPI_Finalize();
    return 0;
}
