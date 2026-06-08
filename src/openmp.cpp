// boruvka_omp.cpp
// Algoritmo MST di Borůvka parallelizzato con OpenMP.
//
// Build:  g++ -O3 -std=c++17 -fopenmp boruvka_omp.cpp -o boruvka_omp
// Run:    OMP_NUM_THREADS=8 ./boruvka_omp
//
// Assunzioni: pesi interi non negativi che entrano in 32 bit, e numero di
// archi < 2^32. Vedi note nella scelta della chiave impacchettata.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <algorithm>
#include <chrono>
#include <omp.h>

struct Edge {
    int u, v;
    int w;
};

// --- Union-Find (Disjoint Set Union) -------------------------------------
// Il lavoro costoso O(E) (la scansione degli archi) è parallelizzato; il
// passo di unione è O(V) per round e viene eseguito in modo seriale, così la
// logica di merge resta semplice e priva di race condition.
struct DSU {
    std::vector<int> parent, rank_;
    explicit DSU(int n) : parent(n), rank_(n, 0) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }
    // find di sola lettura: segue i puntatori al padre SENZA path compression,
    // quindi è sicuro chiamarlo concorrentemente da più thread (nessuna
    // scrittura su parent[]). Con union-by-rank l'altezza resta O(log V).
    int find(int x) const {
        while (parent[x] != x) x = parent[x];
        return x;
    }
    // union-by-rank seriale. Ritorna true se è avvenuta una fusione.
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) std::swap(a, b);
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }
};

// Atomic-min lock-free su un intero a 64 bit (loop di compare-and-swap).
// È l'equivalente OpenMP dell'atomicMin usato sulla GPU.
static inline void atomic_min_u64(std::atomic<uint64_t>& tgt, uint64_t val) {
    uint64_t cur = tgt.load(std::memory_order_relaxed);
    // compare_exchange_weak ricarica `cur` automaticamente quando fallisce.
    while (val < cur &&
           !tgt.compare_exchange_weak(cur, val, std::memory_order_relaxed)) {
    }
}

// Ritorna il peso totale dell'MST; riempie `mst` con gli indici degli archi.
long long boruvkaMST(int n, const std::vector<Edge>& edges,
                     std::vector<int>& mst) {
    DSU dsu(n);
    mst.clear();
    long long total = 0;
    int numComp = n;

    const int m = (int)edges.size();
    std::vector<int> comp(n);

    // Uno slot per indice di vertice: una componente è identificata dall'id
    // del suo vertice radice, quindi n slot bastano.
    auto cheapest = std::make_unique<std::atomic<uint64_t>[]>(n);
    const uint64_t NONE = std::numeric_limits<uint64_t>::max();

    while (numComp > 1) {
        // 1) Snapshot della componente (radice) di ogni vertice. find di sola
        //    lettura -> eseguibile in parallelo senza race.
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < n; ++v) comp[v] = dsu.find(v);

        // 2) Reset degli slot "arco più leggero".
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < n; ++c)
            cheapest[c].store(NONE, std::memory_order_relaxed);

        // 3) Scansione parallela di tutti gli archi: per ogni componente trova
        //    l'arco uscente di peso minimo. Impacchettiamo (peso, indice) in
        //    una sola chiave a 64 bit, così un singolo atomic-min sceglie
        //    l'arco più leggero e rompe i pareggi a favore dell'indice più
        //    piccolo. Questo tie-break coerente è ciò che impedisce a due
        //    componenti di scegliere archi diversi e formare un ciclo.
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < m; ++i) {
            int cu = comp[edges[i].u];
            int cv = comp[edges[i].v];
            if (cu == cv) continue;            // arco interno, si ignora
            uint64_t key = ((uint64_t)(uint32_t)edges[i].w << 32)
                         | (uint32_t)i;
            atomic_min_u64(cheapest[cu], key);
            atomic_min_u64(cheapest[cv], key);
        }

        // 4) Passo di merge seriale. Al più un arco per componente, quindi è
        //    O(V), trascurabile rispetto alla scansione O(E). Il guard
        //    find()==find() scarta un arco già fuso in precedenza nello stesso
        //    round (il classico caso arco-duplicato / ciclo di Borůvka).
        bool progress = false;
        for (int c = 0; c < n; ++c) {
            uint64_t key = cheapest[c].load(std::memory_order_relaxed);
            if (key == NONE) continue;
            int ei = (int)(uint32_t)(key & 0xFFFFFFFFu);
            const Edge& e = edges[ei];
            if (dsu.unite(e.u, e.v)) {
                total += e.w;
                mst.push_back(ei);
                --numComp;
                progress = true;
            }
        }

        // 5) Se un intero round non aggiunge nulla, il grafo è disconnesso.
        if (!progress) break;
    }
    return total;
}

// Riferimento seriale (Kruskal) usato solo per validare il risultato.
long long kruskalMST(int n, std::vector<Edge> edges) {
    std::sort(edges.begin(), edges.end(),
              [](const Edge& a, const Edge& b) { return a.w < b.w; });
    DSU dsu(n);
    long long total = 0; int cnt = 0;
    for (const auto& e : edges)
        if (dsu.unite(e.u, e.v)) { total += e.w; if (++cnt == n - 1) break; }
    return total;
}

int main() {
    // ---- Grafo connesso casuale --------------------------------------
    const int n = 200000;
    const long long extraEdges = 2000000;
    std::mt19937 rng(12345);

    std::vector<Edge> edges;
    edges.reserve(n - 1 + extraEdges);

    // Spanning tree casuale -> garantisce la connessione.
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    std::uniform_int_distribution<int> wdist(1, 1000);
    for (int i = 1; i < n; ++i) {
        int parent = perm[rng() % i];
        edges.push_back({perm[i], parent, wdist(rng)});
    }
    // Archi extra casuali.
    std::uniform_int_distribution<int> vdist(0, n - 1);
    for (long long e = 0; e < extraEdges; ++e) {
        int u = vdist(rng), v = vdist(rng);
        if (u != v) edges.push_back({u, v, wdist(rng)});
    }

    printf("Grafo: %d vertici, %zu archi, %d thread\n",
           n, edges.size(), omp_get_max_threads());

    // ---- Borůvka parallelo -------------------------------------------
    std::vector<int> mst;
    auto t0 = std::chrono::high_resolution_clock::now();
    long long w = boruvkaMST(n, edges, mst);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Boruvka (OpenMP): peso=%lld, archi=%zu, %.1f ms\n",
           w, mst.size(), ms);

    // ---- Verifica con Kruskal ----------------------------------------
    long long wk = kruskalMST(n, edges);
    printf("Kruskal (seriale): peso=%lld  -> %s\n",
           wk, (wk == w ? "MATCH" : "MISMATCH"));
    return 0;
}
