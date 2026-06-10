// boruvka_seq.cpp
// Algoritmo MST di Borůvka, versione sequenziale di riferimento.
//
// Stessa generazione del grafo (a parità di seed) e stesso schema di I/O di
// src/openmp.cpp, ma boruvkaMST è la versione seriale: stessa struttura a
// round (snapshot componenti, scan archi, merge), senza parallel for né
// atomici. Serve come baseline T_s per lo speedup misurato (Capitolo 3 del
// report).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

struct Edge {
  int u, v;
  int w;
};

struct DSU {
  std::vector<int> parent, rank_;
  explicit DSU(int n) : parent(n), rank_(n, 0) {
    for (int i = 0; i < n; ++i)
      parent[i] = i;
  }
  int find(int x) const {
    while (parent[x] != x)
      x = parent[x];
    return x;
  }
  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;
    if (rank_[a] < rank_[b])
      std::swap(a, b);
    parent[b] = a;
    if (rank_[a] == rank_[b])
      ++rank_[a];
    return true;
  }
};

// Borůvka seriale: O(log V) round, ciascuno O(|V|) per lo snapshot delle
// componenti, O(|E|) per la scan e O(|V|) per il merge -- la dinamica di
// @eq:seq-work, senza alcun parallelismo.
long long boruvkaMST(int n, const std::vector<Edge>& edges, std::vector<int>& mst) {
  DSU dsu(n);
  mst.clear();
  long long total = 0;
  int numComp = n;

  const int m = (int)edges.size();
  const long long NONE = -1;
  std::vector<int> comp(n);
  std::vector<long long> cheapest(n);

  while (numComp > 1) {
    // 1) Snapshot della componente (radice) di ogni vertice.
    for (int v = 0; v < n; ++v)
      comp[v] = dsu.find(v);

    // 2) Reset degli slot "arco più leggero".
    std::fill(cheapest.begin(), cheapest.end(), NONE);

    // 3) Scan seriale di tutti gli archi: per ogni componente trova l'arco
    //    uscente di peso minimo. Stesso impacchettamento (peso, indice) e
    //    stesso tie-break di src/openmp.cpp, qui senza atomic-min.
    for (int i = 0; i < m; ++i) {
      int cu = comp[edges[i].u];
      int cv = comp[edges[i].v];
      if (cu == cv)
        continue; // arco interno, si ignora
      long long key = ((long long)(uint32_t)edges[i].w << 32) | (uint32_t)i;
      if (cheapest[cu] == NONE || key < cheapest[cu])
        cheapest[cu] = key;
      if (cheapest[cv] == NONE || key < cheapest[cv])
        cheapest[cv] = key;
    }

    // 4) Merge: al più un arco per componente, O(|V|).
    bool progress = false;
    for (int c = 0; c < n; ++c) {
      if (cheapest[c] == NONE)
        continue;
      int ei = (int)(uint32_t)(cheapest[c] & 0xFFFFFFFFu);
      const Edge& e = edges[ei];
      if (dsu.unite(e.u, e.v)) {
        total += e.w;
        mst.push_back(ei);
        --numComp;
        progress = true;
      }
    }

    // 5) Se un intero round non aggiunge nulla, il grafo è disconnesso.
    if (!progress)
      break;
  }
  return total;
}

// Riferimento Kruskal, usato solo per validare il risultato (non cronometrato
// nell'esecuzione principale).
long long kruskalMST(int n, std::vector<Edge> edges) {
  std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
    return a.w < b.w;
  });
  DSU dsu(n);
  long long total = 0;
  int cnt = 0;
  for (const auto& e : edges)
    if (dsu.unite(e.u, e.v)) {
      total += e.w;
      if (++cnt == n - 1)
        break;
    }
  return total;
}

// Uso: sequential_app <n> <edges> <seed>
int main(int argc, char** argv) {
  const int n = (argc > 1) ? std::atoi(argv[1]) : 200000;
  const long long m = (argc > 2) ? std::atoll(argv[2]) : 2000000;
  const unsigned seed =
      (argc > 3) ? (unsigned)std::strtoul(argv[3], nullptr, 10) : 12345u;

  // ---- Grafo connesso casuale (overhead: setup) --------------------
  // Identica alla generazione di src/openmp.cpp, src/mpi.cpp e src/cuda.cu, a
  // parità di seed: stesso grafo, confrontabile arco per arco.
  auto to0 = std::chrono::high_resolution_clock::now();
  std::mt19937 rng(seed);
  std::vector<Edge> edges;
  edges.reserve(m);

  // Spanning tree casuale -> garantisce la connessione.
  std::vector<int> perm(n);
  for (int i = 0; i < n; ++i)
    perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  std::uniform_int_distribution<int> wdist(1, 1000);
  for (int i = 1; i < n; ++i) {
    int parent = perm[rng() % i];
    edges.push_back({perm[i], parent, wdist(rng)});
  }
  // Archi casuali finché il grafo non ha esattamente m archi.
  std::uniform_int_distribution<int> vdist(0, n - 1);
  while (static_cast<long long>(edges.size()) < m) {
    int u = vdist(rng), v = vdist(rng);
    if (u != v)
      edges.push_back({u, v, wdist(rng)});
  }
  auto to1 = std::chrono::high_resolution_clock::now();

  printf("Grafo: %d vertici, %zu archi (sequenziale)\n", n, edges.size());

  // ---- Borůvka seriale (esecuzione pura) -----------------------------
  std::vector<int> mst;
  auto te0 = std::chrono::high_resolution_clock::now();
  long long w = boruvkaMST(n, edges, mst);
  auto te1 = std::chrono::high_resolution_clock::now();
  printf("Boruvka (sequenziale): peso=%lld, archi=%zu\n", w, mst.size());

  // ---- Verifica con Kruskal (cronometrata a parte) --------------------
  auto tv0 = std::chrono::high_resolution_clock::now();
  long long wk = kruskalMST(n, edges);
  auto tv1 = std::chrono::high_resolution_clock::now();
  const char* verification = (wk == w) ? "MATCH" : "MISMATCH";
  printf("Kruskal (seriale): peso=%lld  -> %s\n", wk, verification);

  double overhead_seconds = std::chrono::duration<double>(to1 - to0).count();
  double exec_seconds = std::chrono::duration<double>(te1 - te0).count();
  double verify_seconds = std::chrono::duration<double>(tv1 - tv0).count();
  double total_seconds = overhead_seconds + exec_seconds;

  printf("verification=%s\n", verification);
  printf("overhead_seconds=%.6f\n", overhead_seconds);
  printf("exec_seconds=%.6f\n", exec_seconds);
  printf("total_seconds=%.6f\n", total_seconds);
  printf("verify_seconds=%.6f\n", verify_seconds);
  return 0;
}
