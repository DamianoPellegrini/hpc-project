// boruvka_cuda.cu
//
// Parallel Boruvka Minimum Spanning Tree (forest) on the GPU, in CUDA.
//
// Graph representation: undirected weighted graph as an edge list
//   (src[e], dst[e], w[e]).
//
// Per Boruvka round (all steps are data-parallel kernels):
//   1. Each component finds its minimum-weight OUTGOING edge using a packed
//      64-bit atomicMin: the float weight is mapped to an order-preserving
//      uint32 and placed in the high 32 bits, the edge index in the low 32.
//      The edge index acts as a deterministic tie-breaker, so the packed key
//      gives a TOTAL order on edges -> only length-2 conflict cycles can form.
//   2. Build a per-component "successor" link (which component it merges into).
//   3. Break the only possible cycles (length 2) deterministically: in a
//      mutual pair the smaller id becomes the root, the larger one contributes
//      the connecting edge (added exactly once).
//   4. Mark the selected edges as MST edges.
//   5. Pointer-jump the successor array down to roots (double-buffered), then
//      relabel every vertex to its new component root.
// Repeat until a round adds no edge -> handles disconnected graphs and yields
// a minimum spanning FOREST. Connected input gives V-1 edges.
//
// Work per round O(E), rounds O(log V)  =>  total O(E log V).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// ----------------------------------------------------------------------------
// Error checking
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                               \
  do {                                                                                 \
    cudaError_t _e = (call);                                                           \
    if (_e != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                    \
              cudaGetErrorString(_e));                                                 \
      std::exit(EXIT_FAILURE);                                                         \
    }                                                                                  \
  } while (0)

typedef unsigned long long u64; // matches atomicMin(unsigned long long*)
typedef unsigned int u32;

static const u64 KEY_SENTINEL = ~0ull; // "no edge"

// ----------------------------------------------------------------------------
// Key packing: (orderable_weight << 32) | edge_index
// ----------------------------------------------------------------------------

// Map an IEEE-754 float to an order-preserving uint32 (assumes finite, no NaN).
__host__ __device__ inline u32 float_to_orderable(float f) {
  u32 u;
#ifdef __CUDA_ARCH__
  u = __float_as_uint(f);
#else
  std::memcpy(&u, &f, sizeof(u));
#endif
  // Flip sign bit for positives, flip all bits for negatives.
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__host__ __device__ inline u64 pack_key(float w, u32 edge_idx) {
  return (static_cast<u64>(float_to_orderable(w)) << 32) | static_cast<u64>(edge_idx);
}

__host__ __device__ inline u32 unpack_idx(u64 key) {
  return static_cast<u32>(key & 0xFFFFFFFFu);
}

// ----------------------------------------------------------------------------
// Kernels
// ----------------------------------------------------------------------------

__global__ void k_reset_min(u64* min_edge, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    min_edge[i] = KEY_SENTINEL;
}

// For each edge, atomicMin its packed key into BOTH endpoint components.
__global__ void k_find_min_edges(const int* src, const int* dst, const float* w,
                                 const int* comp, u64* min_edge, int m) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= m)
    return;
  int cu = comp[src[e]];
  int cv = comp[dst[e]];
  if (cu == cv)
    return; // internal edge: ignore
  u64 key = pack_key(w[e], static_cast<u32>(e));
  atomicMin(&min_edge[cu], key);
  atomicMin(&min_edge[cv], key);
}

// successor[c] = the OTHER component reached by c's chosen minimum edge.
// Non-root / isolated components point to themselves.
__global__ void k_build_successor(const int* src, const int* dst, const int* comp,
                                  const u64* min_edge, int* successor, int n) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n)
    return;
  u64 key = min_edge[c];
  if (key == KEY_SENTINEL) {
    successor[c] = c;
    return;
  }
  int e = unpack_idx(key);
  int cu = comp[src[e]];
  int cv = comp[dst[e]];
  successor[c] = (cu == c) ? cv : cu;
}

// Break length-2 cycles and mark the MST edge each component contributes.
// Reads ONLY from `succ` (immutable here); writes to nsucc / in_mst / added.
__global__ void k_mark_and_break(const u64* min_edge, const int* succ, int* nsucc,
                                 char* in_mst, int* added, int n) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n)
    return;
  int s = succ[c];
  nsucc[c] = s; // default: keep the link
  if (s == c)
    return;                    // root / isolated: nothing to add
  bool cycle = (succ[s] == c); // c -> s -> c
  if (cycle && c < s) {
    nsucc[c] = c; // smaller id becomes the root
    return;       // larger id will add the shared edge
  }
  int e = unpack_idx(min_edge[c]); // each tree edge added exactly once
  in_mst[e] = 1;
  atomicAdd(added, 1);
}

// One double-buffered pointer-jumping step: out[c] = in[in[c]].
__global__ void k_jump(const int* in, int* out, char* changed, int n) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n)
    return;
  int s = in[c];
  int ss = in[s];
  out[c] = ss;
  if (ss != s)
    *changed = 1; // not yet fully flattened
}

// Map every vertex to its new component root.
__global__ void k_relabel(int* comp, const int* root, int nv) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= nv)
    return;
  comp[v] = root[comp[v]];
}

__global__ void k_iota(int* a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = i;
}

// ----------------------------------------------------------------------------
// Host driver
// ----------------------------------------------------------------------------
static inline int grid(int n, int block) {
  return (n + block - 1) / block;
}

struct GpuResult {
  std::vector<char> in_mst;
  int rounds;
  float exec_ms;
};

GpuResult boruvka_gpu(int nv, const std::vector<int>& h_src,
                      const std::vector<int>& h_dst, const std::vector<float>& h_w) {
  const int m = static_cast<int>(h_src.size());

  // Block size adattata alla macchina: l'occupancy calculator di CUDA
  // sceglie, per il kernel più pesante (la scansione O(E) degli archi, che
  // domina ogni round), la dimensione di blocco che massimizza l'occupazione
  // degli SM sulla GPU effettivamente in uso. La riusiamo per tutti i kernel:
  // `grid(n, B)` spezzetta già la griglia sulla dimensione reale del problema.
  int min_grid_size = 0, B = 256;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &B, k_find_min_edges));

  int *d_src, *d_dst, *d_comp, *d_succ, *d_nsucc, *d_root, *d_added;
  float* d_w;
  u64* d_min;
  char *d_in_mst, *d_changed;

  CUDA_CHECK(cudaMalloc(&d_src, m * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_dst, m * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_w, m * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_comp, nv * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_succ, nv * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_nsucc, nv * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_root, nv * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_min, nv * sizeof(u64)));
  CUDA_CHECK(cudaMalloc(&d_in_mst, m * sizeof(char)));
  CUDA_CHECK(cudaMalloc(&d_added, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_changed, sizeof(char)));

  CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), m * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dst, h_dst.data(), m * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), m * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_in_mst, 0, m * sizeof(char)));

  k_iota<<<grid(nv, B), B>>>(d_comp, nv); // comp[v] = v

  // Cronometro interno: SOLO il loop dell'algoritmo (esecuzione pura),
  // alloc/copie restano fuori e contribuiscono all'overhead esterno.
  cudaEvent_t te0, te1;
  CUDA_CHECK(cudaEventCreate(&te0));
  CUDA_CHECK(cudaEventCreate(&te1));
  CUDA_CHECK(cudaEventRecord(te0));

  int rounds = 0;
  while (true) {
    ++rounds;
    k_reset_min<<<grid(nv, B), B>>>(d_min, nv);
    k_find_min_edges<<<grid(m, B), B>>>(d_src, d_dst, d_w, d_comp, d_min, m);
    k_build_successor<<<grid(nv, B), B>>>(d_src, d_dst, d_comp, d_min, d_succ, nv);

    CUDA_CHECK(cudaMemset(d_added, 0, sizeof(int)));
    k_mark_and_break<<<grid(nv, B), B>>>(d_min, d_succ, d_nsucc, d_in_mst, d_added, nv);
    int added = 0;
    CUDA_CHECK(cudaMemcpy(&added, d_added, sizeof(int), cudaMemcpyDeviceToHost));
    if (added == 0)
      break; // no inter-component edge left

    // Pointer-jump nsucc -> roots (double buffered: d_nsucc <-> d_root).
    int *cur = d_nsucc, *nxt = d_root;
    while (true) {
      CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(char)));
      k_jump<<<grid(nv, B), B>>>(cur, nxt, d_changed, nv);
      char ch = 0;
      CUDA_CHECK(cudaMemcpy(&ch, d_changed, sizeof(char), cudaMemcpyDeviceToHost));
      std::swap(cur, nxt);
      if (!ch)
        break;
    }
    k_relabel<<<grid(nv, B), B>>>(d_comp, cur, nv);
  }

  CUDA_CHECK(cudaEventRecord(te1));
  CUDA_CHECK(cudaEventSynchronize(te1));
  float exec_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&exec_ms, te0, te1));
  cudaEventDestroy(te0);
  cudaEventDestroy(te1);

  GpuResult res;
  res.in_mst.resize(m);
  res.rounds = rounds;
  res.exec_ms = exec_ms;
  CUDA_CHECK(cudaMemcpy(res.in_mst.data(), d_in_mst, m * sizeof(char),
                        cudaMemcpyDeviceToHost));

  cudaFree(d_src);
  cudaFree(d_dst);
  cudaFree(d_w);
  cudaFree(d_comp);
  cudaFree(d_succ);
  cudaFree(d_nsucc);
  cudaFree(d_root);
  cudaFree(d_min);
  cudaFree(d_in_mst);
  cudaFree(d_added);
  cudaFree(d_changed);
  return res;
}

// ----------------------------------------------------------------------------
// CPU reference (Kruskal + union-find) for verification
// ----------------------------------------------------------------------------
struct DSU {
  std::vector<int> p, r;
  explicit DSU(int n) : p(n), r(n, 0) {
    std::iota(p.begin(), p.end(), 0);
  }
  int find(int x) {
    while (p[x] != x) {
      p[x] = p[p[x]];
      x = p[x];
    }
    return x;
  }
  bool unite(int a, int b) {
    a = find(a);
    b = find(b);
    if (a == b)
      return false;
    if (r[a] < r[b])
      std::swap(a, b);
    p[b] = a;
    if (r[a] == r[b])
      ++r[a];
    return true;
  }
};

double kruskal_weight(int nv, const std::vector<int>& s, const std::vector<int>& d,
                      const std::vector<float>& w, int& edges_out) {
  int m = static_cast<int>(s.size());
  std::vector<int> idx(m);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return w[a] < w[b];
  });
  DSU dsu(nv);
  double total = 0.0;
  int e = 0;
  for (int i : idx)
    if (dsu.unite(s[i], d[i])) {
      total += w[i];
      ++e;
    }
  edges_out = e;
  return total;
}

// ----------------------------------------------------------------------------
// Random CONNECTED graph generator (random spanning tree + archi casuali fino
// a raggiungere esattamente target_m archi)
// ----------------------------------------------------------------------------
void gen_graph(int nv, int target_m, unsigned seed, std::vector<int>& s,
               std::vector<int>& d, std::vector<float>& w) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> wd(0.0f, 1.0f);
  std::vector<int> perm(nv);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);

  s.clear();
  d.clear();
  w.clear();
  for (int i = 1; i < nv; ++i) { // spanning tree => connected
    std::uniform_int_distribution<int> pick(0, i - 1);
    s.push_back(perm[i]);
    d.push_back(perm[pick(rng)]);
    w.push_back(wd(rng));
  }
  std::uniform_int_distribution<int> vrand(0, nv - 1);
  while (static_cast<int>(s.size()) < target_m) {
    int a = vrand(rng), b = vrand(rng);
    if (a == b)
      continue;
    s.push_back(a);
    d.push_back(b);
    w.push_back(wd(rng));
  }
}

// Uso: boruvka_cuda <n> <edges> <seed>
int main(int argc, char** argv) {
  int nv = (argc > 1) ? std::atoi(argv[1]) : 100000;
  int m = (argc > 2) ? std::atoi(argv[2]) : 900000;
  unsigned sd = (argc > 3) ? static_cast<unsigned>(std::atoi(argv[3])) : 42u;

  // ---- overhead: generazione grafo + alloc/copie H2D/D2H -----------
  auto to0 = std::chrono::high_resolution_clock::now();

  std::vector<int> s, d;
  std::vector<float> w;
  gen_graph(nv, m, sd, s, d, w);
  printf("Graph: V=%d  E=%d  seed=%u\n", nv, (int)s.size(), sd);

  cudaEvent_t t0, t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));
  GpuResult r = boruvka_gpu(nv, s, d, w);
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float total_ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);

  auto to1 = std::chrono::high_resolution_clock::now();

  double gpu_w = 0.0;
  int gpu_e = 0;
  for (size_t i = 0; i < r.in_mst.size(); ++i)
    if (r.in_mst[i]) {
      gpu_w += w[i];
      ++gpu_e;
    }

  // ---- verifica sequenziale (Kruskal lato host, cronometrata a parte)
  auto tv0 = std::chrono::high_resolution_clock::now();
  int cpu_e = 0;
  double cpu_w = kruskal_weight(nv, s, d, w, cpu_e);
  auto tv1 = std::chrono::high_resolution_clock::now();

  printf("GPU Boruvka : weight=%.6f  edges=%d  rounds=%d\n", gpu_w, gpu_e, r.rounds);
  printf("CPU Kruskal : weight=%.6f  edges=%d\n", cpu_w, cpu_e);
  bool ok = (gpu_e == cpu_e) && (std::abs(gpu_w - cpu_w) < 1e-3 * (1.0 + cpu_w));
  const char* verification = ok ? "PASS" : "FAIL";
  printf("Verification: %s\n", verification);

  // overhead = tempo totale fuori dal loop puro (gen grafo + alloc/copie H2D/D2H)
  double overhead_seconds =
      std::chrono::duration<double>(to1 - to0).count() - (double)r.exec_ms / 1000.0;
  double exec_seconds = (double)r.exec_ms / 1000.0;
  double verify_seconds = std::chrono::duration<double>(tv1 - tv0).count();
  double total_seconds = overhead_seconds + exec_seconds;

  printf("verification=%s\n", verification);
  printf("overhead_seconds=%.6f\n", overhead_seconds);
  printf("exec_seconds=%.6f\n", exec_seconds);
  printf("total_seconds=%.6f\n", total_seconds);
  printf("verify_seconds=%.6f\n", verify_seconds);
  return ok ? 0 : 1;
}
