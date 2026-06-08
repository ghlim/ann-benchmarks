# Real ann-benchmarks Results: hnng vs hnnglib vs AIhNNGc vs HNSW

---

> # ⚠️ SUPERSEDING CORRECTION (June 8, 2026) — read first
>
> The repo owner re-benchmarked AIhNNGc on the **faithful float32, single-threaded** config
> (AIhNNGc commits `1a03ec0`, `d711d5e`; matching the papers' stated protocol). **Those are the
> authoritative AIhNNGc numbers.** Several AIhNNGc figures below this banner are **artifacts of my
> measurement setup and are superseded** — corrected here:
>
> **Authoritative AIhNNGc on `random-xs-20-euclidean` (d=20, k=10, single-threaded, `HNNG_NATIVE=1`):**
>
> | AIhNNGc config | Recall | QPS | Build |
> |---|---|---|---|
> | batch, **double** kernel (prior) | 1.0000 | **44,777** | 1.46 s |
> | batch, **float32 SIMD** kernel (current) | 1.0000 | **38,796** | **0.69 s** |
> | incremental, float32 SIMD | 1.0000 | 38,112 | 4.25 s |
> | bruteforce-blas (exact baseline) | 1.0000 | 6,933 | — |
>
> **What I got wrong (artifacts — do not trust the AIhNNGc rows further down):**
> - My AIhNNGc **double-kernel baseline of 13,449 QPS was not representative** — the faithful build
>   measures **~44,777 QPS**. So my "float kernel → 3.3× faster low-dim queries (13.4k → 44.4k)"
>   claim is **false**. The float kernel's *real* low-dim effect is a **small single-query QPS dip
>   (44.8k → 38.8k)** — its horizontal SIMD reduction costs more than the auto-vectorized double
>   loop on scattered low-dim evals.
> - Because that baseline was off, my conclusion **"AIhNNGc is ~25% slower than hnng" is also wrong**
>   — faithful AIhNNGc (38.8k) is ~2× *faster* than my hnng measurement (17.8k) at exact recall.
> - My **parallel-query refactor introduced a single-query regression** (per-call O(n) cache
>   re-prime); fixed in `1a03ec0` with a persistent per-index scratch.
>
> **What actually holds up (the float32 SIMD kernel's genuine wins, faithful single-threaded):**
> - **Build 2–5× faster** (batch 32k/d128: 132 s → 28.6 s; random-xs-20: → 0.69 s).
> - **High-dim query +127%** (Track B d=128 — now beats sklearn `kd_tree`). *(Consistent with my
>   fashion-mnist d=784 finding of ~no kernel gain: at 784-dim it's memory-bound; the kernel win
>   lands at moderate dim, not the extreme.)*
> - **insert/delete 15–60% faster**, insert latency ~flat in d.
> - **float32 is more faithful** to the ROS original / TKDE float32 schema (the double accumulation
>   was a numpy-parity deviation). Recall stays **1.0**; dist-eval counts identical → exactness unchanged.
> - **Thread-parallel batch queries remain available** (my ~8.1× at 20k×784) but the **published
>   numbers are single-threaded** per the papers' protocol.
>
> Everything below this banner is the **original investigation record, kept for provenance**; where
> it states AIhNNGc QPS/build or "float kernel sped up low-dim queries," defer to this banner.

---

**Generated:** June 7, 2026 — **actual measured runs** (replaces the synthetic numbers in
`ann-benchmarks/simulate_benchmarks.py` / `BENCHMARK_RESULTS.md`).

- **Harness:** ann-benchmarks, Docker (1 CPU, single-threaded), `--runs 1`
- **Dataset:** `random-xs-20-euclidean` — 9,000 train / 1,000 query, 20-dim, Euclidean, k=10
- **Algorithms:** `hnng` (complete impl, ghlim/hnng), `hnnglib` (partial impl), `aihnngc`
  (github.com/ghlim/AIhNNGc — clean C++/SIMD rewrite, batch + incremental build modes),
  `hnswlib` (real HNSW baseline, 81 configs), `bruteforce` (ground truth)
- **Recall** = mean fraction of true 10-NN returned (ann-benchmarks `k-nn` metric)

## Headline measured numbers (random-xs-20-euclidean)

| Algorithm | Recall@10 | QPS | Build time |
|---|---|---|---|
| **hnng** | **1.0000 (exact)** | 17,840 | 1.33 s |
| **aihnngc** (batch) | **1.0000 (exact)** | 13,449 | 3.08 s |
| **aihnngc** (incremental) | **1.0000 (exact)** | 13,836 | 10.81 s |
| **hnnglib** | **0.0541** | 66,515 | 1.28 s |
| hnswlib @ recall=1.0 | 1.0000 | 46,101 | 1.65 s |
| hnswlib @ recall≥0.99 | ≥0.99 | 64,814 | 1.53 s |
| hnswlib @ recall≥0.90 | ≥0.90 | 80,856 | 1.53 s |
| hnswlib max QPS | 0.824 | 81,316 | — |

**AIhNNGc note:** *[⚠️ SUPERSEDED — the 13.4K AIhNNGc baseline is a measurement artifact; faithful
is ~44.8K (double) / 38.8K (float). See top banner. The "slower than hnng" claim is wrong.]* it is
genuinely **exact (recall 1.0)**, confirming its core claim. On this
low-dim set it is ~25% slower in QPS than the older `hnng` (13.4K vs 17.8K) and ~3.4× below HNSW
at exact recall — consistent with its own README's honest framing that exact triangle-inequality
pruning cannot beat HNSW/BLAS, its value being incremental maintenance + balance, not raw QPS.
Its `memory_bytes()` accounting returned a negative index size (a reporting bug; does not affect
recall/QPS).

## AIhNNGc optimization (branch perf/hnswlib-distance): hnswlib float SIMD kernel + buffer reuse

> *[⚠️ SUPERSEDED — QPS/build in this section rest on the artifact 13.4K baseline. Faithful numbers:
> double 44.8K → float 38.8K QPS (a low-dim dip, not a 3.3× gain), build 0.69 s. See top banner.]*

| Dataset | Version | Recall | QPS | Build |
|---|---|---|---|---|
| random-xs-20 (9K×20, fits cache) | baseline (double kernel) | 1.0 | 13,449 | 3.08 s |
| random-xs-20 | **Phase 2+3 (float SIMD + buffer reuse)** | 1.0 | **44,411** | **1.47 s** |
| fashion-mnist-784 (60K×784, exceeds cache) | baseline | 1.0 | 26.5 | 3,016.9 s |
| fashion-mnist-784 | **Phase 2+3** | 1.0 | **26.1** | **2,993.6 s** |

**Key finding — the kernel optimization is cache-regime-dependent:**
- **Low-dim / small-n (working set 720 KB, fits L2/L3): 3.3× faster queries, 2.1× faster build.** Here the search is compute-bound, so swapping the `double` scalar loop for hnswlib's `float` AVX kernel pays off fully — AIhNNGc flips from slower-than-hnng (13.4K vs 17.8K) to **faster than hnng** (44.4K) and near hnswlib (46.1K).
- **High-dim / large-n (working set 188 MB ≫ 24.75 MB L3): no change (26.5→26.1 QPS, build unchanged).** At 784-dim near brute force, both build and query are **memory-bandwidth-bound** — both kernels read the same float32 data, so faster arithmetic is hidden under memory latency.
- **Caveat on the earlier micro-benchmark:** its 6.2× at dim=784 used a 4,096-vector DB that fits in L2 (compute-bound regime). That speedup materializes end-to-end only when the dataset fits in cache; it does **not** at fashion-mnist scale. Honest correction.
- **Implication:** closing the high-dim gap to hnng (53 vs 26 QPS) is NOT a kernel problem — it's memory layout (Phase 4: flat contiguous storage like hnng's level-0 block → better bandwidth/prefetch) and parallelism (Phase 6). Those are outside the approved Phases 0–3.

## Phase 4 (flatten storage): DIAGNOSED, NOT EXECUTED — it would not help

Measured at d=784 (the memory-bound regime) to decide whether flattening storage helps:
- AIhNNGc search degrades to **full brute force** at high dim: dist-evals == n (10K, 20K).
- AIhNNGc per-eval cost: **~1,100 ns/eval**.
- A bare brute-force scan over the SAME (already-contiguous) coordinates: **237 ns/eval sequential, 350 ns/eval scattered** (scattered == AIhNNGc's member access pattern).
- => Coordinate memory is only ~350 ns; the other **~750 ns/eval is best-first search machinery** — per-leaf `std::push_heap`/`pop_heap` (O(log n) on a PQ that grows large), the `delta()` cache check, lower-bound math, and `offer()` — paid for every one of the n points.

**Conclusion:** flattening `Element`/inlinks storage cannot move the high-dim number — coordinates are already contiguous and aren't the bottleneck; the graph structures are tiny next to the 3,136-byte vectors. The real high-dim levers are:
1. **Leaf PQ-bypass** — at cluster expansion, offer level-0 (leaf) children directly to the result heap instead of round-tripping them through the cluster PQ (mirrors hnng's flat level-0 scan). Directly attacks the ~750 ns overhead. Exactness-preserving in principle; must verify recall + the 193-test gate.
2. **Query parallelism** (Phase 6) — 36 cores, near-linear QPS, trivially safe; the biggest wall-clock lever.

## Leaf PQ-bypass (executed, test-gated): correct but NO high-dim benefit

Implemented the leaf PQ-bypass (offer level-0 children directly instead of PQ
round-trip). Result: 193/193 tests pass, recall 1.0, low-dim unchanged
(random-xs-20 43,787 QPS), but high-dim ns/eval was **unchanged**:

| d=784 | pre-bypass | post-bypass |
|---|---|---|
| n=10K | ~1,100 ns/eval, 91.6 QPS | 1,102 ns/eval, 90.8 QPS |
| n=20K | ~1,150 ns/eval, 43.5 QPS | 1,161 ns/eval, 43.1 QPS |

**Hypothesis falsified:** the ~750 ns/eval overhead is NOT the per-leaf heap
push/pop (removing them changed nothing). It is the search's **multi-stream
random memory access** per evaluation — the coordinate row (`data_`), the fat
`Element` struct (~56 B × n, exceeds L2 at these sizes), and the two flat
distance-cache arrays — three independent random streams vs the single stream of
a tight brute-force loop (350 ns/eval). The kernel, storage flattening, and the
PQ-bypass all leave this untouched.

**Remaining real lever for high dim: parallelism (Phase 6)** — 36 cores give
~linear QPS and dominate every per-eval micro-optimization. Fundamentally,
though, at 784-dim the method runs full brute force (evals = n), and an indexed
exact method cannot beat a tight BLAS brute-force scan there — a documented
curse-of-dimensionality limit, not an implementation defect.

## Phase 6 (query parallelism): EXECUTED — 8.1× at high dim, exact

> *[⚠️ PARTIALLY SUPERSEDED — the 8.1× parallel capability is real and retained, but the published
> AIhNNGc numbers are SINGLE-THREADED (papers' protocol). This refactor also introduced a
> single-query regression (per-call O(n) cache re-prime), fixed upstream in commit 1a03ec0 with a
> persistent per-index scratch. See top banner.]*

Reverted the no-benefit leaf PQ-bypass, then added thread-parallel batch queries:
each thread owns a `QueryScratch` (its own distance cache + result heap + cluster
PQ + counters), so queries share no mutable state and run concurrently. Binding
parallelizes the batch path with the GIL released (`num_threads=0` auto); the
ann-benchmarks module gained `batch_query`/`get_batch_results` (use `--batch`).

- **193/193 tests pass** (they run multi-query batches → exercise the parallel
  path), recall 1.0.
- **Parallel == sequential results: bit-identical** (verified) — each query is
  deterministic regardless of thread count.
- **High-dim speedup (synthetic 20K×784, 2000 queries, i9-10980XE 36c):**
  1 thread = 111 QPS → auto (36 threads) = **901.9 QPS = 8.1×**. Sub-linear
  because the workload is memory-bandwidth-bound (cores contend for DRAM reading
  the 3,136-byte coordinate vectors) — consistent with the per-eval finding.
- **Low-dim (random-xs-20) unchanged** (43,787 QPS): queries are ~22 µs each, so
  thread-spawn overhead cancels the gain on tiny work — parallelism helps where
  each query is expensive (high dim).

**Net:** parallelism is the lever that actually moves the high-dim number (~8×),
and it's exactness-preserving. Projected fashion-mnist (60K×784): ~26 → ~200 QPS.

## Build-scaling: fashion-mnist-784-euclidean (60,000 × 784, k=10)

| Algorithm | Recall@10 | QPS @ recall 1.0 | Build time | Build scale 9K→60K |
|---|---|---|---|---|
| hnswlib (HNSW) | 1.0 (tunable) | **897.3** | **75.1 s** | ~45× |
| **hnng** | 1.0000 | 53.1 | 2,237.9 s (37 min) | ~1,680× |
| **aihnngc** (batch) | 1.0000 | 26.5 | 3,016.9 s (50 min) | ~980× |
| hnnglib | 0.0176 ❌ | 32,961 (meaningless) | 929.8 s | ~728× |

**Findings:**
- Both exact hNNG methods collapse at 784 dims: hnng 53 QPS, aihnngc 26.5 QPS vs HNSW 897 — i.e.
  **3–6% of HNSW** (vs ~30–39% at 20 dims). Textbook curse of dimensionality for triangle-inequality
  pruning (AIhNNGc's README predicts exactly this).
- **AIhNNGc is the *slowest* exact method on this high-dim set** — half hnng's QPS (26.5 vs 53) and
  the longest build (50 min vs 37 min) — despite the best engineering. Its bounding-sphere
  branch-and-bound prunes no better than hnng's here, with higher constant factors. Its real value
  (incremental insert/delete, balance) is not captured by this static batch benchmark.
- **hnnglib recall falls further with dimension: 0.054 → 0.018.** Confirmed broken as a k-NN index.
- HNSW builds the 60K/784 index in 75 s and answers at 897 QPS exact — ~30–40× faster build and
  17–34× faster queries than the exact hNNGs.

## Key findings (all measured, not simulated)

1. **`hnng` is exact (recall = 1.0)** — confirms the design claim, on real ann-benchmarks.
2. **`hnnglib` is NOT exact — recall ≈ 0.05.** Verified in-container vs brute force: it returns
   the nearest ~1–4 neighbors with *correct* labels, then misses the rest (recall 0.138 @ 2K pts →
   0.054 @ 9K pts). This is a genuine search-completeness deficit of the partial implementation
   (descends into one cluster, under-explores), **not** a binding/label bug. The simulation's
   assumption that hnnglib reaches 0.5–0.99 recall is wrong for the current code.
3. **`hnng` vs HNSW at identical (exact) recall:** hnng = 17,840 QPS vs HNSW's best exact config
   46,101 QPS → **hnng ≈ 38.7% of HNSW QPS** at recall 1.0. (The simulation guessed 20–30%.)
4. **Build time** is ~1.3 s for both hnng/hnnglib at 9K/20d — comparable to HNSW (~1.5–1.7 s) at
   this scale. The old "3–4× slower build" claim was simulated and not reproduced here (small n;
   O(n²) effects appear at larger n).

## Reproduce

```bash
cd ann-benchmarks
python install.py --algorithm hnnglib   # build images (hnng/hnswlib/bruteforce already built)
python run.py --dataset random-xs-20-euclidean --algorithm hnng     --runs 1 --force
python run.py --dataset random-xs-20-euclidean --algorithm hnnglib  --runs 1 --force
python run.py --dataset random-xs-20-euclidean --algorithm hnswlib  --runs 1 --force
python run.py --dataset random-xs-20-euclidean --algorithm bruteforce --runs 1 --force
python data_export.py --output real_results.csv
```

## Fixes applied to make the real harness run (were blocking)

- `ann_benchmarks/algorithms/hnnglib/src/setup.py` — `read_file()` tolerates missing `README.md`.
- `ann_benchmarks/algorithms/{hnng,hnnglib}/module.py` — `__init__` accepts the config's empty
  param dict; `query()` returns the full k-label row (was returning a single scalar; hnng flattens
  the (1,k) tuple output).
- Rebuilt `ann-benchmarks-hnng` image (the existing one lacked the compiled `hnnglib` module) and
  built the missing `ann-benchmarks-hnnglib` image.
