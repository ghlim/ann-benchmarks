// hnng.hpp -- the exact hierarchical Nearest Neighbor Graph index.
//
// Compiles and is tested: built with g++ 13.3 / pybind11 3.0.4; the full suite
// (incl. C++/Python parity) runs in CI.
//
// This is a header-only template-free core (all `coord_t = float`). It mirrors
// hnng_ref/core.py field-for-field and step-for-step so that, for any input,
// build() / knn_query() / insert() / remove() produce identical results:
//
//   * build()      : bottom-up NNG clustering (sort objects, link nearest,
//                    connected components, HDE rep, accumulating radius, promote)
//   * knn_query()  : top-down best-first branch-and-bound with per-query
//                    distance memoization and (dist, object_index) tie-breaking
//   * insert()/remove() : TRUE incremental AIhNNGc (port of ROS
//                    race_view_clustering). insert() appends the leaf, links it
//                    to its level-0 NN and asymmetrically re-links others
//                    (updateTheOtherElements), then reconcile_up() recomputes
//                    each level's clusters from the maintained links and
//                    propagates rep add/remove/field-update changes upward;
//                    remove() re-links the removed node's inlinkers and
//                    reconciles up. Neither calls rebuild_from_active(); cost is
//                    ~O(n*d) per op (vs the O(n^2) rebuild). The hierarchy is not
//                    guaranteed structurally identical to a batch rebuild, but it
//                    stays a VALID hNNG (full coverage + accumulating radii that
//                    bound each subtree), so the exact search returns the true
//                    k-NN -- verified by tests/test_cpp_parity.py against the
//                    rebuild-based hnng_ref oracle (tie-safe). build() still uses
//                    the batch rebuild_from_active().
//
// Determinism notes (must match the reference):
//   * level 0 elements are created in ascending object_index order;
//   * nearest-neighbour ties break to the smaller object_index;
//   * components are emitted sorted by smallest object_index, members sorted by
//     object_index;
//   * rep selection uses the exact HDE key;
//   * results are ordered by (distance, object_index).

#ifndef HNNG_HNNG_HPP
#define HNNG_HNNG_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hnng/distance.hpp"
#include "hnng/element.hpp"
#include "hnng/heap.hpp"
#include "hnng/level.hpp"

namespace hnng {

// HDE (Highest-Density Element) selection rule. Both rules end with the same
// deterministic object_index tiebreak; they differ only in the leading keys:
//   * V2026 (default): num_child -> #inlinks -> avg(inlink dist) -> outlink dist
//   * V2015 (original ROS): #inlinks -> avg(inlink dist) -> outlink dist
//     (no num_child term)
enum class HdeRule : std::uint8_t { V2026 = 0, V2015 = 1 };

// Per-query mutable search state, bundled so a batch of queries can run on
// separate threads, each with its OWN scratch: the flat per-object distance
// cache, the bounded result heap, the best-first cluster PQ, and the stat
// counters. One scratch is reused across all queries a single thread handles
// (no per-query reallocation). The build path keeps using the HNNG members;
// only the query path uses QueryScratch, so queries never touch shared mutable
// state -> they are embarrassingly parallel and each is deterministic.
struct QueryScratch {
    std::vector<double> cache_val;           // memoized distance per object
    std::vector<std::uint32_t> cache_epoch;  // per-object epoch stamp
    std::uint32_t epoch = 0;                 // current query epoch (0 = never)
    std::uint64_t dist_evals = 0;            // distinct evals, last query
    std::uint64_t clusters_visited = 0;      // clusters expanded, last query
    BoundedMaxResults results;               // k best leaves
    MinClusterPQ pq;                         // best-first cluster frontier

    explicit QueryScratch(std::size_t k) : results(k) {}

    // Size the flat cache to n objects (once, before a thread's queries).
    void prime(std::size_t n) {
        if (cache_val.size() != n) {
            cache_val.assign(n, 0.0);
            cache_epoch.assign(n, 0u);
            epoch = 0;
        }
    }
    // Start a fresh query: O(1) cache invalidation via epoch bump (re-zero on
    // wrap), clear the heaps, reset counters.
    void begin() {
        if (epoch == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(cache_epoch.begin(), cache_epoch.end(), 0u);
            epoch = 0;
        }
        ++epoch;
        results.clear();
        pq.clear();
        dist_evals = 0;
        clusters_visited = 0;
    }
};

inline HdeRule hde_rule_from_string(const std::string& name) {
    if (name == "v2026") return HdeRule::V2026;
    if (name == "v2015") return HdeRule::V2015;
    throw std::invalid_argument("unknown hde_rule '" + name +
                                "'; expected 'v2026' or 'v2015'");
}

inline const char* hde_rule_to_string(HdeRule r) {
    switch (r) {
        case HdeRule::V2026: return "v2026";
        case HdeRule::V2015: return "v2015";
    }
    return "v2026";
}

struct Stats {
    std::size_t height = 0;
    std::size_t num_levels = 0;
    std::vector<std::size_t> clusters_per_level;
    double avg_cluster_size = 0.0;
    std::size_t last_query_dist_evals = 0;
    std::size_t last_query_clusters_visited = 0;
};

class HNNG {
public:
    explicit HNNG(Metric m, HdeRule hde = HdeRule::V2026)
        : metric_(m), hde_rule_(hde) {}
    explicit HNNG(const std::string& metric, const std::string& hde_rule = "v2026")
        : metric_(metric_from_string(metric)),
          hde_rule_(hde_rule_from_string(hde_rule)) {}

    Metric metric() const { return metric_; }
    HdeRule hde_rule() const { return hde_rule_; }
    std::size_t dim() const { return dim_; }
    // Number of objects currently stored in the backing store (includes removed
    // objects whose vectors are retained so ids stay stable, matching the
    // reference which keeps removed vectors in self._data).
    std::size_t num_stored() const { return dim_ == 0 ? 0 : data_.size() / dim_; }

    // -------------------------------------------------------------- build
    // Build from a row-major (n, dim) buffer of RAW (un-prepared) float data.
    void build(const coord_t* data, std::size_t n, std::size_t dim) {
        if (dim == 0) throw std::invalid_argument("build: dim must be > 0");
        dim_ = dim;
        data_ = prepare_matrix(data, n, dim, metric_);  // float64-style prepare
        active_.resize(n);
        for (std::size_t i = 0; i < n; ++i) active_[i] = static_cast<std::uint32_t>(i);
        rebuild_from_active();
    }

    // -------------------------------------------------------------- insert
    // Insert a single RAW vector of length dim_. Returns the new object id.
    // Mirrors hnng_ref.insert: append to the store, add to the active set,
    // rebuild. (If the index is empty and no dim is known yet, the first insert
    // establishes the dimension from `dim` arg.)
    std::uint32_t insert(const coord_t* vec, std::size_t dim) {
        if (dim_ == 0) {
            if (dim == 0) throw std::invalid_argument("insert: dim must be > 0");
            dim_ = dim;
        } else if (dim != dim_) {
            throw std::invalid_argument("insert: dimension mismatch");
        }
        std::vector<coord_t> prepared(dim_);
        prepare_vector(vec, prepared.data(), dim_, metric_);
        const std::uint32_t new_id = static_cast<std::uint32_t>(num_stored());
        data_.insert(data_.end(), prepared.begin(), prepared.end());
        active_.push_back(new_id);
        insert_incremental(new_id);
        return new_id;
    }

    // -------------------------------------------------------------- remove
    // Remove an object id from the active set and rebuild. The vector stays in
    // the backing store so existing ids remain stable (matches the reference).
    void remove(std::uint32_t object_id) {
        auto it = std::find(active_.begin(), active_.end(), object_id);
        if (it == active_.end()) {
            throw std::out_of_range("object " + std::to_string(object_id) +
                                    " is not present in the index");
        }
        active_.erase(it);
        remove_incremental(object_id);
    }

    // -------------------------------------------------------------- query
    // Exact k-NN for a batch of m RAW query vectors (row-major (m, dim_)).
    // Fills `out_ids` (size m*k, row-major), right-padding short rows with -1.
    // Updates the per-query stat counters to reflect the LAST query in the batch
    // (consistent with the reference, whose stats() reports the last query).
    void knn_query_batch(const coord_t* queries, std::size_t m, std::size_t dim,
                         std::size_t k, std::int64_t* out_ids) {
        if (k == 0) throw std::invalid_argument("k must be positive");
        // Initialize all output to -1 (padding).
        for (std::size_t i = 0; i < m * k; ++i) out_ids[i] = -1;

        last_query_dist_evals_ = 0;
        last_query_clusters_visited_ = 0;

        // Empty / never-built index: no dimension is established yet and there
        // is nothing to search, so every row is all -1 padding. Mirrors
        // hnng_ref, whose _knn_single returns [] when there are no levels (the
        // (m, k) / (k,) output stays -1-filled). Return BEFORE the dimension
        // check so an empty index does not raise on a query of any width.
        if (dim_ == 0 || levels_.empty()) return;

        if (dim != dim_) throw std::invalid_argument("knn_query: dimension mismatch");

        // One reusable QueryScratch for the whole batch (Phase 3 buffer reuse):
        // the flat distance cache, result heap, and cluster PQ are allocated once
        // and reset per query. knn_single writes sorted ids straight into out_ids.
        QueryScratch scratch(k);
        scratch.prime(num_stored());
        std::vector<coord_t> q(dim_);
        for (std::size_t qi = 0; qi < m; ++qi) {
            prepare_vector(queries + qi * dim_, q.data(), dim_, metric_);
            scratch.begin();
            knn_single(q.data(), k, scratch, out_ids + qi * k);
        }
        // Report stats for the LAST query (matches the reference convention).
        last_query_dist_evals_ = scratch.dist_evals;
        last_query_clusters_visited_ = scratch.clusters_visited;
    }

    // Parallel exact k-NN over a batch: each thread owns its QueryScratch, so
    // queries run concurrently with NO shared mutable state. Each query reads
    // only the (immutable-during-query) index and writes its own scratch + its
    // own disjoint output rows, so results are identical to the sequential path
    // and deterministic regardless of thread count. num_threads<=1, m<=1, or an
    // empty index falls back to the sequential knn_query_batch.
    void knn_query_batch_parallel(const coord_t* queries, std::size_t m,
                                  std::size_t dim, std::size_t k,
                                  std::int64_t* out_ids, unsigned num_threads) {
        if (num_threads <= 1 || m <= 1) {
            knn_query_batch(queries, m, dim, k, out_ids);
            return;
        }
        if (k == 0) throw std::invalid_argument("k must be positive");
        for (std::size_t i = 0; i < m * k; ++i) out_ids[i] = -1;
        last_query_dist_evals_ = 0;
        last_query_clusters_visited_ = 0;
        if (dim_ == 0 || levels_.empty()) return;
        if (dim != dim_) throw std::invalid_argument("knn_query: dimension mismatch");

        const unsigned T = static_cast<unsigned>(std::min<std::size_t>(num_threads, m));
        const std::size_t n = num_stored();
        const std::size_t chunk = (m + T - 1) / T;  // static block partition
        std::vector<QueryScratch> scratch;
        scratch.reserve(T);
        for (unsigned t = 0; t < T; ++t) {
            scratch.emplace_back(k);
            scratch.back().prime(n);
        }
        auto worker = [&](unsigned t) {
            std::vector<coord_t> q(dim_);
            QueryScratch& s = scratch[t];
            const std::size_t begin = static_cast<std::size_t>(t) * chunk;
            const std::size_t end = std::min(begin + chunk, m);
            for (std::size_t qi = begin; qi < end; ++qi) {
                prepare_vector(queries + qi * dim_, q.data(), dim_, metric_);
                s.begin();
                knn_single(q.data(), k, s, out_ids + qi * k);
            }
        };
        std::vector<std::thread> pool;
        pool.reserve(T - 1);
        for (unsigned t = 1; t < T; ++t) pool.emplace_back(worker, t);
        worker(0);
        for (auto& th : pool) th.join();
        // Stats reflect the last query, handled by the thread owning chunk end.
        const unsigned last_t = static_cast<unsigned>((m - 1) / chunk);
        last_query_dist_evals_ = scratch[last_t].dist_evals;
        last_query_clusters_visited_ = scratch[last_t].clusters_visited;
    }

    // -------------------------------------------------------------- stats
    Stats stats() const {
        Stats s;
        s.num_levels = levels_.size();
        s.height = levels_.empty() ? 0 : levels_.size() - 1;
        s.clusters_per_level.reserve(clusters_.size());
        std::size_t total_members = 0;
        std::size_t total_clusters = 0;
        for (const auto& level_clusters : clusters_) {
            s.clusters_per_level.push_back(level_clusters.size());
            for (const auto& cl : level_clusters) {
                total_members += cl.members.size();
                ++total_clusters;
            }
        }
        s.avg_cluster_size = total_clusters ? static_cast<double>(total_members) /
                                                  static_cast<double>(total_clusters)
                                            : 0.0;
        s.last_query_dist_evals = last_query_dist_evals_;
        s.last_query_clusters_visited = last_query_clusters_visited_;
        return s;
    }

    // -------------------------------------------------------------- memory
    // Approximate resident byte-size of the index: the prepared coordinate
    // store plus every level's element array and every cluster's member array
    // (the inlink/member vectors are the dominant per-element heap allocations).
    // This is a structural estimate of the index's own buffers, not a process
    // RSS figure; it is what get_memory_usage() reports to ann-benchmarks.
    std::size_t memory_bytes() const {
        std::size_t bytes = 0;
        bytes += data_.capacity() * sizeof(coord_t);
        bytes += active_.capacity() * sizeof(std::uint32_t);
        for (const Level& lvl : levels_) {
            bytes += lvl.elements.capacity() * sizeof(Element);
            for (const Element& e : lvl.elements) {
                bytes += e.inlinks.capacity() * sizeof(std::uint32_t);
            }
        }
        for (const std::vector<Cluster>& level_clusters : clusters_) {
            bytes += level_clusters.capacity() * sizeof(Cluster);
            for (const Cluster& cl : level_clusters) {
                bytes += cl.members.capacity() * sizeof(std::uint32_t);
            }
        }
        bytes += delta_cache_val_.capacity() * sizeof(double);
        bytes += delta_cache_epoch_.capacity() * sizeof(std::uint32_t);
        return bytes;
    }

    // Read-only access to internal structure (useful for parity tests / bindings).
    const std::vector<Level>& levels() const { return levels_; }
    const std::vector<std::vector<Cluster>>& clusters() const { return clusters_; }

private:
    // -------------------------------------------------- coordinate access
    const coord_t* vec(std::uint32_t object_index) const {
        return data_.data() + static_cast<std::size_t>(object_index) * dim_;
    }

    // Memoized per-query distance from prepared query `q` to a stored object.
    // Mirrors hnng_ref._delta: the first evaluation of an object computes and
    // caches the distance and bumps the eval counter; later requests reuse the
    // cached value WITHOUT counting a new evaluation. So last_query_dist_evals_
    // counts DISTINCT object-distance evaluations (never exceeds n).
    //
    // Implementation: an O(1) flat cache instead of an std::unordered_map. We
    // keep one slot per stored object (`delta_cache_val_`) plus a per-slot epoch
    // stamp (`delta_cache_epoch_`); a slot is "present for this query" iff its
    // stamp equals the current query epoch (`cur_epoch_`). Bumping cur_epoch_
    // per query invalidates the whole cache in O(1) with no allocation/clear,
    // and the membership test / value lookup are O(1). Behaviour is identical to
    // the map: each distinct object is evaluated (and counted) at most once.
    double delta(const coord_t* q, std::uint32_t object_index) {
        if (delta_cache_epoch_[object_index] == cur_epoch_) {
            return delta_cache_val_[object_index];
        }
        ++last_query_dist_evals_;
        const double d = l2_distance(q, vec(object_index), dim_);
        delta_cache_val_[object_index] = d;
        delta_cache_epoch_[object_index] = cur_epoch_;
        return d;
    }

    // Thread-safe memoized distance: identical logic to delta() but reads/writes
    // a caller-owned QueryScratch instead of the shared HNNG members, so
    // concurrent queries on different scratches never interfere.
    double delta_s(const coord_t* q, std::uint32_t object_index, QueryScratch& s) {
        if (s.cache_epoch[object_index] == s.epoch) {
            return s.cache_val[object_index];
        }
        ++s.dist_evals;
        const double d = l2_distance(q, vec(object_index), dim_);
        s.cache_val[object_index] = d;
        s.cache_epoch[object_index] = s.epoch;
        return d;
    }

    // Ensure the flat distance cache has one slot per stored object and start a
    // fresh per-query epoch (O(1) invalidation). Epoch 0 is reserved as "never
    // stamped", so the first query runs at epoch 1; the wrap-around case (after
    // ~4 billion queries) is handled by re-zeroing the stamps.
    void begin_query_epoch() {
        const std::size_t n = num_stored();
        if (delta_cache_val_.size() != n) {
            delta_cache_val_.assign(n, 0.0);
            delta_cache_epoch_.assign(n, 0);
            cur_epoch_ = 0;
        }
        if (cur_epoch_ == std::numeric_limits<std::uint32_t>::max()) {
            std::fill(delta_cache_epoch_.begin(), delta_cache_epoch_.end(), 0);
            cur_epoch_ = 0;
        }
        ++cur_epoch_;
    }

    // -------------------------------------------------- NNG build helpers

    // For each element set outlink/dist to its nearest neighbour among the
    // others at this level (linear scan over prepared coordinates), and rebuild
    // inlinks. Deterministic: ties on distance break to the smaller object_index;
    // never link to self. Mirrors hnng_ref._link_nearest.
    void link_nearest(Level& level) {
        const std::size_t n = level.size();
        for (auto& e : level.elements) e.inlinks.clear();
        if (n <= 1) {
            if (n == 1) {
                level[0].outlink = kNone;
                level[0].dist = 0.0;
            }
            return;
        }

        // Gather coordinate pointers once.
        std::vector<const coord_t*> coords(n);
        for (std::size_t i = 0; i < n; ++i) coords[i] = vec(level[i].object_index);

        for (std::size_t i = 0; i < n; ++i) {
            std::uint32_t best_j = kNone;
            double best_d = std::numeric_limits<double>::infinity();
            std::uint32_t best_obj = 0;
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i) continue;  // never link to self
                const double dj = l2_distance(coords[i], coords[j], dim_);
                const std::uint32_t obj_j = level[j].object_index;
                // best_j == kNone || dj < best_d || (dj == best_d && obj_j < best_obj)
                if (best_j == kNone || dj < best_d ||
                    (dj == best_d && obj_j < best_obj)) {
                    best_d = dj;
                    best_j = static_cast<std::uint32_t>(j);
                    best_obj = obj_j;
                }
            }
            level[i].outlink = best_j;
            level[i].dist = best_d;
        }
        // Rebuild inlinks. The reference uses a set; iteration order over i is
        // ascending, so we naturally append in ascending position order and keep
        // them unique.
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t out = level[i].outlink;
            level[out].inlinks.push_back(static_cast<std::uint32_t>(i));
        }
    }

    // Partition a level into NNG connected components using the undirected
    // version of the NN edges (outlink + inlinks). Components are returned with
    // members sorted by object_index; the component list is sorted by the
    // smallest object_index it contains. Mirrors hnng_ref._connected_components.
    std::vector<std::vector<std::uint32_t>> connected_components(const Level& level) {
        const std::size_t n = level.size();
        std::vector<char> seen(n, 0);
        std::vector<std::vector<std::uint32_t>> comps;
        std::vector<std::uint32_t> stack;

        for (std::size_t start = 0; start < n; ++start) {
            if (seen[start]) continue;
            stack.clear();
            stack.push_back(static_cast<std::uint32_t>(start));
            seen[start] = 1;
            std::vector<std::uint32_t> comp;
            while (!stack.empty()) {
                const std::uint32_t i = stack.back();
                stack.pop_back();
                comp.push_back(i);
                const Element& ei = level[i];
                // Neighbours = inlinks + outlink (undirected). Visit each once.
                for (std::uint32_t j : ei.inlinks) {
                    if (!seen[j]) { seen[j] = 1; stack.push_back(j); }
                }
                if (ei.outlink != kNone && !seen[ei.outlink]) {
                    seen[ei.outlink] = 1;
                    stack.push_back(ei.outlink);
                }
            }
            std::sort(comp.begin(), comp.end(),
                      [&](std::uint32_t a, std::uint32_t b) {
                          return level[a].object_index < level[b].object_index;
                      });
            comps.push_back(std::move(comp));
        }
        std::sort(comps.begin(), comps.end(),
                  [&](const std::vector<std::uint32_t>& a,
                      const std::vector<std::uint32_t>& b) {
                      return level[a[0]].object_index < level[b[0]].object_index;
                  });
        return comps;
    }

    // Pick the Highest-Density Element (HDE) of a cluster. Priority, all with a
    // final deterministic tiebreak by object_index (mirrors hnng_ref._select_rep):
    //   1. num_child         (higher better)  -> key -num_child
    //   2. #inlinks          (higher better)  -> key -n_in
    //   3. avg(inlink dist)  (lower better, inf if no inlinks)
    //   4. outlink dist      (lower better, 0 if no outlink)
    //   5. object_index      (lower better)
    std::uint32_t select_rep(const Level& level,
                             const std::vector<std::uint32_t>& members) const {
        // Build the comparison key for one member position.
        struct Key {
            std::int64_t neg_num_child;
            std::int64_t neg_n_in;
            double avg_in;
            double out_d;
            std::uint32_t object_index;
        };
        auto make_key = [&](std::uint32_t i) -> Key {
            const Element& e = level[i];
            const std::size_t n_in = e.inlinks.size();
            double avg_in;
            if (n_in > 0) {
                double s = 0.0;
                for (std::uint32_t j : e.inlinks) s += level[j].dist;
                avg_in = s / static_cast<double>(n_in);
            } else {
                avg_in = std::numeric_limits<double>::infinity();
            }
            const double out_d = (e.outlink != kNone) ? e.dist : 0.0;
            return Key{-static_cast<std::int64_t>(e.num_child),
                       -static_cast<std::int64_t>(n_in), avg_in, out_d,
                       e.object_index};
        };
        const bool use_num_child = (hde_rule_ == HdeRule::V2026);
        auto less = [use_num_child](const Key& a, const Key& b) {
            // V2026 leads with num_child; V2015 (original ROS) omits it.
            if (use_num_child && a.neg_num_child != b.neg_num_child)
                return a.neg_num_child < b.neg_num_child;
            if (a.neg_n_in != b.neg_n_in) return a.neg_n_in < b.neg_n_in;
            if (a.avg_in != b.avg_in) return a.avg_in < b.avg_in;
            if (a.out_d != b.out_d) return a.out_d < b.out_d;
            return a.object_index < b.object_index;
        };
        std::uint32_t best = members.front();
        Key best_key = make_key(best);
        for (std::size_t idx = 1; idx < members.size(); ++idx) {
            Key k = make_key(members[idx]);
            if (less(k, best_key)) {
                best_key = k;
                best = members[idx];
            }
        }
        return best;
    }

    // Cluster one level's elements, compute accumulating radii, return clusters.
    //   R = max_{m in members}[ delta(rep, m) + subtree_radius(m) ].
    // Mirrors hnng_ref._build_level_clusters. Note: the reference computes the
    // rep<->member distances here with metric.dist directly (NOT through the
    // memoized _delta), so these do not affect query eval counters.
    std::vector<Cluster> build_level_clusters(std::uint32_t level_idx) {
        Level& level = levels_[level_idx];
        std::vector<std::vector<std::uint32_t>> comps = connected_components(level);
        std::vector<Cluster> clusters;
        clusters.reserve(comps.size());
        for (auto& comp : comps) {
            // refresh_cluster sets cl.rep (HDE) and cl.radius (accumulating) from
            // the members; the placeholder rep is overwritten. Its num_child
            // return is unused here -- promotion recomputes num_child below.
            Cluster cl(level_idx, comp.front(), comp);  // copies comp into members
            refresh_cluster(level_idx, cl);
            clusters.push_back(std::move(cl));
        }
        return clusters;
    }

    // (Re)build the whole hierarchy bottom-up from the active object set. The
    // active set is sorted ascending here (the reference sorts object_indices),
    // so the resulting level-0 ordering is deterministic and id-stable.
    void rebuild_from_active() {
        levels_.clear();
        clusters_.clear();
        elem_cluster_.clear();
        if (active_.empty()) return;

        std::vector<std::uint32_t> ordered = active_;
        std::sort(ordered.begin(), ordered.end());

        Level level0;
        level0.elements.reserve(ordered.size());
        for (std::uint32_t obj : ordered) level0.elements.emplace_back(obj);
        levels_.push_back(std::move(level0));

        std::uint32_t level = 0;
        for (;;) {
            link_nearest(levels_[level]);
            if (levels_[level].size() == 1) {
                // Single element: it is the root. No cluster list above it.
                clusters_.emplace_back();  // empty cluster list for this level
                break;
            }
            std::vector<Cluster> cls = build_level_clusters(level);

            // Promote each Rep to the next level.
            Level next_level;
            next_level.elements.reserve(cls.size());
            for (std::size_t ci = 0; ci < cls.size(); ++ci) {
                Cluster& cl = cls[ci];
                const Element& rep_elem = levels_[level][cl.rep];
                Element pe(rep_elem.object_index);
                pe.num_child = sum_num_child(levels_[level], cl.members);
                pe.subtree_radius = cl.radius;
                pe.heads_cluster = static_cast<std::uint32_t>(ci);
                cl.parent = static_cast<std::uint32_t>(next_level.elements.size());
                next_level.elements.push_back(std::move(pe));
            }

            clusters_.push_back(std::move(cls));
            levels_.push_back(std::move(next_level));
            ++level;
        }
        // Build the position->cluster inverse map for every level (the root
        // level has an empty cluster list, so its map is all kNone).
        elem_cluster_.assign(levels_.size(), {});
        for (std::uint32_t L = 0; L < levels_.size(); ++L) sync_elem_cluster(L);
    }

    // ============================================================ INCREMENTAL
    //
    // True incremental insert/remove (port of the ROS AIhNNGc algorithm). These
    // do NOT call rebuild_from_active(). They maintain, after every op and for
    // every level L: each element's exact NN outlink/dist among level L (ties ->
    // smaller object_index, never self), correct inlinks, clusters_[L] =
    // connected components with HDE rep + accumulating radius, and for L>=1 a 1:1
    // mapping between levels_[L] elements and clusters_[L-1] clusters (object_index
    // = cluster's rep object_index, num_child = sum of members' num_child,
    // subtree_radius = cluster.radius, heads_cluster = cluster index;
    // cluster.parent = the up-element's position).
    //
    // Correctness only requires the two invariants the top-down search relies on
    // (coverage + radius validity); maintaining the real NNG is for pruning speed
    // and fidelity to the algorithm.

    // Rebuild a level's inlinks from its current outlinks (ascending position
    // order, matching link_nearest / the reference).
    static void rebuild_inlinks(Level& level) {
        for (auto& e : level.elements) e.inlinks.clear();
        const std::size_t n = level.size();
        for (std::size_t i = 0; i < n; ++i) {
            const std::uint32_t out = level[i].outlink;
            if (out != kNone) level[out].inlinks.push_back(static_cast<std::uint32_t>(i));
        }
    }

    // Find the nearest neighbour of element `i` among the other elements of
    // `level` (linear scan; ties -> smaller object_index; never self). Returns
    // {position, distance}. `level` must have >= 2 elements.
    std::pair<std::uint32_t, double> find_nn(const Level& level, std::uint32_t i) const {
        const std::size_t n = level.size();
        const coord_t* ci = vec(level[i].object_index);
        std::uint32_t best_j = kNone;
        double best_d = std::numeric_limits<double>::infinity();
        std::uint32_t best_obj = 0;
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dj = l2_distance(ci, vec(level[j].object_index), dim_);
            const std::uint32_t obj_j = level[j].object_index;
            if (best_j == kNone || dj < best_d ||
                (dj == best_d && obj_j < best_obj)) {
                best_d = dj;
                best_j = static_cast<std::uint32_t>(j);
                best_obj = obj_j;
            }
        }
        return {best_j, best_d};
    }

    // Append a fresh element (given object_index/num_child/subtree_radius) to
    // `level`, link it to its NN among the pre-existing elements, and apply the
    // asymmetric re-link (ROS updateTheOtherElements): for every other element e,
    // if delta(e, new) < e.dist, repoint e.outlink to the new element. Rebuilds
    // inlinks for the whole level. The new element's position is level.size()
    // before the call. Returns the new element's position.
    std::uint32_t append_and_link(Level& level, std::uint32_t object_index,
                                  std::uint32_t num_child, double subtree_radius) {
        const std::uint32_t new_pos = static_cast<std::uint32_t>(level.size());
        Element e(object_index);
        e.num_child = num_child;
        e.subtree_radius = subtree_radius;
        level.elements.push_back(std::move(e));

        if (level.size() == 1) {
            level[0].outlink = kNone;
            level[0].dist = 0.0;
            rebuild_inlinks(level);
            return new_pos;
        }

        // Link the new element to its NN among the existing ones.
        auto nn = find_nn(level, new_pos);
        level[new_pos].outlink = nn.first;
        level[new_pos].dist = nn.second;

        // Asymmetric re-link: the new element may now be the NN of others.
        const coord_t* new_vec = vec(object_index);
        for (std::uint32_t p = 0; p < new_pos; ++p) {
            const double d = l2_distance(vec(level[p].object_index), new_vec, dim_);
            // A pre-existing element with one neighbour always has an outlink.
            if (level[p].outlink == kNone || d < level[p].dist ||
                (d == level[p].dist && object_index < level[level[p].outlink].object_index)) {
                level[p].outlink = new_pos;
                level[p].dist = d;
            }
        }
        rebuild_inlinks(level);
        return new_pos;
    }

    // Recompute clusters_[level_idx] from levels_[level_idx]'s current links and
    // refresh the elem_cluster_[level_idx] position->cluster map from the result.
    void recompute_clusters(std::uint32_t level_idx) {
        clusters_[level_idx] = build_level_clusters(level_idx);
        sync_elem_cluster(level_idx);
    }

    // Rebuild elem_cluster_[level_idx] (position -> cluster index) from the
    // current clusters_[level_idx].members. Keeps the maintained inverse map in
    // lock-step with clusters_ on every slow-path recompute. O(level size).
    void sync_elem_cluster(std::uint32_t level_idx) {
        if (elem_cluster_.size() <= level_idx) elem_cluster_.resize(level_idx + 1);
        std::vector<std::uint32_t>& map = elem_cluster_[level_idx];
        map.assign(levels_[level_idx].size(), kNone);
        const std::vector<Cluster>& cls = clusters_[level_idx];
        for (std::uint32_t ci = 0; ci < cls.size(); ++ci) {
            for (std::uint32_t m : cls[ci].members) map[m] = ci;
        }
    }

    // Promoted num_child for a cluster = sum of its members' num_child. The
    // single definition of that rule, shared by every promotion site.
    std::uint32_t sum_num_child(Level& level,
                                const std::vector<std::uint32_t>& members) {
        std::uint32_t n = 0;
        for (std::uint32_t m : members) n += level[m].num_child;
        return n;
    }

    // Recompute one cluster's HDE rep, accumulating radius, and promoted
    // num_child (sum of members' num_child) in place from its current members.
    // Returns the promoted num_child. Used by the incremental fast path.
    std::uint32_t refresh_cluster(std::uint32_t level_idx, Cluster& cl) {
        Level& level = levels_[level_idx];
        cl.rep = select_rep(level, cl.members);
        const coord_t* rep_vec = vec(level[cl.rep].object_index);
        double radius = 0.0;
        std::uint32_t nchild = 0;
        for (std::uint32_t m : cl.members) {
            const double d = (m == cl.rep)
                                 ? 0.0
                                 : l2_distance(rep_vec, vec(level[m].object_index), dim_);
            radius = std::max(radius, d + level[m].subtree_radius);
            nchild += level[m].num_child;
        }
        cl.radius = radius;
        return nchild;
    }

    // After every structural change at level 0, reconcile all higher levels so
    // the whole hierarchy is consistent again. See class header for invariants.
    void reconcile_up(std::uint32_t L) {
        // Root condition: a level with <= 1 element is the top.
        if (levels_[L].size() <= 1) {
            // Truncate any stale levels above L and keep clusters_ aligned with a
            // trailing empty (root) cluster list.
            levels_.resize(L + 1);
            clusters_.resize(L + 1);
            clusters_[L].clear();  // root has no cluster list
            elem_cluster_.resize(L + 1);  // drop any stale levels above new root
            sync_elem_cluster(L);         // root: no cluster list -> all kNone
            return;
        }

        // Recompute this level's clusters from its (now-current) links.
        recompute_clusters(L);
        const std::vector<Cluster>& cls = clusters_[L];

        // Desired level-(L+1) element set: one per cluster.
        struct Desired {
            std::uint32_t object_index;
            std::uint32_t num_child;
            double subtree_radius;
            std::uint32_t cluster_index;  // index into clusters_[L]
        };
        std::vector<Desired> desired;
        desired.reserve(cls.size());
        for (std::size_t ci = 0; ci < cls.size(); ++ci) {
            const Cluster& cl = cls[ci];
            desired.push_back(Desired{levels_[L][cl.rep].object_index,
                                      sum_num_child(levels_[L], cl.members),
                                      cl.radius, static_cast<std::uint32_t>(ci)});
        }

        // Ensure level L+1 + its cluster slot exist.
        if (levels_.size() <= static_cast<std::size_t>(L) + 1) {
            levels_.emplace_back();
            clusters_.emplace_back();
        }
        Level& up = levels_[L + 1];

        // ----- Reconcile the CURRENT up-level element SET to the desired set,
        // keyed by object_index. Removals + additions change the element set (and
        // therefore the links); survivors keep their links (only fields update).

        // Desired object_index set.
        std::unordered_map<std::uint32_t, const Desired*> desired_by_obj;
        desired_by_obj.reserve(desired.size() * 2);
        for (const Desired& d : desired) desired_by_obj.emplace(d.object_index, &d);

        // (1) REMOVE up-elements whose object_index is not desired anymore.
        // Collect positions to remove, then compact + relink (handles the
        // position remapping all at once).
        std::vector<char> remove_mask(up.size(), 0);
        bool any_removed = false;
        for (std::uint32_t p = 0; p < up.size(); ++p) {
            if (desired_by_obj.find(up[p].object_index) == desired_by_obj.end()) {
                remove_mask[p] = 1;
                any_removed = true;
            }
        }
        if (any_removed) remove_elements_and_relink(up, remove_mask);

        // (2) ADD desired reps not currently present.
        std::unordered_map<std::uint32_t, std::uint32_t> present;  // obj -> pos
        present.reserve(up.size() * 2);
        for (std::uint32_t p = 0; p < up.size(); ++p) present.emplace(up[p].object_index, p);
        bool any_added = false;
        for (const Desired& d : desired) {
            if (present.find(d.object_index) == present.end()) {
                std::uint32_t pos = append_and_link(up, d.object_index, d.num_child,
                                                    d.subtree_radius);
                present.emplace(d.object_index, pos);
                any_added = true;
            }
        }

        // (3) UPDATE survivors' num_child / subtree_radius in place. Track
        // whether any (num_child, subtree_radius) actually changed: if nothing
        // above this level can see a difference, the recursion is a no-op.
        bool any_field_changed = false;
        for (const Desired& d : desired) {
            std::uint32_t pos = present[d.object_index];
            if (up[pos].num_child != d.num_child ||
                up[pos].subtree_radius != d.subtree_radius) {
                any_field_changed = true;
            }
            up[pos].num_child = d.num_child;
            up[pos].subtree_radius = d.subtree_radius;
        }

        // (4) Wire heads_cluster (up-element -> its cluster in clusters_[L]) and
        // cluster.parent (cluster -> its rep's up-element position). Cluster
        // indices changed on recompute, so this MUST run even on the early-exit
        // path (the links above are otherwise unchanged).
        for (const Desired& d : desired) {
            std::uint32_t pos = present[d.object_index];
            up[pos].heads_cluster = d.cluster_index;
            clusters_[L][d.cluster_index].parent = pos;
        }

        // Early-exit: if the level-(L+1) element SET is unchanged (no adds, no
        // removes) AND no surviving up-element's (num_child, subtree_radius)
        // changed, then nothing at level L+1 or above can differ from before
        // this op -- its links, clusters, reps, radii and all higher levels are
        // identical. Skip the upward recursion (we already re-wired the
        // heads_cluster/parent indices above, which is all that the recompute of
        // clusters_[L] invalidated). Preserves exactness: the search reads only
        // members/radius/heads_cluster, all of which are now consistent.
        if (!any_removed && !any_added && !any_field_changed) {
            return;
        }

        reconcile_up(L + 1);
    }

    // Remove the elements flagged in `remove_mask` from `level`, compacting the
    // element vector and remapping all outlink/inlink positions. Any surviving
    // element whose outlink pointed at a removed element is re-linked to its NN
    // among the survivors. Rebuilds inlinks. `remove_mask.size() == level.size()`.
    void remove_elements_and_relink(Level& level, const std::vector<char>& remove_mask) {
        const std::size_t n = level.size();
        // old position -> new position (kNone if removed).
        std::vector<std::uint32_t> remap(n, kNone);
        std::vector<Element> kept;
        kept.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (!remove_mask[i]) {
                remap[i] = static_cast<std::uint32_t>(kept.size());
                kept.push_back(level.elements[i]);  // copy; links remapped below
            }
        }
        level.elements = std::move(kept);

        // Remap surviving outlinks; relink any whose target was removed.
        const std::size_t m = level.size();
        for (std::size_t i = 0; i < m; ++i) {
            std::uint32_t out = level[i].outlink;
            if (out == kNone) continue;
            std::uint32_t new_out = remap[out];
            if (new_out != kNone) {
                level[i].outlink = new_out;  // target survived; position shifts
            } else {
                // Target was removed: recompute NN among the survivors.
                if (m <= 1) {
                    level[i].outlink = kNone;
                    level[i].dist = 0.0;
                } else {
                    auto nn = find_nn(level, static_cast<std::uint32_t>(i));
                    level[i].outlink = nn.first;
                    level[i].dist = nn.second;
                }
            }
        }
        if (m == 1) {
            level[0].outlink = kNone;
            level[0].dist = 0.0;
        }
        rebuild_inlinks(level);
    }

    // ----------------------------------------------------- insert (incremental)
    void insert_incremental(std::uint32_t new_id) {
        if (levels_.empty()) {
            // First object: a one-element level-0 root.
            Level level0;
            level0.elements.emplace_back(new_id);
            level0[0].outlink = kNone;
            level0[0].dist = 0.0;
            levels_.push_back(std::move(level0));
            clusters_.emplace_back();  // root: empty cluster list
            elem_cluster_.clear();
            sync_elem_cluster(0);  // 1-element root: no cluster list -> [kNone]
            return;
        }
        // Level-0 incremental link maintenance, ported from the author's
        // ghlim/hnng implementation:
        //   1. Find the new point's nearest level-0 neighbour via the EXACT
        //      hierarchical search (best-first BnB), not an O(n) linear scan.
        //   2. CLUSTER-OVERLAP-PRUNED asymmetric re-link: only inspect level-0
        //      clusters whose bounding sphere can possibly contain an element
        //      now closer to `new` than its current NN (triangle inequality).
        // This stays EXACT: the search returns the true 1-NN, and a looser
        // (pruned) re-link only leaves some outlinks "stale" -- cluster radii
        // accumulate from those links and can only GROW, preserving coverage +
        // radius validity (the two invariants the top-down search relies on).
        Level& level0 = levels_[0];
        const coord_t* new_vec = vec(new_id);

        // (1) Exact NN of the new point over the CURRENT hierarchy (BEFORE the
        //     new element is appended). begin_query_epoch() primes the flat
        //     distance cache; the member distances it fills are reused in (3).
        begin_query_epoch();
        auto [nn_pos, nn_dist] = nn_pos_via_hierarchy(new_vec);

        // (2) Append the new leaf and link it to its NN.
        const std::uint32_t new_pos = static_cast<std::uint32_t>(level0.size());
        Element ne(new_id);
        ne.num_child = 1;
        ne.subtree_radius = 0.0;
        ne.outlink = nn_pos;
        ne.dist = nn_dist;
        level0.elements.push_back(std::move(ne));

        // (3) Cluster-pruned asymmetric re-link over the CURRENT clusters_[0]
        //     (which still reflect the pre-insert state -- the new element is not
        //     a member of any cluster yet). Returns whether any EXISTING element
        //     repointed its outlink to `new` (a "rewire"). The query vector is
        //     vec(new_id) and the epoch from (1) is still active, so delta()
        //     reuses distances already computed in the NN search.
        const bool any_rewire =
            pruned_relink(new_vec, new_pos, nn_dist, level0);

        // (4)+(5) Reconcile the hierarchy upward.
        if (any_rewire) {
            // SLOW PATH: an existing element's outlink changed to the new node. A
            // rewire can merge or split level-0 connected components, so the exact
            // way to re-cluster is a full recompute of clusters_[0] (and the
            // upward reconcile). Recompute ALL level-0 inlinks from the updated
            // outlinks first.
            rebuild_inlinks(level0);
            reconcile_up(0);
            return;
        }

        // FAST PATH: no existing element rewired -- the new element merely
        // attached to its NN. Its single undirected edge joins exactly ONE
        // existing level-0 cluster (the one holding nn_pos); no components merge
        // or split. The only inlink change is `new` -> nn_pos (and `new` starts
        // with no inlinks), an O(1) edit. Grow that one cluster and propagate
        // only the affected spine upward, skipping the O(n) connected-components
        // recompute that dominated the per-insert cost.
        if (nn_pos != kNone) level0[nn_pos].inlinks.push_back(new_pos);
        insert_fast_path(new_pos, nn_pos);
    }

    // No-rewire fast path: the new level-0 element at `new_pos` attached to its
    // NN at `nn_pos` without changing any other element's outlink. Grow the
    // single level-0 cluster that contains `nn_pos` to include `new_pos`, refresh
    // that cluster (rep / radius / promoted num_child), then propagate the change
    // up the spine without recomputing whole levels. Preserves exactness
    // (coverage + accumulating radii) just like the slow path.
    void insert_fast_path(std::uint32_t new_pos, std::uint32_t nn_pos) {
        // A single-element level 0 has no cluster list; the new element makes it
        // size 2, which must become the first real cluster. Fall back to the
        // (cheap, n==2) slow reconcile to create clusters_[0] and the root.
        if (clusters_[0].empty()) {
            reconcile_up(0);
            return;
        }

        const std::uint32_t c0 = elem_cluster_[0][nn_pos];
        Cluster& cl = clusters_[0][c0];

        // Insert new_pos into members keeping the ascending-object_index order
        // that connected_components / the slow path maintain.
        const std::uint32_t new_obj = levels_[0][new_pos].object_index;
        auto it = std::lower_bound(
            cl.members.begin(), cl.members.end(), new_obj,
            [&](std::uint32_t m, std::uint32_t obj) {
                return levels_[0][m].object_index < obj;
            });
        cl.members.insert(it, new_pos);
        if (elem_cluster_[0].size() <= new_pos) elem_cluster_[0].resize(new_pos + 1, kNone);
        elem_cluster_[0][new_pos] = c0;

        propagate_spine(0, c0);
    }

    // Propagate a single changed cluster up the spine. clusters_[L][ci] just had
    // its membership/links perturbed; recompute its (rep, radius, num_child) and
    // reflect the change in the level-(L+1) element it promotes to, recursing.
    // No connected-components, no whole-level recompute.
    void propagate_spine(std::uint32_t L, std::uint32_t ci) {
        for (;;) {
            Cluster& cl = clusters_[L][ci];
            const std::uint32_t old_rep_obj = levels_[L][cl.rep].object_index;
            const std::uint32_t nchild = refresh_cluster(L, cl);
            const std::uint32_t new_rep_obj = levels_[L][cl.rep].object_index;

            // Top of the spine: level L is the root (no level above to update).
            if (L + 1 >= levels_.size()) return;

            if (new_rep_obj != old_rep_obj) {
                // The promoted element's IDENTITY changed (the field bump flipped
                // the HDE winner). The level-(L+1) element SET changes, so fall
                // back to the exact full reconcile from this level. Rarer; we
                // trade speed for guaranteed correctness here.
                reconcile_up(L);
                return;
            }

            // Rep object unchanged: update the matching level-(L+1) element in
            // place (it sits at cl.parent) and then perturb ITS cluster.
            const std::uint32_t up_pos = cl.parent;
            Level& up = levels_[L + 1];
            const bool field_changed = (up[up_pos].num_child != nchild ||
                                        up[up_pos].subtree_radius != cl.radius);
            up[up_pos].num_child = nchild;
            up[up_pos].subtree_radius = cl.radius;

            // If the promoted (num_child, subtree_radius) is unchanged, nothing
            // above can differ -- stop.
            if (!field_changed) return;

            // The up element's field change perturbs its OWN cluster at L+1.
            const std::uint32_t up_ci = elem_cluster_[L + 1][up_pos];
            if (up_ci == kNone) return;  // up element is the root: no cluster
            L = L + 1;
            ci = up_ci;
        }
    }

    // ----------------------------------------------------- remove (incremental)
    void remove_incremental(std::uint32_t object_id) {
        // Find the removed object's level-0 position.
        Level& level0 = levels_[0];
        std::uint32_t rpos = kNone;
        for (std::uint32_t p = 0; p < level0.size(); ++p) {
            if (level0[p].object_index == object_id) { rpos = p; break; }
        }
        if (rpos == kNone) return;  // shouldn't happen (caller validated)

        std::vector<char> remove_mask(level0.size(), 0);
        remove_mask[rpos] = 1;
        remove_elements_and_relink(level0, remove_mask);

        if (level0.empty()) {
            levels_.clear();
            clusters_.clear();
            elem_cluster_.clear();
            return;
        }
        reconcile_up(0);
    }

    // ---------------------------------------------- cluster-pruned relink
    // Asymmetric re-link (ROS updateTheOtherElements), done as a HIERARCHICAL
    // descent instead of a flat O(num_level0_clusters) scan. For every level-0
    // element `m` that is now strictly closer to `new` than to its current NN
    // (d(new,m) < m.dist), repoint m.outlink to `new_pos`. Returns whether any
    // existing element was repointed (a "rewire").
    //
    // The flat version inspects a level-0 cluster `c` iff d(new,c.rep) < nn_dist
    // + c.radius. We reproduce EXACTLY that set of inspected clusters, but find
    // them by descending the cluster tree and pruning whole subtrees that cannot
    // contain such a cluster. For a level-L element `elem` (L>=1) heading a
    // subtree whose leaves all lie within elem.subtree_radius of it, every
    // level-0 cluster `c` under it has d(new,c.rep) >= d(new,elem) -
    // subtree_radius and c.radius <= subtree_radius (radii accumulate upward).
    // Hence if d(new,elem) - subtree_radius >= nn_dist + subtree_radius, i.e.
    //   d(new,elem) >= nn_dist + 2*elem.subtree_radius,
    // NO cluster under `elem` can satisfy the flat inspection test -- the whole
    // subtree is pruned. This never skips a cluster the flat loop would inspect,
    // so the resulting links are identical to the flat relink. (Relinking is in
    // any case exactness-optional: a missed relink only leaves a larger outlink
    // dist, which can only grow radii, preserving coverage + radius validity.)
    bool pruned_relink(const coord_t* new_vec, std::uint32_t new_pos,
                       double nn_dist, Level& level0) {
        bool any_rewire = false;
        const std::uint32_t top_level =
            static_cast<std::uint32_t>(levels_.size() - 1);

        // Apply the relink test to one level-0 cluster (members already exclude
        // new_pos: the new element is not a member of any cluster yet).
        auto relink_cluster = [&](const Cluster& c) {
            if (c.members.empty()) return;
            const double d_rep = delta(new_vec, level0[c.rep].object_index);
            if (d_rep - c.radius >= nn_dist) return;  // same prune as the flat loop
            for (std::uint32_t m : c.members) {
                const double d = (m == c.rep)
                                     ? d_rep
                                     : delta(new_vec, level0[m].object_index);
                if (d < level0[m].dist) {
                    level0[m].outlink = new_pos;
                    level0[m].dist = d;
                    any_rewire = true;
                }
            }
        };

        // Single-level hierarchy: level 0 IS the top; clusters_[0] may be empty
        // (n==1) -- nothing to relink against -- or hold the lone cluster.
        if (top_level == 0) {
            for (const Cluster& c : clusters_[0]) relink_cluster(c);
            return any_rewire;
        }

        // Descend from the top. Stack holds (level, pos) of internal elements
        // not yet pruned. A level-1 element heads a level-0 cluster
        // (clusters_[0][heads_cluster]); higher elements are expanded into their
        // child cluster members.
        std::vector<std::pair<std::uint32_t, std::uint32_t>> stack;
        const Level& top = levels_[top_level];
        for (std::uint32_t pos = 0; pos < top.size(); ++pos) {
            stack.emplace_back(top_level, pos);
        }
        while (!stack.empty()) {
            const auto [level, pos] = stack.back();
            stack.pop_back();
            const Element& elem = levels_[level][pos];
            const double d_elem = delta(new_vec, elem.object_index);
            // Prune the whole subtree if it cannot contain an inspectable cluster.
            if (d_elem - elem.subtree_radius >= nn_dist + elem.subtree_radius) {
                continue;
            }
            if (level == 1) {
                // This element heads a level-0 cluster: inspect it directly.
                relink_cluster(clusters_[0][elem.heads_cluster]);
                continue;
            }
            // Internal: push its child cluster's members (level-1 .. down).
            const Cluster& cl = clusters_[level - 1][elem.heads_cluster];
            for (std::uint32_t m : cl.members) stack.emplace_back(level - 1, m);
        }
        return any_rewire;
    }

    // -------------------------------------------------- exact hierarchical NN
    // Find the nearest LEVEL-0 element to a prepared query `q` by running the
    // SAME best-first branch-and-bound as knn_single for k=1, returning that
    // element's level-0 array POSITION and its distance. Reuses the per-query
    // flat distance cache (delta()); the caller must have started a fresh epoch
    // via begin_query_epoch() so the cached member distances are available for
    // the subsequent cluster-pruned re-link. The hierarchy must be non-empty.
    //
    // This replaces the O(n) linear scan the incremental insert used to do to
    // locate the new point's NN: the hierarchy's coverage + accumulating radii
    // make the pruned descent exact (returns the TRUE 1-NN at level 0).
    std::pair<std::uint32_t, double> nn_pos_via_hierarchy(const coord_t* q) {
        const std::uint32_t top_level = static_cast<std::uint32_t>(levels_.size() - 1);

        MinClusterPQ pq;
        std::uint64_t counter = 0;

        std::uint32_t best_pos = kNone;
        double best_dist = std::numeric_limits<double>::infinity();
        std::uint32_t best_obj = 0;

        // Seed with the top level's elements.
        const Level& top = levels_[top_level];
        for (std::uint32_t pos = 0; pos < top.size(); ++pos) {
            const double d = delta(q, top[pos].object_index);
            const double lb = std::max(0.0, d - top[pos].subtree_radius);
            pq.push(lb, counter++, top_level, pos);
        }

        // If the hierarchy is a single level, the seeds are already the level-0
        // leaves: pick the best among them (top == 0 handled by the loop below,
        // since their lower bound == exact distance and we update best on pop).
        while (!pq.empty()) {
            const PQEntry e = pq.pop();
            const double lb = e.lower_bound;
            const std::uint32_t level = e.level;
            const std::uint32_t pos = e.pos;

            // Best-first optimality: nothing remaining can beat the current best.
            if (best_pos != kNone && lb > best_dist) break;

            const Element& elem = levels_[level][pos];

            if (level == 0) {
                // Leaf popped from the PQ (only reachable as a top-level seed in
                // the single-level case): its lower bound is its exact distance.
                const double d = delta(q, elem.object_index);
                if (best_pos == kNone || d < best_dist ||
                    (d == best_dist && elem.object_index < best_obj)) {
                    best_dist = d;
                    best_pos = pos;
                    best_obj = elem.object_index;
                }
                continue;
            }

            // Internal element: expand into the members of the cluster it heads.
            const Cluster& cl = clusters_[level - 1][elem.heads_cluster];
            const Level& child_level = levels_[level - 1];
            if (level == 1) {
                // Children are level-0 LEAVES: a leaf's lower bound equals its
                // exact distance, so update best directly instead of routing it
                // through the PQ (halves heap traffic on the dominant level).
                for (std::uint32_t m : cl.members) {
                    const Element& child = child_level[m];
                    const double d = delta(q, child.object_index);
                    if (best_pos == kNone || d < best_dist ||
                        (d == best_dist && child.object_index < best_obj)) {
                        best_dist = d;
                        best_pos = m;
                        best_obj = child.object_index;
                    }
                }
                continue;
            }
            for (std::uint32_t m : cl.members) {
                const Element& child = child_level[m];
                const double d = delta(q, child.object_index);
                const double child_lb = std::max(0.0, d - child.subtree_radius);
                if (best_pos != kNone && child_lb > best_dist) continue;
                pq.push(child_lb, counter++, level - 1, m);
            }
        }
        return {best_pos, best_dist};
    }

    // -------------------------------------------------- exact k-NN search
    // Best-first branch-and-bound over the hierarchy for one prepared query `q`.
    // Returns up to k object indices ordered by (distance, object_index).
    // Mirrors hnng_ref._knn_single exactly, including the per-query memoization
    // and the (dist, object_index) tie-break.
    // Best-first BnB writing up to k sorted object indices into `out_row`
    // (right-padded with -1). All per-query state lives in the caller-owned
    // `s` (distance cache, result heap, cluster PQ, counters), cleared via
    // s.begin() before each query -- so concurrent queries on distinct scratches
    // are independent and thread-safe (Phase 6). Single query implementation
    // used by both the sequential and the parallel batch paths.
    void knn_single(const coord_t* q, std::size_t k, QueryScratch& s,
                    std::int64_t* out_row) {
        for (std::size_t j = 0; j < k; ++j) out_row[j] = -1;
        if (levels_.empty()) return;

        std::uint64_t counter = 0;

        const std::uint32_t top_level = static_cast<std::uint32_t>(levels_.size() - 1);

        // Seed with the top level's elements.
        const Level& top = levels_[top_level];
        for (std::uint32_t pos = 0; pos < top.size(); ++pos) {
            const double d = delta_s(q, top[pos].object_index, s);
            const double lb = std::max(0.0, d - top[pos].subtree_radius);
            s.pq.push(lb, counter++, top_level, pos);
        }

        while (!s.pq.empty()) {
            const PQEntry e = s.pq.pop();
            const double lb = e.lower_bound;
            const std::uint32_t level = e.level;
            const std::uint32_t pos = e.pos;

            // Best-first optimality: nothing remaining can improve the result.
            if (s.results.full() && lb > s.results.kth_distance()) break;

            const Element& elem = levels_[level][pos];

            if (level == 0) {
                // Leaf: lower bound equals delta(q, leaf) (radius 0). The
                // distance was already computed (cached) when this leaf was
                // pushed, so delta_s() returns it without a new evaluation.
                const double d = delta_s(q, elem.object_index, s);
                s.results.offer(d, elem.object_index);
                continue;
            }

            // Internal element: expand into the members of the cluster it heads.
            ++s.clusters_visited;
            const Cluster& cl = clusters_[level - 1][elem.heads_cluster];
            const Level& child_level = levels_[level - 1];
            const double kth = s.results.kth_distance();
            const bool have_k = s.results.full();
            for (std::uint32_t m : cl.members) {
                const Element& child = child_level[m];
                const double d = delta_s(q, child.object_index, s);
                const double child_lb = std::max(0.0, d - child.subtree_radius);
                // Prune branches that cannot beat the current kth distance.
                if (have_k && child_lb > kth) continue;
                s.pq.push(child_lb, counter++, level - 1, m);
            }
        }

        s.results.sort_in_place();
        const std::vector<ResultEntry>& sorted = s.results.entries();
        for (std::size_t j = 0; j < sorted.size() && j < k; ++j) {
            out_row[j] = static_cast<std::int64_t>(sorted[j].object_index);
        }
    }

    // -------------------------------------------------- data members
    Metric metric_;
    HdeRule hde_rule_ = HdeRule::V2026;
    std::size_t dim_ = 0;
    std::vector<coord_t> data_;          // prepared, row-major (num_stored, dim_)
    std::vector<std::uint32_t> active_;  // object ids currently in the hierarchy

    std::vector<Level> levels_;                  // levels_[L].elements
    std::vector<std::vector<Cluster>> clusters_; // clusters_[L] (empty for root)

    // elem_cluster_[L][pos] = index into clusters_[L] of the cluster that holds
    // element `pos` at level L (kNone where level L has no cluster list, e.g. the
    // root, or for a position not yet assigned). A maintained inverse of
    // cluster.members that lets the incremental fast path locate, in O(1), the
    // cluster a newly inserted element joins -- avoiding the O(n)
    // connected-components recompute that dominated per-insert cost. Whenever a
    // level's cluster list is (re)built it is refilled through the single
    // sync_elem_cluster() choke-point; the incremental fast path is the one
    // exception, patching a single slot in O(1).
    std::vector<std::vector<std::uint32_t>> elem_cluster_;

    // Per-query instrumentation + memoization (mirrors the reference).
    std::size_t last_query_dist_evals_ = 0;
    std::size_t last_query_clusters_visited_ = 0;
    // Flat O(1) per-query distance cache (replaces the per-query
    // std::unordered_map): one value slot + one epoch stamp per stored object,
    // and a current-epoch counter bumped each query. A slot is live for the
    // current query iff its stamp == cur_epoch_.
    std::vector<double> delta_cache_val_;
    std::vector<std::uint32_t> delta_cache_epoch_;
    std::uint32_t cur_epoch_ = 0;
};

}  // namespace hnng

#endif  // HNNG_HNNG_HPP
