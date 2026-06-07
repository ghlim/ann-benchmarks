// heap.hpp -- the two priority structures the best-first BnB search needs:
//
//   * MinClusterPQ : a min-priority-queue of (lower_bound, counter, level, pos)
//                    entries, popped in ascending lower_bound order. Mirrors the
//                    reference's `heapq` of (lb, next(counter), level, pos): the
//                    monotonically increasing `counter` is a FIFO-ish tiebreak
//                    that guarantees a total order so heap entries never have to
//                    compare the trailing fields. We reproduce that exact order.
//
//   * BoundedMaxResults : the bounded max-heap of the k best leaves. The
//                    reference stores (-dist, object_index) in a min-heap so the
//                    current WORST (kth) candidate sits at the top and is cheap
//                    to evict. We model the same: the heap root is the current
//                    kth-nearest, ordered by (dist desc, object_index desc) so
//                    that on a distance tie the LARGER object_index is evicted
//                    first -- exactly what `(-dist, object_index)` min-heap does.
//
// Compiles and is tested (g++ 13.3 / pybind11 3.0.4; CI runs the suite).

#ifndef HNNG_HEAP_HPP
#define HNNG_HEAP_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace hnng {

// ---------------------------------------------------------------- cluster PQ
struct PQEntry {
    double lower_bound;
    std::uint64_t counter;  // strictly increasing insertion order (tiebreak)
    std::uint32_t level;
    std::uint32_t pos;
};

// Comparator for a std::priority_queue-style MAX-heap that we *invert* to behave
// as a MIN-heap on (lower_bound, counter). std::push_heap/pop_heap keep the
// LARGEST element (per `comp`) at front, so to pop the smallest (lb, counter)
// first we make `comp` return true when `a` should come AFTER `b`.
struct PQGreater {
    bool operator()(const PQEntry& a, const PQEntry& b) const {
        if (a.lower_bound != b.lower_bound) return a.lower_bound > b.lower_bound;
        return a.counter > b.counter;  // counters are unique, so this is total
    }
};

// A thin min-heap over PQEntry using std::*_heap on a vector (so we control the
// exact comparator and can read it during review). Same semantics as Python's
// heapq over (lb, counter, level, pos).
class MinClusterPQ {
public:
    bool empty() const { return data_.empty(); }
    std::size_t size() const { return data_.size(); }

    void push(double lb, std::uint64_t counter, std::uint32_t level, std::uint32_t pos) {
        data_.push_back(PQEntry{lb, counter, level, pos});
        std::push_heap(data_.begin(), data_.end(), PQGreater{});
    }

    // Returns and removes the entry with the smallest (lower_bound, counter).
    PQEntry pop() {
        std::pop_heap(data_.begin(), data_.end(), PQGreater{});
        PQEntry e = data_.back();
        data_.pop_back();
        return e;
    }

    // The smallest (lower_bound, counter) without removing it.
    const PQEntry& top() const { return data_.front(); }

    // Reset for reuse across queries WITHOUT releasing the backing capacity, so
    // a batch of queries reuses one allocation instead of malloc/free per query.
    void clear() { data_.clear(); }

private:
    std::vector<PQEntry> data_;
};

// ---------------------------------------------------------- bounded k-results
struct ResultEntry {
    double dist;
    std::uint32_t object_index;
};

// Comparator that places at the heap ROOT the SAME entry the reference's
// `(-dist, object_index)` min-heap surfaces as results[0]. This matters because
// the reference's eviction test reads results[0][1] (the root's object_index):
//
//   worst = -results[0][0]                  # the largest dist
//   if dist < worst or (dist == worst and object_index < results[0][1]): replace
//
// A Python min-heap of (-dist, object_index) keeps the SMALLEST such tuple at
// root: smallest -dist == LARGEST dist, and on a dist tie the smaller tuple has
// the SMALLER object_index. So the root is (largest dist, smallest object_index).
//
// std::push_heap/pop_heap keep the element that is GREATEST per `comp` at front.
// We want front = (largest dist, smallest object_index), so define `comp(a, b)`
// (a ordered before b, i.e. a is "less") as:
//   a.dist < b.dist, or (a.dist == b.dist and a.object_index > b.object_index).
// Then the front is the max: largest dist, and on a tie the smallest obj. This
// reproduces the reference's results[0] exactly, so the eviction tie rule
// (object_index < root.object_index) is identical.
struct ResultLess {
    bool operator()(const ResultEntry& a, const ResultEntry& b) const {
        if (a.dist != b.dist) return a.dist < b.dist;
        return a.object_index > b.object_index;  // tie: larger obj is "less"
    }
};

// Bounded max-heap keeping the k best (smallest-distance) leaves seen so far.
class BoundedMaxResults {
public:
    explicit BoundedMaxResults(std::size_t k) : k_(k) { data_.reserve(k + 1); }

    std::size_t size() const { return data_.size(); }
    bool full() const { return data_.size() >= k_; }

    // Current kth distance: +inf until we have k candidates, else the root dist.
    double kth_distance() const {
        if (data_.size() < k_) return std::numeric_limits<double>::infinity();
        return data_.front().dist;  // root = largest dist among the k kept
    }

    // Offer a candidate leaf. Mirrors hnng_ref `_knn_single.offer`:
    //   if len(results) < k: push
    //   else if dist < worst or (dist == worst and obj < worst_obj): replace root
    void offer(double dist, std::uint32_t object_index) {
        if (data_.size() < k_) {
            data_.push_back(ResultEntry{dist, object_index});
            std::push_heap(data_.begin(), data_.end(), ResultLess{});
            return;
        }
        const ResultEntry& worst = data_.front();
        const bool better = (dist < worst.dist) ||
                            (dist == worst.dist && object_index < worst.object_index);
        if (better) {
            std::pop_heap(data_.begin(), data_.end(), ResultLess{});
            data_.back() = ResultEntry{dist, object_index};
            std::push_heap(data_.begin(), data_.end(), ResultLess{});
        }
    }

    // Return the kept entries sorted ascending by (dist, object_index).
    std::vector<ResultEntry> sorted_ascending() const {
        std::vector<ResultEntry> out = data_;
        std::sort(out.begin(), out.end(), [](const ResultEntry& a, const ResultEntry& b) {
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.object_index < b.object_index;
        });
        return out;
    }

    // Reset for reuse across queries, keeping backing capacity (no realloc).
    void clear() { data_.clear(); }

    // Sort the kept entries ascending by (dist, object_index) IN PLACE (destroys
    // the heap order; callers do this only after the search, before clear()).
    // Lets the caller read results without allocating a sorted copy per query.
    void sort_in_place() {
        std::sort(data_.begin(), data_.end(), [](const ResultEntry& a, const ResultEntry& b) {
            if (a.dist != b.dist) return a.dist < b.dist;
            return a.object_index < b.object_index;
        });
    }
    const std::vector<ResultEntry>& entries() const { return data_; }

private:
    std::size_t k_;
    std::vector<ResultEntry> data_;
};

}  // namespace hnng

#endif  // HNNG_HEAP_HPP
