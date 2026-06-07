// smoke_main.cpp -- a tiny no-Python sanity check for the hNNG core.
//
// Compiles and is tested (g++ 13.3; CI runs the suite). Build with:
//   cmake -S . -B build -DHNNG_BUILD_SMOKE=ON && cmake --build build
//   ./build/hnng_smoke
//
// This is NOT a correctness oracle (that is the Python parity suite against
// hnng_ref). It only checks the core builds an index and returns k indices with
// the nearest being the query point itself when the query is a dataset point.

#include <cstdio>
#include <random>
#include <vector>

#include "hnng/hnng.hpp"

int main() {
    const std::size_t n = 500;
    const std::size_t d = 8;
    const std::size_t k = 10;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> data(n * d);
    for (auto& x : data) x = nd(rng);

    hnng::HNNG idx(hnng::Metric::L2);
    idx.build(data.data(), n, d);

    // Query with a dataset point: the nearest neighbour must be itself (dist 0).
    const std::size_t qi = 123;
    std::vector<std::int64_t> out(k, -1);
    idx.knn_query_batch(data.data() + qi * d, 1, d, k, out.data());

    bool self_first = (out[0] == static_cast<std::int64_t>(qi));
    int valid = 0;
    for (auto v : out) if (v != -1) ++valid;

    auto s = idx.stats();
    std::printf("height=%zu num_levels=%zu avg_cluster_size=%.3f\n", s.height,
                s.num_levels, s.avg_cluster_size);
    std::printf("query #%zu -> first=%lld valid=%d dist_evals=%zu (n=%zu)\n", qi,
                static_cast<long long>(out[0]), valid, s.last_query_dist_evals, n);

    if (!self_first) {
        std::printf("FAIL: nearest neighbour of a dataset point was not itself\n");
        return 1;
    }
    if (valid != static_cast<int>(k)) {
        std::printf("FAIL: expected %zu valid neighbours, got %d\n", k, valid);
        return 1;
    }
    if (s.last_query_dist_evals >= n) {
        std::printf("WARN: no pruning (dist_evals >= n); ok for adversarial data\n");
    }
    std::printf("OK\n");
    return 0;
}
