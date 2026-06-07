#pragma once

#include "hnnglib.h"
#include "visited_list_pool.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace hnnglib {

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

static constexpr tableint INVALID_ID = static_cast<tableint>(-1);
static constexpr int32_t  INVALID_CLUSTER = -1;

#ifdef HNNG_VERIFY
#define HNNG_VERIFY_CHECK(cond, msg) \
    do { if (!(cond)) throw std::runtime_error(std::string("HNNG_VERIFY: ") + (msg)); } while (0)
#else
#define HNNG_VERIFY_CHECK(cond, msg) ((void)0)
#endif


// Hierarchical Nearest Neighbor Graph (Lim et al. 2015 ICONIP; 2026 manuscript for exact NN search).
//
// Storage model (hnswlib-style hot path):
//   - Per element at level 0: contiguous block in data_level0_memory_ holding
//       [outlink_id : tableint]
//       [outlink_dist : dist_t]
//       [inlink_count : linklistsizeint]
//       [data : dim*sizeof(dist_t)]
//       [label : labeltype]
//   - Per element at level 0: inlink_lists_[id] is a std::vector<tableint>*, nullptr until first inlink.
//   - Higher levels (sparse — only reps appear):
//       outlinks_per_level_[level][id] : tableint
//       outlink_dist_per_level_[level][id] : dist_t
//       inlinks_per_level_[level][id] : std::vector<tableint>
//     For element id, level l > 0 fields exist iff element_levels_[id] >= l.
//
// Clusters:
//   clusters_[level] is a std::vector<Cluster>. A Cluster carries its rep, members, radius,
//   parent representative at level+1, and a c_status flag for the level-promotion diff.
//   elem_to_cluster_[level][id] = cluster index at that level (INVALID_CLUSTER if absent).
//
template<typename dist_t>
class HierarchicalNNG : public AlgorithmInterface<dist_t> {
 public:
    struct Cluster {
        tableint              rep_id        = INVALID_ID;
        std::vector<tableint> member_ids;
        dist_t                radius        = 0;
        tableint              parent_elem_id = INVALID_ID;
        uint8_t               c_status      = 0;  // NOC=0, ADD=1, DEL=2, UPD=3
    };

    HierarchicalNNG(SpaceInterface<dist_t>* s, size_t max_elements);
    HierarchicalNNG(SpaceInterface<dist_t>* s, const std::string& location);
    ~HierarchicalNNG() override;

    // Public API (algorithm bodies arrive in later phases).
    void addPoint(void* data_point, labeltype label) override;
    std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(void* query_data, size_t k) override;

    void saveIndex(const std::string& location) override;
    void loadIndex(const std::string& location, SpaceInterface<dist_t>* s);

    size_t getMaxElements()         const { return max_elements_; }
    size_t getCurrentElementCount() const { return cur_element_count_; }
    int    getMaxLevel()            const { return maxlevel_; }

    // ---- Phase-1 raw accessors used by tests to hand-build a hierarchy ----
    // These bypass the algorithm. They will not be used by addPoint/searchKnn,
    // and they perform no invariant maintenance.
    tableint _test_addRawElement(const void* data, labeltype label, int top_level);
    void     _test_setOutlink(tableint id, int level, tableint target, dist_t dist);
    void     _test_addInlink (tableint id, int level, tableint source);
    int32_t  _test_addCluster(int level, tableint rep_id,
                              const std::vector<tableint>& member_ids,
                              dist_t radius,
                              tableint parent_elem_id);
    void     _test_setRoot   (tableint id) { enterpoint_node_ = id; }

    // Accessors for tests/verification.
    tableint  getOutlink     (tableint id, int level) const;
    dist_t    getOutlinkDist (tableint id, int level) const;
    const std::vector<tableint>& getInlinks(tableint id, int level) const;
    int       getElementLevel(tableint id) const { return element_levels_[id]; }
    labeltype getLabel       (tableint id) const;
    const char* getData      (tableint id) const;

    const std::vector<Cluster>& getClustersAtLevel(int level) const { return clusters_[level]; }
    int32_t   getClusterIndex(tableint id, int level) const;

 protected:
    void init(size_t max_elements);
    void freeAll();
    void ensureLevelCapacity(int level);

    // Phase-3 helpers
    void addInlinkL0_   (tableint owner, tableint source);
    void removeInlinkL0_(tableint owner, tableint source);
    std::vector<tableint> componentOf_(tableint start, int level);
    tableint  selectRepresentative_(const std::vector<tableint>& members, int level);
    dist_t    computeRadiusMetric_ (const Cluster& c, int level);
    void      rebuildClustersAtLevel_(int level, bool compute_radii = true);
    // Phase-4: tear down and rebuild all levels >= 1 by iteratively promoting
    // cluster representatives. Called after every level-0 mutation.
    void      rebuildUpperLevels_(tableint new_elem);
    // Path-only radius refresh: walk from `start` up the hierarchy of
    // containing clusters, refreshing the radius at each level. Used by the
    // fast path in rebuildUpperLevels_ when the rep set hasn't changed.
    void      refreshRadiiAlongPath_(tableint start);
#ifdef HNNG_VERIFY
    void      verifyInvariantsLevel0_();
    void      verifyInvariantsAllLevels_();
#endif

    // Flat-block accessors (level 0 hot path).
    char* level0BlockPtr(tableint id) const {
        return data_level0_memory_ + size_t(id) * size_per_element_;
    }
    tableint&        outlinkId0Ref   (tableint id) { return *reinterpret_cast<tableint*>       (level0BlockPtr(id) + offset_outlink_id_);   }
    dist_t&          outlinkDist0Ref (tableint id) { return *reinterpret_cast<dist_t*>         (level0BlockPtr(id) + offset_outlink_dist_); }
    linklistsizeint& inlinkCount0Ref (tableint id) { return *reinterpret_cast<linklistsizeint*>(level0BlockPtr(id) + offset_inlink_count_); }
    labeltype&       label0Ref       (tableint id) { return *reinterpret_cast<labeltype*>      (level0BlockPtr(id) + offset_label_);        }
    char*            data0Ptr        (tableint id) { return level0BlockPtr(id) + offset_data_; }

    // ---- members ----
    SpaceInterface<dist_t>* space_ = nullptr;
    DISTFUNC<dist_t>        fstdistfunc_ = nullptr;
    void*                   dist_func_param_ = nullptr;
    size_t                  data_size_ = 0;

    size_t max_elements_ = 0;
    size_t cur_element_count_ = 0;

    // Level-0 flat block layout
    char*  data_level0_memory_ = nullptr;
    size_t size_per_element_   = 0;
    size_t offset_outlink_id_    = 0;
    size_t offset_outlink_dist_  = 0;
    size_t offset_inlink_count_  = 0;
    size_t offset_data_          = 0;
    size_t offset_label_         = 0;

    // Level-0 inlinks (nullptr until first inlink)
    std::vector<std::vector<tableint>*> inlink_lists_l0_;

    // Higher-level sparse storage, indexed [level][id].
    // outlinks_per_level_[l] has size max_elements; only meaningful for ids with element_levels_[id] >= l.
    std::vector<std::vector<tableint>> outlinks_per_level_;
    std::vector<std::vector<dist_t>>   outlink_dist_per_level_;
    std::vector<std::vector<std::vector<tableint>>> inlinks_per_level_;

    std::vector<int> element_levels_;
    int      maxlevel_ = -1;
    tableint enterpoint_node_ = INVALID_ID;

    // Cluster store
    std::vector<std::vector<Cluster>> clusters_;
    std::vector<std::vector<int32_t>> elem_to_cluster_;

    // External -> internal id map
    std::unordered_map<labeltype, tableint> label_lookup_;

    // Concurrency
    std::vector<std::mutex> link_list_locks_;
    std::mutex global_lock_;

    std::unique_ptr<VisitedListPool> visited_list_pool_;
};


// ---- Implementation ----

template<typename dist_t>
HierarchicalNNG<dist_t>::HierarchicalNNG(SpaceInterface<dist_t>* s, size_t max_elements) {
    space_ = s;
    fstdistfunc_     = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    data_size_       = s->get_data_size();
    init(max_elements);
}

template<typename dist_t>
HierarchicalNNG<dist_t>::HierarchicalNNG(SpaceInterface<dist_t>* s, const std::string& location) {
    space_ = s;
    fstdistfunc_     = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    data_size_       = s->get_data_size();
    loadIndex(location, s);
}

template<typename dist_t>
HierarchicalNNG<dist_t>::~HierarchicalNNG() { freeAll(); }

template<typename dist_t>
void HierarchicalNNG<dist_t>::init(size_t max_elements) {
    max_elements_ = max_elements;
    cur_element_count_ = 0;

    offset_outlink_id_   = 0;
    offset_outlink_dist_ = offset_outlink_id_   + sizeof(tableint);
    offset_inlink_count_ = offset_outlink_dist_ + sizeof(dist_t);
    offset_data_         = offset_inlink_count_ + sizeof(linklistsizeint);
    offset_label_        = offset_data_         + data_size_;
    size_per_element_    = offset_label_        + sizeof(labeltype);

    data_level0_memory_ = static_cast<char*>(std::malloc(max_elements_ * size_per_element_));
    if (!data_level0_memory_) throw std::runtime_error("hnng: malloc level-0 memory failed");
    std::memset(data_level0_memory_, 0, max_elements_ * size_per_element_);

    inlink_lists_l0_.assign(max_elements_, nullptr);
    element_levels_.assign(max_elements_, 0);

    // Per-level sparse storage starts with level 0 only (which is the flat block).
    // Higher levels are added on demand by ensureLevelCapacity().
    outlinks_per_level_.clear();
    outlink_dist_per_level_.clear();
    inlinks_per_level_.clear();

    // Cluster storage: level 0 always exists.
    clusters_.assign(1, std::vector<Cluster>{});
    elem_to_cluster_.assign(1, std::vector<int32_t>(max_elements_, INVALID_CLUSTER));

    maxlevel_        = -1;
    enterpoint_node_ = INVALID_ID;

    link_list_locks_ = std::vector<std::mutex>(max_elements_);
    visited_list_pool_.reset(new VisitedListPool(1, max_elements_));
}

template<typename dist_t>
void HierarchicalNNG<dist_t>::freeAll() {
    if (data_level0_memory_) {
        std::free(data_level0_memory_);
        data_level0_memory_ = nullptr;
    }
    for (auto* v : inlink_lists_l0_) delete v;
    inlink_lists_l0_.clear();
    outlinks_per_level_.clear();
    outlink_dist_per_level_.clear();
    inlinks_per_level_.clear();
    clusters_.clear();
    elem_to_cluster_.clear();
    label_lookup_.clear();
    visited_list_pool_.reset();
}

template<typename dist_t>
void HierarchicalNNG<dist_t>::ensureLevelCapacity(int level) {
    if (level <= 0) return;
    // Grow per-level structures to cover `level`.
    while (static_cast<int>(outlinks_per_level_.size()) < level) {
        outlinks_per_level_.emplace_back(max_elements_, INVALID_ID);
        outlink_dist_per_level_.emplace_back(max_elements_, dist_t{});
        inlinks_per_level_.emplace_back(max_elements_);
    }
    while (static_cast<int>(clusters_.size()) < level + 1) {
        clusters_.emplace_back();
        elem_to_cluster_.emplace_back(max_elements_, INVALID_CLUSTER);
    }
    if (level > maxlevel_) maxlevel_ = level;
}

// ---- Level-0 link maintenance ----
template<typename dist_t>
void HierarchicalNNG<dist_t>::addInlinkL0_(tableint owner, tableint source) {
    auto*& list_ptr = inlink_lists_l0_[owner];
    if (!list_ptr) list_ptr = new std::vector<tableint>();
    list_ptr->push_back(source);
    inlinkCount0Ref(owner) = static_cast<linklistsizeint>(list_ptr->size());
}

template<typename dist_t>
void HierarchicalNNG<dist_t>::removeInlinkL0_(tableint owner, tableint source) {
    auto* list_ptr = inlink_lists_l0_[owner];
    if (!list_ptr) return;
    auto& v = *list_ptr;
    auto it = std::find(v.begin(), v.end(), source);
    if (it != v.end()) v.erase(it);
    inlinkCount0Ref(owner) = static_cast<linklistsizeint>(v.size());
}


// ---- addPoint: level-0 with cluster-pruned rewire ----
//
// 1. Allocate level-0 block.
// 2. Find NN via the same exact hierarchical searchKnn used at query time.
// 3. Connect new → NN (outlink + reverse inlink).
// 4. Bounded rewire: visit only clusters whose bounding sphere could contain
//    an element that the new point is now closer to than its current NN.
//    The keep test is the same triangle-inequality bound used by searchKnn:
//        M(new, C.rep) - C.radius < nn_dist_metric
//    A loose bound (matching the ROS reference / 2026 manuscript). Misses
//    rewires only for elements whose current outlink_dist > nn_dist; these
//    leave the NNG with stale (correct-for-component-structure but locally
//    suboptimal) outlinks. Safe for search exactness because cluster radii
//    only grow under staleness.
// 5. Rebuild level-0 clusters by component-walking the post-rewire NNG.
// 6. rebuildUpperLevels_ (existing fast path / full rebuild).
//
// Per-insert level-0 work is O(num clusters × 1 dist call to rep) plus
// O(num overlapping clusters × avg cluster size). The first term grows
// linearly with n; the second is bounded by the paper's empirical ~6.
//
template<typename dist_t>
void HierarchicalNNG<dist_t>::addPoint(void* data_point, labeltype label) {
    std::unique_lock<std::mutex> lock(global_lock_);

    if (cur_element_count_ >= max_elements_)
        throw std::runtime_error("hnng: capacity exceeded");
    if (label_lookup_.count(label))
        throw std::runtime_error("hnng: duplicate label");

    tableint new_id = static_cast<tableint>(cur_element_count_++);
    label_lookup_[label] = new_id;

    // Initialize new element at level 0.
    outlinkId0Ref(new_id)   = INVALID_ID;
    outlinkDist0Ref(new_id) = std::numeric_limits<dist_t>::max();
    inlinkCount0Ref(new_id) = 0;
    inlink_lists_l0_[new_id] = nullptr;
    std::memcpy(data0Ptr(new_id), data_point, data_size_);
    label0Ref(new_id)        = label;
    element_levels_[new_id]  = 0;
    elem_to_cluster_[0][new_id] = INVALID_CLUSTER;

    // First element: singleton cluster, no neighbors.
    if (new_id == 0) {
        Cluster c;
        c.rep_id = 0;
        c.member_ids = {0};
        c.radius = 0;
        c.parent_elem_id = INVALID_ID;
        clusters_[0].push_back(std::move(c));
        elem_to_cluster_[0][0] = 0;
        enterpoint_node_ = 0;
        maxlevel_ = 0;
        return;
    }

    // 2. Find NN via hierarchical search.
    tableint nn_id;
    dist_t   nn_dist;
    {
        auto pq = searchKnn(data_point, 1);
        if (pq.empty())
            throw std::runtime_error("hnng: addPoint NN search empty");
        nn_dist = pq.top().first;
        auto it = label_lookup_.find(pq.top().second);
        if (it == label_lookup_.end() || it->second == new_id)
            throw std::runtime_error("hnng: addPoint NN label lookup failed");
        nn_id = it->second;
    }

    // 3. Connect new → NN.
    outlinkId0Ref(new_id)   = nn_id;
    outlinkDist0Ref(new_id) = nn_dist;
    addInlinkL0_(nn_id, new_id);

    // 4. Bounded rewire: visit only clusters whose bounding sphere overlaps
    //    the new point's NN ball. Track rewires so we can decide between a
    //    no-rewire fast path and a structural rebuild.
    bool rewires_happened = false;
    std::vector<tableint> dirty_elems;
    dirty_elems.push_back(new_id);
    dirty_elems.push_back(nn_id);

    const dist_t nn_dist_m = space_->to_metric(nn_dist);
    for (int32_t ci = 0; ci < static_cast<int32_t>(clusters_[0].size()); ++ci) {
        const auto& c = clusters_[0][ci];
        if (c.member_ids.empty()) continue;
        dist_t d_rep   = fstdistfunc_(data_point, getData(c.rep_id), dist_func_param_);
        dist_t d_rep_m = space_->to_metric(d_rep);
        if (d_rep_m - c.radius >= nn_dist_m) continue;  // prune

        const auto members = c.member_ids;
        for (tableint m : members) {
            if (m == nn_id || m == new_id) continue;
            dist_t d = (m == c.rep_id) ? d_rep
                                       : fstdistfunc_(data_point, getData(m), dist_func_param_);
            if (d < outlinkDist0Ref(m)) {
                tableint old_outlink = outlinkId0Ref(m);
                if (old_outlink != INVALID_ID) {
                    removeInlinkL0_(old_outlink, m);
                    dirty_elems.push_back(old_outlink);
                }
                outlinkId0Ref(m)   = new_id;
                outlinkDist0Ref(m) = d;
                addInlinkL0_(new_id, m);
                dirty_elems.push_back(m);
                rewires_happened = true;
            }
        }
    }

    int32_t nn_ci = elem_to_cluster_[0][nn_id];
    if (nn_ci == INVALID_CLUSTER)
        throw std::runtime_error("hnng: NN has no cluster (corrupt state)");

    if (!rewires_happened) {
        // Fast path: only new_id joined NN's cluster.
        auto& c = clusters_[0][nn_ci];
        c.member_ids.push_back(new_id);
        elem_to_cluster_[0][new_id] = nn_ci;
        c.rep_id = selectRepresentative_(c.member_ids, 0);
        c.radius = computeRadiusMetric_(c, 0);
    } else {
        // Slow path: rewires shifted edges between components. Rebuild
        // level-0 cluster structure via component walks. Reuse cached radii
        // for clusters whose members didn't touch any dirty element.
        std::unordered_map<tableint, dist_t> old_radii;
        old_radii.reserve(clusters_[0].size());
        for (const auto& c : clusters_[0]) old_radii[c.rep_id] = c.radius;

        rebuildClustersAtLevel_(0, /*compute_radii=*/false);

        std::vector<bool> dirty_cluster(clusters_[0].size(), false);
        for (tableint e : dirty_elems) {
            int32_t ci = elem_to_cluster_[0][e];
            if (ci != INVALID_CLUSTER) dirty_cluster[ci] = true;
        }
        for (int32_t ci = 0; ci < static_cast<int32_t>(clusters_[0].size()); ++ci) {
            auto& c = clusters_[0][ci];
            if (dirty_cluster[ci]) {
                c.radius = computeRadiusMetric_(c, 0);
            } else {
                auto it = old_radii.find(c.rep_id);
                c.radius = (it != old_radii.end()) ? it->second
                                                   : computeRadiusMetric_(c, 0);
            }
        }
    }

    // 6. Propagate to upper levels.
    rebuildUpperLevels_(new_id);

#ifdef HNNG_VERIFY
    verifyInvariantsAllLevels_();
#endif
}

// ---- rebuild all levels >= 1 ----
// Tear down per-element link state at every higher level and the cluster store
// from level 1 up, then build levels iteratively: at each level the population
// is the set of representatives from the level below.
//
// Fast path: if the level-0 rep set matches the set of elements currently
// at level 1 (i.e., no level-0 structural change since the last build), only
// recompute radii top-down. The level-1+ NNGs and clusters remain valid.
template<typename dist_t>
void HierarchicalNNG<dist_t>::refreshRadiiAlongPath_(tableint start) {
    tableint at_level = start;
    for (int l = 0; l <= maxlevel_; ++l) {
        int32_t ci = elem_to_cluster_[l][at_level];
        if (ci == INVALID_CLUSTER) break;
        clusters_[l][ci].radius = computeRadiusMetric_(clusters_[l][ci], l);
        at_level = clusters_[l][ci].rep_id;
    }
}

template<typename dist_t>
void HierarchicalNNG<dist_t>::rebuildUpperLevels_(tableint new_elem) {
    // Fast path: structural reps unchanged → only refresh radii along the
    // path from new_elem up to the root. Other clusters keep their existing
    // radii (which may be stale-large after rewires; that's safe for the
    // search prune since over-large radii only cause over-inclusion, never
    // incorrect omission).
    if (maxlevel_ >= 1 && !clusters_[0].empty()) {
        std::vector<tableint> l0_reps;
        l0_reps.reserve(clusters_[0].size());
        for (const auto& c : clusters_[0]) l0_reps.push_back(c.rep_id);
        std::sort(l0_reps.begin(), l0_reps.end());

        std::vector<tableint> l1_elems;
        l1_elems.reserve(clusters_[0].size());
        for (tableint i = 0; i < cur_element_count_; ++i)
            if (element_levels_[i] >= 1) l1_elems.push_back(i);
        std::sort(l1_elems.begin(), l1_elems.end());

        if (l0_reps == l1_elems) {
            refreshRadiiAlongPath_(new_elem);
            return;
        }
    }

    // Demote everyone to level 0.
    for (tableint i = 0; i < cur_element_count_; ++i) element_levels_[i] = 0;
    // Clear per-level link storage. (vectors stay sized; entries reset.)
    for (size_t l = 0; l < outlinks_per_level_.size(); ++l) {
        std::fill(outlinks_per_level_[l].begin(), outlinks_per_level_[l].end(), INVALID_ID);
        std::fill(outlink_dist_per_level_[l].begin(), outlink_dist_per_level_[l].end(),
                  std::numeric_limits<dist_t>::max());
        for (auto& v : inlinks_per_level_[l]) v.clear();
    }
    // Clear cluster store at every level >= 1.
    for (size_t l = 1; l < clusters_.size(); ++l) {
        clusters_[l].clear();
        std::fill(elem_to_cluster_[l].begin(), elem_to_cluster_[l].end(), INVALID_CLUSTER);
    }

    // Set initial maxlevel from level-0 cluster count.
    if (clusters_[0].empty()) {
        maxlevel_ = -1; enterpoint_node_ = INVALID_ID;
        return;
    }
    if (clusters_[0].size() == 1) {
        maxlevel_ = 0;
        enterpoint_node_ = clusters_[0].front().rep_id;
        clusters_[0].front().parent_elem_id = INVALID_ID;
        return;
    }

    int level = 1;
    while (clusters_[level - 1].size() > 1) {
        ensureLevelCapacity(level);

        // Collect reps from the level below — these are the elements that exist at this level.
        std::vector<tableint> reps;
        reps.reserve(clusters_[level - 1].size());
        for (const auto& c : clusters_[level - 1]) reps.push_back(c.rep_id);

        for (tableint rep : reps) element_levels_[rep] = level;

        // Build NNG at this level by brute-force NN among the reps.
        const int lidx = level - 1;  // into per_level_ vectors
        for (size_t i = 0; i < reps.size(); ++i) {
            tableint a = reps[i];
            dist_t best_d = std::numeric_limits<dist_t>::max();
            tableint best_id = INVALID_ID;
            for (size_t j = 0; j < reps.size(); ++j) {
                if (i == j) continue;
                tableint b = reps[j];
                dist_t d = fstdistfunc_(getData(a), getData(b), dist_func_param_);
                if (d < best_d) { best_d = d; best_id = b; }
            }
            outlinks_per_level_[lidx][a]     = best_id;
            outlink_dist_per_level_[lidx][a] = best_d;
        }
        // Reverse-index inlinks at this level.
        for (tableint a : reps) {
            tableint o = outlinks_per_level_[lidx][a];
            if (o != INVALID_ID)
                inlinks_per_level_[lidx][o].push_back(a);
        }

        rebuildClustersAtLevel_(level);

        // Link each level-(l-1) cluster to its parent representative (same element id at level l).
        for (auto& c_below : clusters_[level - 1])
            c_below.parent_elem_id = c_below.rep_id;

        if (clusters_[level].size() == 1) {
            maxlevel_ = level;
            enterpoint_node_ = clusters_[level].front().rep_id;
            clusters_[level].front().parent_elem_id = INVALID_ID;
            return;
        }
        ++level;
    }
    // Reached a level with > 1 cluster but loop exited — shouldn't happen.
    maxlevel_ = level - 1;
    if (!clusters_[maxlevel_].empty())
        enterpoint_node_ = clusters_[maxlevel_].front().rep_id;
}

#ifdef HNNG_VERIFY
template<typename dist_t>
void HierarchicalNNG<dist_t>::verifyInvariantsAllLevels_() {
    verifyInvariantsLevel0_();
    // Per-level checks for levels >= 1.
    for (int l = 1; l <= maxlevel_; ++l) {
        const int lidx = l - 1;
        // Inlink/outlink reverse-index consistency
        for (tableint i = 0; i < cur_element_count_; ++i) {
            if (element_levels_[i] < l) continue;
            tableint o = outlinks_per_level_[lidx][i];
            if (o == INVALID_ID) continue;
            HNNG_VERIFY_CHECK(element_levels_[o] >= l, "level-l outlink target not at level l");
            const auto& il = inlinks_per_level_[lidx][o];
            HNNG_VERIFY_CHECK(std::find(il.begin(), il.end(), i) != il.end(),
                              "level-l outlink/inlink inconsistency");
        }
        for (tableint i = 0; i < cur_element_count_; ++i) {
            if (element_levels_[i] < l) continue;
            for (tableint src : inlinks_per_level_[lidx][i]) {
                HNNG_VERIFY_CHECK(element_levels_[src] >= l, "level-l inlink source not at level l");
                HNNG_VERIFY_CHECK(outlinks_per_level_[lidx][src] == i,
                                  "level-l inlink/outlink inconsistency");
            }
        }
        // Cluster sanity at level l
        for (int32_t ci = 0; ci < (int32_t)clusters_[l].size(); ++ci) {
            const auto& c = clusters_[l][ci];
            HNNG_VERIFY_CHECK(c.rep_id != INVALID_ID, "level-l cluster missing rep");
            HNNG_VERIFY_CHECK(element_levels_[c.rep_id] >= l, "rep not promoted to its level");
            for (tableint m : c.member_ids) {
                HNNG_VERIFY_CHECK(elem_to_cluster_[l][m] == ci, "level-l elem_to_cluster mismatch");
                HNNG_VERIFY_CHECK(element_levels_[m] >= l, "cluster member not at level l");
            }
        }
        // Top level should have exactly one cluster
        if (l == maxlevel_)
            HNNG_VERIFY_CHECK(clusters_[l].size() == 1, "top level should have one cluster");
    }
}
#endif

// ---- helpers used above ----

template<typename dist_t>
std::vector<tableint>
HierarchicalNNG<dist_t>::componentOf_(tableint start, int level) {
    std::vector<tableint> result;
    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    auto* visited = vl->mass;
    auto  marker  = vl->curV;
    std::vector<tableint> stack = {start};
    visited[start] = marker;
    while (!stack.empty()) {
        tableint x = stack.back(); stack.pop_back();
        result.push_back(x);
        tableint o = getOutlink(x, level);
        if (o != INVALID_ID && visited[o] != marker) {
            visited[o] = marker;
            stack.push_back(o);
        }
        for (tableint in : getInlinks(x, level)) {
            if (visited[in] != marker) {
                visited[in] = marker;
                stack.push_back(in);
            }
        }
    }
    visited_list_pool_->releaseVisitedList(vl);
    return result;
}

// selectRepresentative: max |inlinks| then min sum of inlink elements' outlink_distance.
template<typename dist_t>
tableint
HierarchicalNNG<dist_t>::selectRepresentative_(const std::vector<tableint>& members, int level) {
    if (members.empty()) return INVALID_ID;
    size_t best_in = 0;
    std::vector<tableint> cand;
    for (tableint m : members) {
        size_t n = getInlinks(m, level).size();
        if (n > best_in) { best_in = n; cand.clear(); cand.push_back(m); }
        else if (n == best_in) cand.push_back(m);
    }
    if (cand.size() == 1) return cand[0];
    tableint best = cand[0];
    dist_t best_sum = std::numeric_limits<dist_t>::max();
    for (tableint c : cand) {
        dist_t sum = 0;
        for (tableint in : getInlinks(c, level)) sum += getOutlinkDist(in, level);
        if (sum < best_sum) { best_sum = sum; best = c; }
    }
    return best;
}

template<typename dist_t>
dist_t
HierarchicalNNG<dist_t>::computeRadiusMetric_(const Cluster& c, int level) {
    dist_t r = 0;
    for (tableint m : c.member_ids) {
        dist_t d_sq = fstdistfunc_(getData(c.rep_id), getData(m), dist_func_param_);
        dist_t mtr  = space_->to_metric(d_sq);
        dist_t add  = 0;
        if (level > 0) {
            int32_t ci = elem_to_cluster_[level - 1][m];
            if (ci != INVALID_CLUSTER)
                add = clusters_[level - 1][ci].radius;
        }
        r = std::max(r, mtr + add);
    }
    return r;
}

// Rebuild all clusters at `level` by component-walking.
template<typename dist_t>
void HierarchicalNNG<dist_t>::rebuildClustersAtLevel_(int level, bool compute_radii) {
    clusters_[level].clear();
    std::fill(elem_to_cluster_[level].begin(), elem_to_cluster_[level].end(), INVALID_CLUSTER);

    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    auto* visited = vl->mass;
    auto  marker  = vl->curV;

    for (tableint i = 0; i < cur_element_count_; ++i) {
        if (level > 0 && element_levels_[i] < level) continue;
        if (visited[i] == marker) continue;

        std::vector<tableint> members;
        std::vector<tableint> stack = {i};
        visited[i] = marker;
        while (!stack.empty()) {
            tableint x = stack.back(); stack.pop_back();
            members.push_back(x);
            tableint o = getOutlink(x, level);
            if (o != INVALID_ID && visited[o] != marker) {
                visited[o] = marker;
                stack.push_back(o);
            }
            for (tableint in : getInlinks(x, level)) {
                if (visited[in] != marker) {
                    visited[in] = marker;
                    stack.push_back(in);
                }
            }
        }

        Cluster c;
        c.member_ids = std::move(members);
        c.rep_id = selectRepresentative_(c.member_ids, level);
        c.radius = 0;
        c.parent_elem_id = INVALID_ID;
        c.c_status = 0;
        int32_t idx = static_cast<int32_t>(clusters_[level].size());
        for (tableint m : c.member_ids) elem_to_cluster_[level][m] = idx;
        clusters_[level].push_back(std::move(c));
    }
    visited_list_pool_->releaseVisitedList(vl);

    if (compute_radii)
        for (auto& c : clusters_[level]) c.radius = computeRadiusMetric_(c, level);
}

#ifdef HNNG_VERIFY
template<typename dist_t>
void HierarchicalNNG<dist_t>::verifyInvariantsLevel0_() {
    // outlink/inlink consistency
    for (tableint i = 0; i < cur_element_count_; ++i) {
        tableint o = outlinkId0Ref(i);
        if (o == INVALID_ID) continue;
        HNNG_VERIFY_CHECK(o < cur_element_count_, "outlink out of range");
        const auto& il = getInlinks(o, 0);
        HNNG_VERIFY_CHECK(std::find(il.begin(), il.end(), i) != il.end(),
                          "outlink/inlink inconsistency");
    }
    for (tableint i = 0; i < cur_element_count_; ++i) {
        const auto& il = getInlinks(i, 0);
        for (tableint src : il) {
            HNNG_VERIFY_CHECK(src < cur_element_count_, "inlink out of range");
            HNNG_VERIFY_CHECK(outlinkId0Ref(src) == i, "inlink/outlink inconsistency");
        }
        HNNG_VERIFY_CHECK(static_cast<size_t>(inlinkCount0Ref(i)) == il.size(),
                          "inlink count cache wrong");
    }
    // every element belongs to exactly one cluster matching its component
    for (tableint i = 0; i < cur_element_count_; ++i) {
        int32_t ci = elem_to_cluster_[0][i];
        HNNG_VERIFY_CHECK(ci != INVALID_CLUSTER, "element has no cluster");
        const auto& c = clusters_[0][ci];
        HNNG_VERIFY_CHECK(std::find(c.member_ids.begin(), c.member_ids.end(), i)
                          != c.member_ids.end(), "element not in its cluster's members");
    }
    // cluster reps + member↔component consistency (skip empty clusters,
    // which can exist as tombstones after rewires moved all members away).
    for (int32_t ci = 0; ci < (int32_t)clusters_[0].size(); ++ci) {
        const auto& c = clusters_[0][ci];
        if (c.member_ids.empty()) continue;
        HNNG_VERIFY_CHECK(c.rep_id != INVALID_ID, "cluster has no rep");
        HNNG_VERIFY_CHECK(std::find(c.member_ids.begin(), c.member_ids.end(), c.rep_id)
                          != c.member_ids.end(), "rep not in members");
        auto comp = const_cast<HierarchicalNNG<dist_t>*>(this)->componentOf_(c.rep_id, 0);
        std::set<tableint> a(c.member_ids.begin(), c.member_ids.end());
        std::set<tableint> b(comp.begin(), comp.end());
        HNNG_VERIFY_CHECK(a == b, "cluster members != connected component");
    }
}
#endif

// Exact k-NN by hierarchical descent with cluster-radius pruning.
// At each level: compute δ(Q, C.Rep) for each candidate cluster C; consider C.Rep
// as a top-k candidate (with dedup); discard any C whose lower-bound
// δ(Q, C.Rep) − C.R is no smaller than the current k-th best distance. Descend
// into the surviving clusters' children. At level 0, evaluate each surviving
// cluster's non-rep members directly.
template<typename dist_t>
std::priority_queue<std::pair<dist_t, labeltype>>
HierarchicalNNG<dist_t>::searchKnn(void* query_data, size_t k) {
    std::priority_queue<std::pair<dist_t, labeltype>> result;
    if (cur_element_count_ == 0 || k == 0 || maxlevel_ < 0) return result;

    VisitedList* vl = visited_list_pool_->getFreeVisitedList();
    auto* visited_array  = vl->mass;
    auto  visited_marker = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>> heap;  // max-heap of size ≤ k
    auto consider = [&](tableint id, dist_t d) {
        if (visited_array[id] == visited_marker) return;
        visited_array[id] = visited_marker;
        if (heap.size() < k) {
            heap.push({d, id});
        } else if (d < heap.top().first) {
            heap.push({d, id});
            heap.pop();
        }
    };

    // Pruning works in metric space (triangle inequality required). Storage and
    // heap stay in raw-distance space — only the lower-bound test takes a sqrt.
    // Each cluster carries radius in metric units already.
    auto threshold_metric = [&]() {
        return heap.size() < k ? std::numeric_limits<dist_t>::max()
                               : space_->to_metric(heap.top().first);
    };

    std::vector<const Cluster*> cluster_list;
    cluster_list.reserve(clusters_[maxlevel_].size());
    for (const Cluster& c : clusters_[maxlevel_]) cluster_list.push_back(&c);

    for (int level = maxlevel_; level >= 0; --level) {
        std::vector<dist_t> reps_dist(cluster_list.size());
        std::vector<dist_t> reps_dist_m(cluster_list.size());
        for (size_t i = 0; i < cluster_list.size(); ++i) {
            tableint rep = cluster_list[i]->rep_id;
            reps_dist[i]   = fstdistfunc_(query_data, getData(rep), dist_func_param_);
            reps_dist_m[i] = space_->to_metric(reps_dist[i]);
            consider(rep, reps_dist[i]);
        }

        if (level == 0) {
            for (size_t i = 0; i < cluster_list.size(); ++i) {
                const Cluster& c = *cluster_list[i];
                if (reps_dist_m[i] >= c.radius + threshold_metric()) continue;
                for (tableint m : c.member_ids) {
                    if (visited_array[m] == visited_marker) continue;
                    dist_t d = fstdistfunc_(query_data, getData(m), dist_func_param_);
                    consider(m, d);
                }
            }
            break;
        }

        std::vector<const Cluster*> next_list;
        for (size_t i = 0; i < cluster_list.size(); ++i) {
            const Cluster& c = *cluster_list[i];
            if (reps_dist_m[i] >= c.radius + threshold_metric()) continue;
            for (tableint m : c.member_ids) {
                int32_t ci = elem_to_cluster_[level - 1][m];
                if (ci != INVALID_CLUSTER)
                    next_list.push_back(&clusters_[level - 1][ci]);
            }
        }
        cluster_list = std::move(next_list);
    }

    visited_list_pool_->releaseVisitedList(vl);

    while (!heap.empty()) {
        result.push({heap.top().first, getLabel(heap.top().second)});
        heap.pop();
    }
    return result;
}

// ---- Phase-1 raw accessors ----
template<typename dist_t>
tableint HierarchicalNNG<dist_t>::_test_addRawElement(const void* data, labeltype label, int top_level) {
    if (cur_element_count_ >= max_elements_)
        throw std::runtime_error("hnng: capacity exceeded");
    if (label_lookup_.count(label))
        throw std::runtime_error("hnng: duplicate label");

    tableint id = static_cast<tableint>(cur_element_count_++);
    label_lookup_[label] = id;

    outlinkId0Ref(id)    = INVALID_ID;
    outlinkDist0Ref(id)  = std::numeric_limits<dist_t>::max();
    inlinkCount0Ref(id)  = 0;
    std::memcpy(data0Ptr(id), data, data_size_);
    label0Ref(id)        = label;

    element_levels_[id]  = top_level;
    if (top_level > 0) {
        ensureLevelCapacity(top_level);
        for (int l = 1; l <= top_level; ++l) {
            outlinks_per_level_[l - 1][id]     = INVALID_ID;
            outlink_dist_per_level_[l - 1][id] = std::numeric_limits<dist_t>::max();
            inlinks_per_level_[l - 1][id].clear();
        }
    }
    return id;
}

template<typename dist_t>
void HierarchicalNNG<dist_t>::_test_setOutlink(tableint id, int level, tableint target, dist_t dist) {
    if (level == 0) {
        outlinkId0Ref(id)   = target;
        outlinkDist0Ref(id) = dist;
    } else {
        ensureLevelCapacity(level);
        outlinks_per_level_[level - 1][id]     = target;
        outlink_dist_per_level_[level - 1][id] = dist;
    }
}

template<typename dist_t>
void HierarchicalNNG<dist_t>::_test_addInlink(tableint id, int level, tableint source) {
    if (level == 0) {
        if (!inlink_lists_l0_[id]) inlink_lists_l0_[id] = new std::vector<tableint>();
        inlink_lists_l0_[id]->push_back(source);
        inlinkCount0Ref(id) = static_cast<linklistsizeint>(inlink_lists_l0_[id]->size());
    } else {
        ensureLevelCapacity(level);
        inlinks_per_level_[level - 1][id].push_back(source);
    }
}

template<typename dist_t>
int32_t HierarchicalNNG<dist_t>::_test_addCluster(int level, tableint rep_id,
                                                  const std::vector<tableint>& member_ids,
                                                  dist_t radius,
                                                  tableint parent_elem_id) {
    ensureLevelCapacity(level);
    Cluster c;
    c.rep_id         = rep_id;
    c.member_ids     = member_ids;
    c.radius         = radius;
    c.parent_elem_id = parent_elem_id;
    c.c_status       = 0;
    int32_t idx = static_cast<int32_t>(clusters_[level].size());
    clusters_[level].push_back(std::move(c));
    for (tableint m : member_ids) elem_to_cluster_[level][m] = idx;
    return idx;
}

// ---- Read accessors ----
template<typename dist_t>
tableint HierarchicalNNG<dist_t>::getOutlink(tableint id, int level) const {
    if (level == 0)
        return *reinterpret_cast<const tableint*>(level0BlockPtr(id) + offset_outlink_id_);
    return outlinks_per_level_[level - 1][id];
}

template<typename dist_t>
dist_t HierarchicalNNG<dist_t>::getOutlinkDist(tableint id, int level) const {
    if (level == 0)
        return *reinterpret_cast<const dist_t*>(level0BlockPtr(id) + offset_outlink_dist_);
    return outlink_dist_per_level_[level - 1][id];
}

template<typename dist_t>
const std::vector<tableint>& HierarchicalNNG<dist_t>::getInlinks(tableint id, int level) const {
    static const std::vector<tableint> empty;
    if (level == 0)
        return inlink_lists_l0_[id] ? *inlink_lists_l0_[id] : empty;
    return inlinks_per_level_[level - 1][id];
}

template<typename dist_t>
labeltype HierarchicalNNG<dist_t>::getLabel(tableint id) const {
    return *reinterpret_cast<const labeltype*>(level0BlockPtr(id) + offset_label_);
}

template<typename dist_t>
const char* HierarchicalNNG<dist_t>::getData(tableint id) const {
    return level0BlockPtr(id) + offset_data_;
}

template<typename dist_t>
int32_t HierarchicalNNG<dist_t>::getClusterIndex(tableint id, int level) const {
    return elem_to_cluster_[level][id];
}

// ---- Save / Load ----
template<typename dist_t>
void HierarchicalNNG<dist_t>::saveIndex(const std::string& location) {
    std::ofstream out(location, std::ios::binary);
    if (!out) throw std::runtime_error("hnng: cannot open save file");

    writeBinaryPOD(out, max_elements_);
    writeBinaryPOD(out, cur_element_count_);
    writeBinaryPOD(out, data_size_);
    writeBinaryPOD(out, size_per_element_);
    writeBinaryPOD(out, maxlevel_);
    writeBinaryPOD(out, enterpoint_node_);

    out.write(data_level0_memory_, cur_element_count_ * size_per_element_);

    // Element levels
    for (size_t i = 0; i < cur_element_count_; ++i)
        writeBinaryPOD(out, element_levels_[i]);

    // Level-0 inlinks
    for (size_t i = 0; i < cur_element_count_; ++i) {
        const auto* v = inlink_lists_l0_[i];
        uint32_t n = v ? static_cast<uint32_t>(v->size()) : 0;
        writeBinaryPOD(out, n);
        if (n) out.write(reinterpret_cast<const char*>(v->data()), n * sizeof(tableint));
    }

    // Higher-level outlinks and inlinks
    int nlevels = static_cast<int>(outlinks_per_level_.size());
    writeBinaryPOD(out, nlevels);
    for (int l = 0; l < nlevels; ++l) {
        for (size_t i = 0; i < cur_element_count_; ++i) {
            if (element_levels_[i] >= l + 1) {
                writeBinaryPOD(out, outlinks_per_level_[l][i]);
                writeBinaryPOD(out, outlink_dist_per_level_[l][i]);
                uint32_t n = static_cast<uint32_t>(inlinks_per_level_[l][i].size());
                writeBinaryPOD(out, n);
                if (n) out.write(reinterpret_cast<const char*>(inlinks_per_level_[l][i].data()),
                                 n * sizeof(tableint));
            }
        }
    }

    // Clusters per level
    int nclevels = static_cast<int>(clusters_.size());
    writeBinaryPOD(out, nclevels);
    for (int l = 0; l < nclevels; ++l) {
        uint32_t ncls = static_cast<uint32_t>(clusters_[l].size());
        writeBinaryPOD(out, ncls);
        for (const auto& c : clusters_[l]) {
            writeBinaryPOD(out, c.rep_id);
            writeBinaryPOD(out, c.radius);
            writeBinaryPOD(out, c.parent_elem_id);
            writeBinaryPOD(out, c.c_status);
            uint32_t nm = static_cast<uint32_t>(c.member_ids.size());
            writeBinaryPOD(out, nm);
            if (nm) out.write(reinterpret_cast<const char*>(c.member_ids.data()),
                              nm * sizeof(tableint));
        }
    }
}

template<typename dist_t>
void HierarchicalNNG<dist_t>::loadIndex(const std::string& location, SpaceInterface<dist_t>* s) {
    std::ifstream in(location, std::ios::binary);
    if (!in) throw std::runtime_error("hnng: cannot open load file");

    size_t saved_max, saved_count, saved_data_size, saved_size_per;
    int    saved_maxlevel;
    tableint saved_entry;
    readBinaryPOD(in, saved_max);
    readBinaryPOD(in, saved_count);
    readBinaryPOD(in, saved_data_size);
    readBinaryPOD(in, saved_size_per);
    readBinaryPOD(in, saved_maxlevel);
    readBinaryPOD(in, saved_entry);

    freeAll();
    space_ = s;
    fstdistfunc_     = s->get_dist_func();
    dist_func_param_ = s->get_dist_func_param();
    data_size_       = s->get_data_size();
    if (data_size_ != saved_data_size)
        throw std::runtime_error("hnng: data_size mismatch on load");
    init(saved_max);
    cur_element_count_ = saved_count;
    maxlevel_          = saved_maxlevel;
    enterpoint_node_   = saved_entry;
    if (saved_size_per != size_per_element_)
        throw std::runtime_error("hnng: size_per_element mismatch on load");

    in.read(data_level0_memory_, cur_element_count_ * size_per_element_);

    for (size_t i = 0; i < cur_element_count_; ++i)
        readBinaryPOD(in, element_levels_[i]);

    // Rebuild label lookup from the loaded flat block.
    for (size_t i = 0; i < cur_element_count_; ++i)
        label_lookup_[getLabel(static_cast<tableint>(i))] = static_cast<tableint>(i);

    // Level-0 inlinks
    for (size_t i = 0; i < cur_element_count_; ++i) {
        uint32_t n; readBinaryPOD(in, n);
        if (n) {
            auto* v = new std::vector<tableint>(n);
            in.read(reinterpret_cast<char*>(v->data()), n * sizeof(tableint));
            inlink_lists_l0_[i] = v;
        }
    }

    // Higher-level outlinks/inlinks
    int nlevels; readBinaryPOD(in, nlevels);
    if (nlevels > 0) ensureLevelCapacity(nlevels);
    for (int l = 0; l < nlevels; ++l) {
        for (size_t i = 0; i < cur_element_count_; ++i) {
            if (element_levels_[i] >= l + 1) {
                tableint o;        readBinaryPOD(in, o);
                dist_t   d;        readBinaryPOD(in, d);
                uint32_t n;        readBinaryPOD(in, n);
                outlinks_per_level_[l][i]     = o;
                outlink_dist_per_level_[l][i] = d;
                inlinks_per_level_[l][i].resize(n);
                if (n) in.read(reinterpret_cast<char*>(inlinks_per_level_[l][i].data()),
                               n * sizeof(tableint));
            }
        }
    }

    // Clusters
    int nclevels; readBinaryPOD(in, nclevels);
    if (nclevels > static_cast<int>(clusters_.size())) {
        while (static_cast<int>(clusters_.size()) < nclevels) {
            clusters_.emplace_back();
            elem_to_cluster_.emplace_back(max_elements_, INVALID_CLUSTER);
        }
    }
    for (int l = 0; l < nclevels; ++l) {
        uint32_t ncls; readBinaryPOD(in, ncls);
        clusters_[l].resize(ncls);
        for (uint32_t ci = 0; ci < ncls; ++ci) {
            Cluster& c = clusters_[l][ci];
            readBinaryPOD(in, c.rep_id);
            readBinaryPOD(in, c.radius);
            readBinaryPOD(in, c.parent_elem_id);
            readBinaryPOD(in, c.c_status);
            uint32_t nm; readBinaryPOD(in, nm);
            c.member_ids.resize(nm);
            if (nm) in.read(reinterpret_cast<char*>(c.member_ids.data()),
                            nm * sizeof(tableint));
            for (tableint m : c.member_ids)
                elem_to_cluster_[l][m] = static_cast<int32_t>(ci);
        }
    }
}

}  // namespace hnnglib
