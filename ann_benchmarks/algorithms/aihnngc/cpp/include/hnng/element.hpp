// element.hpp -- Element and Cluster structs, index-based uint32 storage.
//
// Compiles and is tested (g++ 13.3 / pybind11 3.0.4; CI runs the suite).
//
// Mirrors hnng_ref/core.py Element and Cluster. Where the reference uses
// Python `Optional[int]` we use uint32_t with a sentinel `kNone`. Where it uses
// a Python `set` for inlinks we use a sorted-on-demand std::vector<uint32_t>;
// the reference only ever iterates inlinks and takes their size/average, so a
// vector is behaviorally equivalent (we keep entries unique on insertion).

#ifndef HNNG_ELEMENT_HPP
#define HNNG_ELEMENT_HPP

#include <cstdint>
#include <limits>
#include <vector>

namespace hnng {

// Sentinel for "no index" (Python None). Using max uint32.
constexpr std::uint32_t kNone = std::numeric_limits<std::uint32_t>::max();

// A node at one level of the hierarchy. Field-for-field mirror of the
// reference Element (see hnng_ref/core.py).
struct Element {
    std::uint32_t object_index = 0;   // index into the prepared object store
    std::uint32_t outlink = kNone;    // position (this level) of nearest neighbour
    double dist = 0.0;                // distance to outlink (NN distance)
    std::vector<std::uint32_t> inlinks;  // positions whose outlink is this element
    std::uint32_t num_child = 1;      // # leaf objects in subtree (1 at level 0)
    double subtree_radius = 0.0;      // radius of the cluster this element heads below
    std::uint32_t heads_cluster = kNone;  // index into level-below cluster list

    Element() = default;
    explicit Element(std::uint32_t obj) : object_index(obj) {}
};

// A connected component of an NNG level. Mirror of the reference Cluster.
struct Cluster {
    std::uint32_t level = 0;
    std::uint32_t rep = 0;                 // position (this level) of representative
    std::vector<std::uint32_t> members;    // positions (this level) of members
    double radius = 0.0;                   // bounding-sphere radius (see core.py)
    std::uint32_t parent = kNone;          // position of promoted element at level+1

    Cluster() = default;
    Cluster(std::uint32_t lvl, std::uint32_t rep_, std::vector<std::uint32_t> members_)
        : level(lvl), rep(rep_), members(std::move(members_)) {}
};

}  // namespace hnng

#endif  // HNNG_ELEMENT_HPP
