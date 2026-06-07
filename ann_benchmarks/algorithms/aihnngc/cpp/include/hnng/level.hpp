// level.hpp -- one NNG level: element & cluster arrays.
//
// Compiles and is tested (g++ 13.3 / pybind11 3.0.4; CI runs the suite).
//
// The reference keeps two parallel Python lists per level:
//   * index.levels[L]   : list[Element]
//   * index.clusters[L] : list[Cluster]  (clusters of level L, empty for root)
// We keep the same shape: `Level` holds the element array for a level, and the
// HNNG owns a parallel vector of cluster arrays. Splitting them keeps the data
// layout identical to the reference and makes the 1:1 mapping easy to audit.

#ifndef HNNG_LEVEL_HPP
#define HNNG_LEVEL_HPP

#include <vector>

#include "hnng/element.hpp"

namespace hnng {

// One level's element array. A thin wrapper kept for clarity / future SoA
// migration; for now it is just a vector<Element>.
struct Level {
    std::vector<Element> elements;

    std::size_t size() const { return elements.size(); }
    bool empty() const { return elements.empty(); }
    Element& operator[](std::size_t i) { return elements[i]; }
    const Element& operator[](std::size_t i) const { return elements[i]; }
};

}  // namespace hnng

#endif  // HNNG_LEVEL_HPP
