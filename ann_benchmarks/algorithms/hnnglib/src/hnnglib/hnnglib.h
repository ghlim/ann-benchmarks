#pragma once

#ifndef HNNGLIB_H
#define HNNGLIB_H

#include <queue>
#include <vector>
#include <iostream>
#include <string.h>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <limits>

namespace hnnglib {

typedef size_t labeltype;

// Distance function type
template<typename MTYPE>
using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);

// Base filter functor for filtering during search
class BaseFilterFunctor {
 public:
    virtual bool operator()(hnnglib::labeltype id) { return true; }
    virtual ~BaseFilterFunctor() {};
};

// Pair comparison for priority queue (smaller distance = higher priority)
template <typename T>
class pairGreater {
 public:
    bool operator()(const T& p1, const T& p2) {
        return p1.first > p2.first;
    }
};

// Binary I/O helpers
template<typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
    out.write((char *) &podRef, sizeof(T));
}

template<typename T>
static void readBinaryPOD(std::istream &in, T &podRef) {
    in.read((char *) &podRef, sizeof(T));
}

// Space interface for distance functions
template<typename MTYPE>
class SpaceInterface {
 public:
    virtual size_t get_data_size() = 0;
    virtual DISTFUNC<MTYPE> get_dist_func() = 0;
    virtual void *get_dist_func_param() = 0;
    virtual ~SpaceInterface() {}
};

// Algorithm interface
template<typename dist_t>
class AlgorithmInterface {
 public:
    virtual void addPoint(const void *datapoint, labeltype label, bool replace_deleted = false) = 0;
    
    virtual std::priority_queue<std::pair<dist_t, labeltype>>
        searchKnn(const void* query, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const = 0;
    
    virtual std::vector<std::pair<dist_t, labeltype>>
        searchKnnCloserFirst(const void* query_data, size_t k, 
                           BaseFilterFunctor* isIdAllowed = nullptr) const;
    
    virtual void saveIndex(const std::string &location) = 0;
    
    virtual ~AlgorithmInterface() {}
};

template<typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void* query_data, size_t k,
                                                 BaseFilterFunctor* isIdAllowed) const {
    std::vector<std::pair<dist_t, labeltype>> result;
    auto ret = searchKnn(query_data, k, isIdAllowed);
    
    size_t sz = ret.size();
    result.resize(sz);
    while (!ret.empty()) {
        result[--sz] = ret.top();
        ret.pop();
    }
    
    return result;
}

}  // namespace hnnglib

// Include space implementations (reuse from hnswlib pattern)
#include "space_l2.h"
#include "space_ip.h"

// Include core algorithm
#include "hnnalg.h"

#endif  // HNNGLIB_H
