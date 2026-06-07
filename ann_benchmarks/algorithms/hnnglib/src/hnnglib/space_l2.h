#pragma once

#include <cstddef>

namespace hnnglib {

// L2 (Euclidean) distance - squared
static float L2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    return res;
}

// L2 Space class
class L2Space : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    
 public:
    L2Space(size_t dim) {
        fstdistfunc_ = L2Sqr;
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }
    
    size_t get_data_size() {
        return data_size_;
    }
    
    DISTFUNC<float> get_dist_func() {
        return fstdistfunc_;
    }
    
    void *get_dist_func_param() {
        return &dim_;
    }
    
    ~L2Space() {}
};

}  // namespace hnnglib
