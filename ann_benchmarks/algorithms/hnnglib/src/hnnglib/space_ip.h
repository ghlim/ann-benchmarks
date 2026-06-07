#pragma once

#include <cstddef>

namespace hnnglib {

// Inner Product distance
static float InnerProductDistance(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);
    
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += pVect1[i] * pVect2[i];
    }
    return 1.0f - res;  // Convert to distance
}

// Inner Product Space class
class InnerProductSpace : public SpaceInterface<float> {
    DISTFUNC<float> fstdistfunc_;
    size_t data_size_;
    size_t dim_;
    
 public:
    InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
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
    
    ~InnerProductSpace() {}
};

}  // namespace hnnglib
