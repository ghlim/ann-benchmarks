#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../hnnglib/hnnglib.h"

namespace py = pybind11;
using namespace hnnglib;

// Python wrapper for HierarchicalNNG
template<typename dist_t, typename data_t = float>
class Index {
 public:
    Index(const std::string &space_name, int dim) : dim_(dim), index_inited_(false), ep_added_(false) {
        normalize_ = false;
        
        if (space_name == "l2") {
            space_ = new L2Space(dim);
        } else if (space_name == "ip") {
            space_ = new InnerProductSpace(dim);
        } else if (space_name == "cosine") {
            space_ = new InnerProductSpace(dim);
            normalize_ = true;
        } else {
            throw std::runtime_error("Space name must be one of: l2, ip, cosine");
        }
        
        alg_hnng_ = NULL;
    }
    
    ~Index() {
        delete space_;
        if (alg_hnng_)
            delete alg_hnng_;
    }
    
    void init_index(size_t max_elements) {
        if (index_inited_) {
            throw std::runtime_error("Index already initialized");
        }
        
        alg_hnng_ = new HierarchicalNNG<dist_t>(space_, max_elements);
        index_inited_ = true;
        ep_added_ = false;
    }
    
    void add_items(py::array_t<data_t, py::array::c_style | py::array::forcecast> data,
                  py::array_t<labeltype> ids) {
        
        if (!index_inited_) {
            throw std::runtime_error("Index not initialized. Call init_index() first.");
        }
        
        py::buffer_info data_buf = data.request();
        py::buffer_info ids_buf = ids.request();
        
        if (data_buf.ndim != 2) {
            throw std::runtime_error("Data must be 2D array");
        }
        
        size_t num_elements = data_buf.shape[0];
        size_t data_dim = data_buf.shape[1];
        
        if (data_dim != dim_) {
            throw std::runtime_error("Data dimension doesn't match index dimension");
        }
        
        if (ids_buf.shape[0] != num_elements) {
            throw std::runtime_error("Number of labels doesn't match number of data points");
        }
        
        data_t *data_ptr = (data_t *) data_buf.ptr;
        labeltype *ids_ptr = (labeltype *) ids_buf.ptr;
        
        // Add items one by one
        for (size_t i = 0; i < num_elements; i++) {
            data_t *point_data = data_ptr + i * dim_;
            
            // Normalize if cosine distance
            if (normalize_) {
                std::vector<data_t> normalized(dim_);
                data_t norm = 0;
                for (size_t j = 0; j < dim_; j++) {
                    norm += point_data[j] * point_data[j];
                }
                norm = std::sqrt(norm);
                for (size_t j = 0; j < dim_; j++) {
                    normalized[j] = point_data[j] / norm;
                }
                alg_hnng_->addPoint(normalized.data(), ids_ptr[i]);
            } else {
                alg_hnng_->addPoint(point_data, ids_ptr[i]);
            }
            
            ep_added_ = true;
        }
    }
    
    py::object knn_query(py::array_t<data_t, py::array::c_style | py::array::forcecast> query,
                        size_t k) {
        
        if (!index_inited_) {
            throw std::runtime_error("Index not initialized");
        }
        
        if (!ep_added_) {
            throw std::runtime_error("No elements added to index");
        }
        
        py::buffer_info query_buf = query.request();
        
        size_t num_queries = 1;
        size_t query_dim = dim_;
        
        if (query_buf.ndim == 1) {
            query_dim = query_buf.shape[0];
        } else if (query_buf.ndim == 2) {
            num_queries = query_buf.shape[0];
            query_dim = query_buf.shape[1];
        } else {
            throw std::runtime_error("Query must be 1D or 2D array");
        }
        
        if (query_dim != dim_) {
            throw std::runtime_error("Query dimension doesn't match index dimension");
        }
        
        data_t *query_ptr = (data_t *) query_buf.ptr;
        
        // Prepare output arrays
        py::array_t<labeltype> labels({num_queries, k});
        py::array_t<dist_t> distances({num_queries, k});
        
        labeltype *labels_ptr = (labeltype *) labels.request().ptr;
        dist_t *distances_ptr = (dist_t *) distances.request().ptr;
        
        // Process each query
        for (size_t i = 0; i < num_queries; i++) {
            data_t *query_data = query_ptr + i * dim_;
            
            // Normalize if cosine
            std::vector<data_t> normalized;
            if (normalize_) {
                normalized.resize(dim_);
                data_t norm = 0;
                for (size_t j = 0; j < dim_; j++) {
                    norm += query_data[j] * query_data[j];
                }
                norm = std::sqrt(norm);
                for (size_t j = 0; j < dim_; j++) {
                    normalized[j] = query_data[j] / norm;
                }
                query_data = normalized.data();
            }
            
            // Search
            auto result = alg_hnng_->searchKnnCloserFirst(query_data, k);
            
            // Copy results
            for (size_t j = 0; j < k && j < result.size(); j++) {
                labels_ptr[i * k + j] = result[j].second;
                distances_ptr[i * k + j] = result[j].first;
            }
            
            // Fill remaining with -1
            for (size_t j = result.size(); j < k; j++) {
                labels_ptr[i * k + j] = -1;
                distances_ptr[i * k + j] = -1;
            }
        }
        
        if (num_queries == 1) {
            // Return 1D arrays for single query
            labels.resize({k});
            distances.resize({k});
        }
        
        return py::make_tuple(labels, distances);
    }
    
    void save_index(const std::string &path) {
        if (!index_inited_) {
            throw std::runtime_error("Index not initialized");
        }
        alg_hnng_->saveIndex(path);
    }
    
    void load_index(const std::string &path) {
        if (!index_inited_) {
            throw std::runtime_error("Index not initialized. Call init_index() first.");
        }
        alg_hnng_->loadIndex(path, space_);
        ep_added_ = true;
    }
    
    size_t get_num_levels() const {
        if (!index_inited_) {
            throw std::runtime_error("Index not initialized");
        }
        return alg_hnng_->getNumLevels();
    }
    
    size_t get_num_clusters(size_t level) const {
        if (!index_inited_) {
            throw std::runtime_error("Index not initialized");
        }
        return alg_hnng_->getNumClusters(level);
    }
    
 private:
    int dim_;
    bool index_inited_;
    bool ep_added_;
    bool normalize_;
    HierarchicalNNG<dist_t> *alg_hnng_;
    SpaceInterface<dist_t> *space_;
};


PYBIND11_MODULE(hnnglib, m) {
    m.doc() = "hnnglib - Hierarchical Nearest Neighbor Graph library";
    
    py::class_<Index<float>>(m, "Index")
        .def(py::init<const std::string &, int>(),
             py::arg("space"), py::arg("dim"))
        .def("init_index", &Index<float>::init_index,
             py::arg("max_elements"),
             "Initialize the index with maximum capacity")
        .def("add_items", &Index<float>::add_items,
             py::arg("data"), py::arg("ids"),
             "Add items to the index")
        .def("knn_query", &Index<float>::knn_query,
             py::arg("data"), py::arg("k") = 1,
             "Query for k nearest neighbors")
        .def("save_index", &Index<float>::save_index,
             py::arg("path"),
             "Save index to file")
        .def("load_index", &Index<float>::load_index,
             py::arg("path"),
             "Load index from file")
        .def("get_num_levels", &Index<float>::get_num_levels,
             "Get number of levels in hierarchy")
        .def("get_num_clusters", &Index<float>::get_num_clusters,
             py::arg("level"),
             "Get number of clusters at a specific level")
        .def("__repr__", [](const Index<float> &idx) {
            return "<hnnglib.Index>";
        });
    
    m.attr("__version__") = "0.1.0";
}
