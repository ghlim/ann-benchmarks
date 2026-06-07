#pragma once

#include <fstream>
#include <mutex>
#include <cstring>

namespace hnnglib {

template<typename dist_t>
class HierarchicalNNG : public AlgorithmInterface<dist_t> {
 public:
    static const labeltype INVALID_LABEL = static_cast<labeltype>(-1);
    static const size_t MAX_LEVELS = 16;  // Maximum hierarchy depth
    
    // Element in NNG - stores links and metadata
    struct Element {
        labeltype label;
        labeltype outlink;  // Nearest neighbor
        std::vector<labeltype> inlinks;  // Elements pointing to this one
        dist_t outlink_distance;
        size_t level;
        
        Element() : label(INVALID_LABEL), outlink(INVALID_LABEL), 
                   outlink_distance(std::numeric_limits<dist_t>::max()), level(0) {}
    };
    
    // Cluster in hierarchy
    struct Cluster {
        size_t id;
        size_t level;
        labeltype representative;  // Representative element
        labeltype parent;  // Parent cluster (in upper level)
        std::vector<labeltype> members;  // All member elements
        dist_t radius;  // Max distance from rep to members
        
        Cluster() : id(0), level(0), representative(INVALID_LABEL), 
                   parent(INVALID_LABEL), radius(0.0) {}
    };
    
 private:
    // Core data structures
    SpaceInterface<dist_t> *space_;
    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_;
    size_t dim_;
    size_t data_size_;
    
    // Storage
    std::vector<float> data_;  // Raw data vectors
    std::unordered_map<labeltype, size_t> label_lookup_;  // label -> internal index
    std::vector<labeltype> index_to_label_;  // internal index -> label
    
    // Hierarchy storage (one vector per level)
    std::vector<Element> elements_[MAX_LEVELS];
    std::vector<Cluster> clusters_[MAX_LEVELS];
    std::unordered_map<labeltype, size_t> element_lookup_[MAX_LEVELS];  // label -> element index at level
    
    size_t max_elements_;
    size_t cur_element_count_;
    size_t num_levels_;
    
    std::mutex global_lock_;
    
 public:
    HierarchicalNNG(SpaceInterface<dist_t> *space, size_t max_elements)
        : space_(space), max_elements_(max_elements), cur_element_count_(0), num_levels_(1) {
        
        fstdistfunc_ = space_->get_dist_func();
        dist_func_param_ = space_->get_dist_func_param();
        dim_ = *((size_t *) dist_func_param_);
        data_size_ = space_->get_data_size();
        
        // Pre-allocate data storage
        data_.resize(max_elements_ * dim_);
    }
    
    ~HierarchicalNNG() {}
    
    // Add a single point to the hierarchy
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) override {
        std::unique_lock<std::mutex> lock(global_lock_);
        
        // Check if label already exists
        if (label_lookup_.find(label) != label_lookup_.end()) {
            throw std::runtime_error("Label already exists");
        }
        
        if (cur_element_count_ >= max_elements_) {
            throw std::runtime_error("Cannot add more elements - max capacity reached");
        }
        
        // Store data
        size_t internal_id = cur_element_count_;
        memcpy(data_.data() + internal_id * dim_, data_point, data_size_);
        label_lookup_[label] = internal_id;
        index_to_label_.push_back(label);
        cur_element_count_++;
        
        // Add to hierarchy starting at level 0
        addToHierarchy(label, 0);
    }
    
    // Get number of hierarchy levels
    size_t getNumLevels() const {
        return num_levels_;
    }
    
    // Get number of clusters at a level
    size_t getNumClusters(size_t level) const {
        if (level >= num_levels_) return 0;
        return clusters_[level].size();
    }
    
    // Search for k nearest neighbors
    std::priority_queue<std::pair<dist_t, labeltype>>
    searchKnn(const void* query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const override {
        std::priority_queue<std::pair<dist_t, labeltype>> result;
        
        if (cur_element_count_ == 0) {
            return result;
        }
        
        // Start from top level, descend hierarchically
        labeltype entry_point = getEntryPoint();
        
        // Perform greedy search at bottom level
        std::priority_queue<std::pair<dist_t, labeltype>, 
                          std::vector<std::pair<dist_t, labeltype>>,
                          pairGreater<std::pair<dist_t, labeltype>>> candidates;
        
        dist_t dist = computeDistance(query_data, entry_point);
        candidates.push({dist, entry_point});
        
        std::unordered_set<labeltype> visited;
        visited.insert(entry_point);
        
        while (!candidates.empty() && result.size() < k) {
            auto current = candidates.top();
            candidates.pop();
            
            labeltype current_label = current.second;
            
            // Add to result if passes filter
            if (isIdAllowed == nullptr || (*isIdAllowed)(current_label)) {
                result.push(current);
            }
            
            // Explore neighbors via outlink and inlinks at level 0
            if (element_lookup_[0].find(current_label) != element_lookup_[0].end()) {
                size_t elem_idx = element_lookup_[0].at(current_label);
                const Element& elem = elements_[0][elem_idx];
                
                // Check outlink
                if (elem.outlink != INVALID_LABEL && visited.find(elem.outlink) == visited.end()) {
                    dist_t d = computeDistance(query_data, elem.outlink);
                    candidates.push({d, elem.outlink});
                    visited.insert(elem.outlink);
                }
                
                // Check inlinks
                for (labeltype inlink : elem.inlinks) {
                    if (visited.find(inlink) == visited.end()) {
                        dist_t d = computeDistance(query_data, inlink);
                        candidates.push({d, inlink});
                        visited.insert(inlink);
                    }
                }
            }
        }
        
        return result;
    }
    
    // Save index to file
    void saveIndex(const std::string &location) override {
        std::ofstream out(location, std::ios::binary);
        if (!out.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + location);
        }
        
        // Write header
        out.write("HNNG", 4);
        writeBinaryPOD(out, static_cast<uint32_t>(1));  // version
        writeBinaryPOD(out, static_cast<uint32_t>(dim_));
        writeBinaryPOD(out, max_elements_);
        writeBinaryPOD(out, cur_element_count_);
        writeBinaryPOD(out, num_levels_);
        
        // Write data
        for (size_t i = 0; i < cur_element_count_; i++) {
            writeBinaryPOD(out, index_to_label_[i]);
            out.write((char*)(data_.data() + i * dim_), data_size_);
        }
        
        // Write hierarchy
        for (size_t level = 0; level < num_levels_; level++) {
            writeBinaryPOD(out, elements_[level].size());
            for (const auto& elem : elements_[level]) {
                writeBinaryPOD(out, elem.label);
                writeBinaryPOD(out, elem.outlink);
                writeBinaryPOD(out, elem.outlink_distance);
                writeBinaryPOD(out, elem.inlinks.size());
                for (labeltype inlink : elem.inlinks) {
                    writeBinaryPOD(out, inlink);
                }
            }
            
            writeBinaryPOD(out, clusters_[level].size());
            for (const auto& cluster : clusters_[level]) {
                writeBinaryPOD(out, cluster.id);
                writeBinaryPOD(out, cluster.representative);
                writeBinaryPOD(out, cluster.parent);
                writeBinaryPOD(out, cluster.radius);
                writeBinaryPOD(out, cluster.members.size());
                for (labeltype member : cluster.members) {
                    writeBinaryPOD(out, member);
                }
            }
        }
        
        out.close();
    }
    
    // Load index from file
    void loadIndex(const std::string &location, SpaceInterface<dist_t> *space) {
        std::ifstream in(location, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + location);
        }
        
        // Read and verify header
        char magic[4];
        in.read(magic, 4);
        if (strncmp(magic, "HNNG", 4) != 0) {
            throw std::runtime_error("Invalid file format");
        }
        
        uint32_t version, dim;
        readBinaryPOD(in, version);
        readBinaryPOD(in, dim);
        readBinaryPOD(in, max_elements_);
        readBinaryPOD(in, cur_element_count_);
        readBinaryPOD(in, num_levels_);
        
        // Initialize
        space_ = space;
        dim_ = dim;
        fstdistfunc_ = space_->get_dist_func();
        dist_func_param_ = space_->get_dist_func_param();
        data_size_ = space_->get_data_size();
        
        // Read data
        data_.resize(cur_element_count_ * dim_);
        index_to_label_.resize(cur_element_count_);
        for (size_t i = 0; i < cur_element_count_; i++) {
            readBinaryPOD(in, index_to_label_[i]);
            label_lookup_[index_to_label_[i]] = i;
            in.read((char*)(data_.data() + i * dim_), data_size_);
        }
        
        // Read hierarchy
        for (size_t level = 0; level < num_levels_; level++) {
            size_t num_elements;
            readBinaryPOD(in, num_elements);
            elements_[level].resize(num_elements);
            
            for (size_t i = 0; i < num_elements; i++) {
                readBinaryPOD(in, elements_[level][i].label);
                readBinaryPOD(in, elements_[level][i].outlink);
                readBinaryPOD(in, elements_[level][i].outlink_distance);
                
                size_t num_inlinks;
                readBinaryPOD(in, num_inlinks);
                elements_[level][i].inlinks.resize(num_inlinks);
                for (size_t j = 0; j < num_inlinks; j++) {
                    readBinaryPOD(in, elements_[level][i].inlinks[j]);
                }
                
                elements_[level][i].level = level;
                element_lookup_[level][elements_[level][i].label] = i;
            }
            
            size_t num_clusters;
            readBinaryPOD(in, num_clusters);
            clusters_[level].resize(num_clusters);
            
            for (size_t i = 0; i < num_clusters; i++) {
                readBinaryPOD(in, clusters_[level][i].id);
                readBinaryPOD(in, clusters_[level][i].representative);
                readBinaryPOD(in, clusters_[level][i].parent);
                readBinaryPOD(in, clusters_[level][i].radius);
                
                size_t num_members;
                readBinaryPOD(in, num_members);
                clusters_[level][i].members.resize(num_members);
                for (size_t j = 0; j < num_members; j++) {
                    readBinaryPOD(in, clusters_[level][i].members[j]);
                }
                
                clusters_[level][i].level = level;
            }
        }
        
        in.close();
    }
    
 private:
    // Compute distance between query and a labeled point
    dist_t computeDistance(const void* query_data, labeltype label) const {
        size_t internal_id = label_lookup_.at(label);
        const void* data_point = data_.data() + internal_id * dim_;
        return fstdistfunc_(query_data, data_point, dist_func_param_);
    }
    
    // Get entry point for search (root representative)
    labeltype getEntryPoint() const {
        if (num_levels_ == 0 || clusters_[num_levels_ - 1].empty()) {
            return index_to_label_[0];  // Return first element
        }
        return clusters_[num_levels_ - 1][0].representative;
    }
    
    // Add element to hierarchy at given level
    void addToHierarchy(labeltype label, size_t level) {
        // Create element at this level
        Element new_elem;
        new_elem.label = label;
        new_elem.level = level;
        new_elem.outlink = INVALID_LABEL;
        new_elem.outlink_distance = std::numeric_limits<dist_t>::max();
        
        // Find nearest neighbor at this level
        labeltype nn_label = INVALID_LABEL;
        dist_t nn_dist = std::numeric_limits<dist_t>::max();
        
        for (const auto& elem : elements_[level]) {
            if (elem.label == label) continue;  // Skip self
            
            size_t internal_id = label_lookup_.at(label);
            const void* query_data = data_.data() + internal_id * dim_;
            dist_t dist = computeDistance(query_data, elem.label);
            
            if (dist < nn_dist) {
                nn_dist = dist;
                nn_label = elem.label;
            }
        }
        
        // Update links
        if (nn_label != INVALID_LABEL) {
            new_elem.outlink = nn_label;
            new_elem.outlink_distance = nn_dist;
            
            // Add inlink to nearest neighbor
            size_t nn_idx = element_lookup_[level][nn_label];
            elements_[level][nn_idx].inlinks.push_back(label);
        }
        
        // Add element to level
        size_t elem_idx = elements_[level].size();
        elements_[level].push_back(new_elem);
        element_lookup_[level][label] = elem_idx;
        
        // Update or create cluster
        if (nn_label != INVALID_LABEL) {
            // Find cluster containing nearest neighbor
            size_t cluster_idx = findClusterWithMember(level, nn_label);
            if (cluster_idx != static_cast<size_t>(-1)) {
                // Add to existing cluster
                clusters_[level][cluster_idx].members.push_back(label);
                updateClusterRadius(level, cluster_idx);
            } else {
                // Create new cluster with both elements
                createCluster(level, {nn_label, label});
            }
        } else {
            // First element - create new cluster
            createCluster(level, {label});
        }
        
        // Check if we need to build upper level
        if (clusters_[level].size() > 1 && level + 1 < MAX_LEVELS) {
            buildUpperLevel(level);
        }
    }
    
    // Find cluster containing a specific member
    size_t findClusterWithMember(size_t level, labeltype label) const {
        for (size_t i = 0; i < clusters_[level].size(); i++) {
            for (labeltype member : clusters_[level][i].members) {
                if (member == label) return i;
            }
        }
        return static_cast<size_t>(-1);
    }
    
    // Create new cluster
    void createCluster(size_t level, const std::vector<labeltype>& members) {
        Cluster cluster;
        cluster.id = clusters_[level].size();
        cluster.level = level;
        cluster.members = members;
        
        // Select representative (first member for simplicity, can be improved)
        if (!members.empty()) {
            cluster.representative = members[0];
        }
        
        cluster.radius = 0.0;
        clusters_[level].push_back(cluster);
        
        updateClusterRadius(level, cluster.id);
    }
    
    // Update cluster radius
    void updateClusterRadius(size_t level, size_t cluster_idx) {
        Cluster& cluster = clusters_[level][cluster_idx];
        if (cluster.members.empty()) return;
        
        size_t rep_id = label_lookup_.at(cluster.representative);
        const void* rep_data = data_.data() + rep_id * dim_;
        
        dist_t max_dist = 0.0;
        for (labeltype member : cluster.members) {
            if (member == cluster.representative) continue;
            dist_t dist = computeDistance(rep_data, member);
            if (dist > max_dist) {
                max_dist = dist;
            }
        }
        
        cluster.radius = max_dist;
    }
    
    // Build upper level from current level's clusters
    void buildUpperLevel(size_t current_level) {
        if (current_level + 1 >= MAX_LEVELS) return;
        if (clusters_[current_level].size() <= 1) return;
        
        size_t next_level = current_level + 1;
        
        // Add representatives to next level
        for (const auto& cluster : clusters_[current_level]) {
            if (element_lookup_[next_level].find(cluster.representative) == element_lookup_[next_level].end()) {
                addToHierarchy(cluster.representative, next_level);
            }
        }
        
        num_levels_ = std::max(num_levels_, next_level + 1);
    }
};

}  // namespace hnnglib
