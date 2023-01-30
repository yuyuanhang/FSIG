/**
 * @file MMatch.cu
 * @brief contains a logic class MMatch that enumerates embeddings
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#include "MMatch.h"
#include "MMatch_kernels.cuh"
#include <cub/cub.cuh>
#include <thrust/scan.h>

void MMatch::copy_graph_to_GPU() {
    if (device_memory_manager_->allocate_memory(d_data_labels_, sizeof(Label)*data_size_, "vertex labels")) {
        cudaMemcpy(d_data_labels_, csr_data_graph_->labels_, sizeof(Label)*data_size_, cudaMemcpyHostToDevice);
    } else {
        exit(0);
    }

    if (device_memory_manager_->allocate_memory(d_data_offset_, sizeof(Edge)*(data_size_+1), "offset array")) {
        cudaMemcpy(d_data_offset_, csr_data_graph_->offset_, sizeof(Edge)*(data_size_+1), cudaMemcpyHostToDevice);
    } else {
        exit(0);
    }

    if (device_memory_manager_->allocate_memory(d_data_edges_, sizeof(Vertex)*csr_data_graph_->num_edges()*2, "edge array")) {
        cudaMemcpy(d_data_edges_, csr_data_graph_->edges_, sizeof(Vertex)*csr_data_graph_->num_edges()*2, cudaMemcpyHostToDevice);
    } else {
        exit(0);
    }
}

bool MMatch::exist_a_mapping(vector<Vertex> src_neighbors, vector<Vertex> dest_neighbors, vector<pair<Vertex, Vertex>> inversed_classes_) {
    for (Vertex i = 0; i < src_neighbors.size(); i++) {
        if (inversed_classes_[src_neighbors[i]].first != inversed_classes_[dest_neighbors[i]].first 
                || plain_query_graph_->labels_[src_neighbors[i]] != plain_query_graph_->labels_[dest_neighbors[i]]) {
            return false;
        }
    }
    return true;
}

void MMatch::find_equivalent_classes() {
    // initialise equivalent groups
    equivalent_classes_.push_back(vector<Vertex>(1, 0));
    inversed_classes_.resize(query_size_);
    inversed_classes_[0] = std::make_pair(0, 0);
    for (Vertex i = 1; i < query_size_; i++) {
        bool is_equivalent = false;
        for (Vertex j = 0; j < equivalent_classes_.size(); j++) {
            // a basic condition of equivalence: degree and label
            if (plain_query_graph_->degree(i) == plain_query_graph_->degree(equivalent_classes_[j][0]) && 
                    plain_query_graph_->labels_[i] == plain_query_graph_->labels_[equivalent_classes_[j][0]]) {
                is_equivalent = true;
                equivalent_classes_[j].push_back(i);
                inversed_classes_[i] = std::make_pair(j, equivalent_classes_[j].size() - 1);

            }
        }
        if (!is_equivalent) {
            equivalent_classes_.push_back(vector<Vertex>(1, i));
            inversed_classes_[i] = std::make_pair(equivalent_classes_.size() - 1, 0);
        }
    }

#ifdef DEBUG
    for (Vertex i = 0; i < equivalent_classes_.size(); i++) {
        cout << "group " << i << ": ";
        for (Vertex j = 0; j < equivalent_classes_[i].size(); j++) {
            cout << equivalent_classes_[i][j] << " ";
        }
        cout << endl;
    }
#endif

    // iteratively check neighbor
    // mainly on these vertices in equivalent classes including more than 1 vertice
    vector<Vertex> checked_vertices;
    for (auto equivalent_class : equivalent_classes_) {
        if (equivalent_class.size() > 1) {
            for (Vertex i = 1; i < equivalent_class.size(); i++) {
                checked_vertices.push_back(equivalent_class[i]);
            }
        }
    }
    Vertex num_old_equivalent_classes_ = equivalent_classes_.size();
    while (num_old_equivalent_classes_ != query_size_ && !checked_vertices.empty()) {
        vector<Vertex> new_checked_vertices;
        for (auto checked_vertex : checked_vertices) {
            // this class cannot be splitted
            if (equivalent_classes_[inversed_classes_[checked_vertex].first].size() == 1) {
                continue;
            }
            // ensure the compared target
            Vertex target = 0;
            if (inversed_classes_[checked_vertex].second == 0) {
                target = 1;
            }
            // to find a possible map
            vector<Vertex> crt_neighbors = plain_query_graph_->neighbors(checked_vertex);
            vector<Vertex> target_neighbors = plain_query_graph_->neighbors(equivalent_classes_[inversed_classes_[checked_vertex].first][target]);
            bool is_equivalent = false;
            if (crt_neighbors.size() == target_neighbors.size()) {
                do {
                    if (exist_a_mapping(crt_neighbors, target_neighbors, inversed_classes_)) {
                        is_equivalent = true;
                        break;
                    }
                } while (next_permutation(target_neighbors.begin(), target_neighbors.end()));
            }
            // different from current equivalent class
            if (!is_equivalent) {
                // delete current query vertex from this equivalent class
                vector<Vertex> crt_equivalent_class = equivalent_classes_[inversed_classes_[checked_vertex].first];
                crt_equivalent_class.erase(crt_equivalent_class.begin() + inversed_classes_[checked_vertex].second);
                equivalent_classes_[inversed_classes_[checked_vertex].first] = crt_equivalent_class;
                // adjust the position index of other vertices
                for (Vertex i = 0; i < crt_equivalent_class.size(); i++) {
                    Vertex vertex_id = crt_equivalent_class[i];
                    inversed_classes_[vertex_id] = std::make_pair(inversed_classes_[checked_vertex].first, i);
                }
                // there may exist new equivalent classes generated in this round
                // check if vertex checked_vertex belongs to new equivalent classes
                for (Vertex i = num_old_equivalent_classes_; i < equivalent_classes_.size(); i++) {
                    if (plain_query_graph_->labels_[checked_vertex] != plain_query_graph_->labels_[equivalent_classes_[i][0]]) {
                        continue;
                    }
                    // currently, the equivalent class of checked_vertex is null
                    // to enable computation, assume that vertex checked_vertex belongs to this equivalent group;
                    inversed_classes_[checked_vertex].first = i;
                    target_neighbors = plain_query_graph_->neighbors(equivalent_classes_[i][0]);
                    if (crt_neighbors.size() == target_neighbors.size()) {
                        do {
                            if (exist_a_mapping(crt_neighbors, target_neighbors, inversed_classes_)) {
                                is_equivalent = true;
                                break;
                            }
                        } while (next_permutation(target_neighbors.begin(), target_neighbors.end()));
                    }
                    // vertex checked_vertex belongs to this equivalent group
                    if (is_equivalent) {
                        equivalent_classes_[i].push_back(checked_vertex);
                        inversed_classes_[checked_vertex].second = equivalent_classes_[i].size() - 1;
                        break;
                    }
                }
                // vertex checked_vertex is a new equivalent group
                if (!is_equivalent) {
                    equivalent_classes_.push_back(vector<Vertex>(1, checked_vertex));
                    inversed_classes_[checked_vertex] = std::make_pair(equivalent_classes_.size() - 1, 0);
                }
                // the class of checked_vertex changes, and this affects the class of its neighbors
                for (auto crt_neighbor : crt_neighbors) {
                    if (find(new_checked_vertices.begin(), new_checked_vertices.end(), crt_neighbor) == new_checked_vertices.end()) {
                        new_checked_vertices.push_back(crt_neighbor);
                    }
                }
            }
        }
        num_old_equivalent_classes_ = equivalent_classes_.size();
        checked_vertices = new_checked_vertices;
    }
#ifdef DEBUG
    for (Vertex i = 0; i < equivalent_classes_.size(); i++) {
        cout << "group " << i << ": ";
        for (Vertex j = 0; j < equivalent_classes_[i].size(); j++) {
            cout << equivalent_classes_[i][j] << " ";
        }
        cout << endl;
    }
#endif
}

bool basic_operator_sorter(BasicOperator const &u, BasicOperator const &v) {
    return u.target_compressed_vertex_ < v.target_compressed_vertex_;
}

bool check_optimizable(vector<BasicOperator> a_, vector<BasicOperator> b_, Vertex* equivalent_class_type) {
    Vertex reduced_intersection_count = 0;
    for (Vertex i = 0; i < b_.size(); i++) {
        Vertex compressed_vertex = b_[i].target_compressed_vertex_;
        Vertex count = b_[i].count_;
        bool is_find = false;
        for (Vertex j = 0; j < a_.size(); j++) {
            if (a_[j].target_compressed_vertex_ == compressed_vertex) {
                if (a_[j].count_ < count) {
                    return false;
                }
                is_find = true;
                if (a_[j].count_ == count && equivalent_class_type[compressed_vertex] == 2) {
                    reduced_intersection_count++;
                }
                break;
            } else if (a_[j].target_compressed_vertex_ > compressed_vertex) {
                break;
            }
        }
        if (!is_find) {
            return false;
        }
    }
    if (reduced_intersection_count == 0) {
        return false;
    } else {
        return true;
    }
}

bool find_basic_operator(BasicOperator element, vector<BasicOperator> vec_operators) {
    for (auto operator_ : vec_operators) {
        if (element.target_compressed_vertex_ == operator_.target_compressed_vertex_ && 
                element.count_ == operator_.count_) {
            return true;
        }
    }
    return false;
}

vector<BasicOperator> remove_duplicate(vector<vector<BasicOperator>> &operator_multiset) {
    vector<BasicOperator> operator_set = operator_multiset[0];
    for (Vertex i = 1; i < operator_multiset.size(); i++) {
        vector<BasicOperator> operators = operator_multiset[i];
        for (auto operator_ : operators) {
            bool is_find = false;
            for (Vertex j = 0; j < operator_set.size(); j++) {
                if (operator_.target_compressed_vertex_ == operator_set[j].target_compressed_vertex_) {
                    is_find = true;
                    if (operator_.count_ > operator_set[j].count_) {
                        operator_set[j].count_ = operator_.count_;
                    }
                    break;
                }
            }
            if (!is_find) {
                operator_set.push_back(operator_);
            }
        }
    }
    return operator_set;
}

void insert_operator_set(Vertex idx, vector<BasicOperator> &set) {
    bool is_find = false;
    for (Vertex i = 0; i < set.size(); i++) {
        if (set[i].target_compressed_vertex_ == idx) {
            is_find = true;
            set[i].count_++;
            break;
        }
    }
    if (!is_find) {
        BasicOperator new_element;
        new_element.target_compressed_vertex_ = idx;
        new_element.count_ = 1;
        new_element.is_optimized_ = false;
        set.push_back(new_element);
    }
}

pair<bool, BasicOperator> find_operator_set(Vertex idx, vector<BasicOperator> set) {
    for (Vertex i = 0; i < set.size(); i++) {
        if (set[i].target_compressed_vertex_ == idx) {
            return make_pair(true, set[i]);
        }
    }
    return make_pair(false, set[0]);
}

double MMatch::compute_weight(Vertex root) {
    bool* is_visited = new bool[query_size_];
    memset(is_visited, 0, sizeof(bool)*query_size_);
    is_visited[root] = true;
    Vertex num_visited_vertices = 1;

    double weight = 0;
    vector<Vertex> frontier;
    frontier.push_back(root);
    Vertex level = 0;
    while (num_visited_vertices < query_size_) { // bfs traversal
        vector<Vertex> next_frontier;
        for (auto vertex_ : frontier) {
            for (auto neighbor : plain_query_graph_->neighbors(vertex_)) {
                if (!is_visited[neighbor]) {
                    next_frontier.push_back(neighbor);
                    is_visited[neighbor] = true;
                }
            }
        }
        weight += next_frontier.size() * (level + 1);
        level++;
        num_visited_vertices += next_frontier.size();
        frontier = next_frontier;
    }
    return weight;
}

Vertex MMatch::select_min_spanning_tree() {
    vector<Vertex> candidates;
    for (Vertex i = 0; i < query_size_; i++) {
        if (query_vertex_type_[i] == 1) {
            candidates.push_back(i);
        }
    }
    double min_weight = compute_weight(candidates[0]);
    Vertex root = candidates[0];
    for (Vertex i = 1; i < candidates.size(); i++) {
        double weight = compute_weight(candidates[i]);
        if (weight < min_weight) {
            min_weight = weight;
            root = candidates[i];
        }
    }
    return root;
}

void MMatch::generate_refine_plan() {
    find_equivalent_classes();
    num_equivalent_classes_ = equivalent_classes_.size();
    if (num_equivalent_classes_ == query_size_) {
        cout << "noequivalentclass" << endl;
    } else {
        cout << "haveequivalentclass" << endl;
    }
    equivalent_class_type_ = new Vertex[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        equivalent_class_type_[i] = query_vertex_type_[equivalent_classes_[i][0]];
    }

    forest_refine_plan_.resize(num_equivalent_classes_);
    core_refine_plan_.resize(num_equivalent_classes_);
    original_core_refine_plan_.resize(num_equivalent_classes_);

    vector<vector<BasicOperator>> compressed_query_graph(num_equivalent_classes_);
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        auto neighbors = plain_query_graph_->neighbors(equivalent_classes_[i][0]);
        for (auto neighbor : neighbors) {
            Vertex target_compressed_vertex = inversed_classes_[neighbor].first;
            insert_operator_set(target_compressed_vertex, compressed_query_graph[i]);
        }
    }
    // collect refine plans for query vertices in 2-core
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] != 2) {
            continue;
        }
        for (auto compressed_neighbor : compressed_query_graph[i]) {
            Vertex compressed_neighbor_type = equivalent_class_type_[compressed_neighbor.target_compressed_vertex_];
            if (compressed_neighbor_type == 2) {
                core_refine_plan_[i].push_back(compressed_neighbor);
            }
        }
    }
    original_core_refine_plan_ = core_refine_plan_;
    // collect forest refine plan for query vertices in forest
    has_core_ = false;
    vector<Vertex> ancestors;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 2) {
            has_core_ = true;
            ancestors.push_back(i);
        }
    }
    if (!has_core_) {
        Vertex root = select_min_spanning_tree();
        ancestors.push_back(inversed_classes_[root].first);
    }
    bool* is_visited = new bool[num_equivalent_classes_];
    memset(is_visited, 0, sizeof(bool)*num_equivalent_classes_);
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        for (auto vertex_ : ancestors) {
            is_visited[vertex_] = true;
        }
    }
    forest_order_.push_back(ancestors);
    while (!ancestors.empty()) {
        vector<Vertex> successors;
        for (auto vertex_ : ancestors) {
            for (auto compressed_neighbor : compressed_query_graph[vertex_]) {
                if (!is_visited[compressed_neighbor.target_compressed_vertex_]) {
                    forest_refine_plan_[vertex_].push_back(compressed_neighbor);
                    is_visited[compressed_neighbor.target_compressed_vertex_] = true;
                    successors.push_back(compressed_neighbor.target_compressed_vertex_);
                } else if (compressed_neighbor.target_compressed_vertex_ == vertex_ && equivalent_class_type_[vertex_] != 2) { // self loop in forest
                    forest_refine_plan_[vertex_].push_back(compressed_neighbor);
                }
            }
        }
        if (!successors.empty()) {
            forest_order_.push_back(successors);
        }
        ancestors = successors;
    }
#ifdef DEBUG
    cout << "core compressed edges" << endl;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        cout << "class " << i << ": ";
        for (Vertex j = 0; j < core_refine_plan_[i].size(); j++) {
            cout << "(" << core_refine_plan_[i][j].target_compressed_vertex_ << ", " << 
                core_refine_plan_[i][j].count_ << ", " << 
                core_refine_plan_[i][j].is_optimized_ << ") ";
        }
        cout << endl;
    }
    cout << "forest compressed edges" << endl;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        cout << "class " << i << ": ";
        for (Vertex j = 0; j < forest_refine_plan_[i].size(); j++) {
            cout << "(" << forest_refine_plan_[i][j].target_compressed_vertex_ << ", " << 
                forest_refine_plan_[i][j].count_ << ", " << 
                forest_refine_plan_[i][j].is_optimized_ << ") ";
        }
        cout << endl;
    }
    cout << "forest structure" << endl;
    for (Vertex i = 0; i < forest_order_.size(); i++) {
        cout << "layer " << i << ": ";
        for (Vertex j = 0; j < forest_order_[i].size(); j++) {
            cout << forest_order_[i][j] << " ";
        }
        cout << endl;
    }
#endif
    // sort neighbors in refine plan
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        sort(compressed_query_graph[i].begin(), compressed_query_graph[i].end(), &basic_operator_sorter);
    }
    // find optimizable refine plan for vertices in 2-core
    bool is_optimizable[num_equivalent_classes_];
    memset(is_optimizable, 0, sizeof(bool)*num_equivalent_classes_);
    vector<vector<BasicOperator>> optimized_plan(num_equivalent_classes_);
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] != 2) {
            continue;
        }
        vector<Vertex> optimizable_units;
        // find potential optimizable units
        for (Vertex j = 0; j < num_equivalent_classes_; j++) {
            if (i == j || compressed_query_graph[j].size() > compressed_query_graph[i].size() || 
                    equivalent_class_type_[j] != 2 || plain_query_graph_->labels_[equivalent_classes_[i][0]] != plain_query_graph_->labels_[equivalent_classes_[j][0]]) {
                continue;
            }
            if (check_optimizable(compressed_query_graph[i], compressed_query_graph[j], equivalent_class_type_)) {
                optimizable_units.push_back(j);
            }
        }
        // forms optimized refine plan via units
        if (optimizable_units.empty()) { // no optimized refine plan
            is_optimizable[i] = false;
            continue;
        } else {
            is_optimizable[i] = true;
            if (optimizable_units.size() == 1) { // only one choice
                BasicOperator optimized_operator;
                optimized_operator.target_compressed_vertex_ = optimizable_units[0];
                optimized_operator.count_ = 0;
                optimized_operator.is_optimized_ = true;
                optimized_plan[i].push_back(optimized_operator);

                for (auto basic_operator : core_refine_plan_[i]) {
                    if (!find_basic_operator(basic_operator, core_refine_plan_[optimizable_units[0]])) {
                        optimized_plan[i].push_back(basic_operator);
                    }
                }
            } else { // there may have multiple combinations
                int num_subsets = (1 << optimizable_units.size());
                Vertex num_check = 0;
                Vertex num_intersection = core_refine_plan_[i].size();
                auto subsets = [](vector<Vertex> units, int bitset) {
                    vector<Vertex> subset;
                    int k = units.size();
                    for (int i = 0; i < k; i++) {
                        if ((1 << i) & bitset) {
                            subset.push_back(units[i]);
                        }
                    }
                    return subset;
                };
                for (int j = 1; j < num_subsets; j++) {
                    // generate an optimized refine plan
                    vector<Vertex> units_combination = subsets(optimizable_units, j);
                    vector<vector<BasicOperator>> operator_conbination;
                    for (auto idx : units_combination) {
                        operator_conbination.push_back(core_refine_plan_[idx]);
                    }
                    // get the union of unit set
                    vector<BasicOperator> operator_set = remove_duplicate(operator_conbination);
                    // compute num of intersection and check
                    Vertex crt_num_intersection = 0;
                    Vertex crt_num_check = units_combination.size();
                    vector<BasicOperator> supplement;
                    for (auto basic_operator : core_refine_plan_[i]) {
                        if (!find_basic_operator(basic_operator, operator_set)) {
                            supplement.push_back(basic_operator);
                            crt_num_intersection++;
                        }
                    }
                    for (auto idx : units_combination) {
                        BasicOperator optimized_operator;
                        optimized_operator.target_compressed_vertex_ = idx;
                        optimized_operator.count_ = 0;
                        optimized_operator.is_optimized_ = true;
                        supplement.push_back(optimized_operator);
                    }
                    if (crt_num_intersection < num_intersection || 
                            (crt_num_intersection == num_intersection && crt_num_check < num_check)) {
                        num_intersection = crt_num_intersection;
                        num_check = crt_num_check;
                        optimized_plan[i] = supplement;
                    }
                }
            }
        }
    }

    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (is_optimizable[i]) {
            core_refine_plan_[i] = optimized_plan[i];
        }
    }

    cout << "optimizable operators" << endl;
    bool is_optimized = false;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (is_optimizable[i]) {
            is_optimized = true;
            cout << "class " << i << ": ";
            for (Vertex j = 0; j < optimized_plan[i].size(); j++) {
                cout << "(" << optimized_plan[i][j].target_compressed_vertex_ << ", " << 
                    optimized_plan[i][j].count_ << ", " << 
                    optimized_plan[i][j].is_optimized_ << ") ";
            }
            cout << endl;
        }
    }
    if (is_optimized) {
        cout << "localoptimized" << endl;
    } else {
        cout << "nolocaloptimized" << endl;
    }
}

void MMatch::decompose_query_graph() {
    query_vertex_type_ = new Vertex[query_size_];
    queue<Vertex> leaves;
    vector<Vertex> inversed_group(query_size_);
    for (Vertex i = 0; i < query_size_; i++) {
        Vertex crt_degree = plain_query_graph_->degree(i);
        inversed_group[i] = crt_degree;
        if (crt_degree == 1) {
            leaves.push(i);
        }
    }

    // keep removing vertices with degree 1
    while (!leaves.empty()) {
        Vertex crt_vertex = leaves.front();
        auto neighbors = plain_query_graph_->neighbors(crt_vertex);
        for (auto neighbor : neighbors) {
            Vertex neighbor_degree = inversed_group[neighbor];
            if (neighbor_degree == 0) {
                continue;
            }
            inversed_group[neighbor]--;
            if (neighbor_degree == 2) {
                leaves.push(neighbor);
            }
        }
        leaves.pop();
        inversed_group[crt_vertex] = 0;
    }

    for (Vertex i = 0; i < query_size_; i++) {
        if (inversed_group[i] >= 2) {
            query_vertex_type_[i] = 2;
        } else {
            if (plain_query_graph_->degree(i) == 1) {
                query_vertex_type_[i] = 0;
            } else {
                query_vertex_type_[i] = 1;
            }
        }
    }
#ifdef DEBUG
    cout << "2-core: ";
    for (Vertex i = 0; i < query_size_; i++) {
        if (query_vertex_type_[i] == 2) {
            cout << i << " ";
        }
    }
    cout << endl;
    cout << "forest: ";
    for (Vertex i = 0; i < query_size_; i++) {
        if (query_vertex_type_[i] == 1) {
            cout << i << " ";
        }
    }
    cout << endl;
    cout << "leaves: ";
    for (Vertex i = 0; i < query_size_; i++) {
        if (query_vertex_type_[i] == 0) {
            cout << i << " ";
        }
    }
    cout << endl;
#endif
}

void MMatch::prepare_initialise() {
    // allocate candidate set for each equivalent class
    ptrs_d_candidate_bitmap_ = new unsigned*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (device_memory_manager_->allocate_memory(ptrs_d_candidate_bitmap_[i], sizeof(unsigned)*(data_size_+1), "bitmap of equivalent class " + to_string(i))) {
            cudaMemset(ptrs_d_candidate_bitmap_[i], 0, sizeof(unsigned)*(data_size_+1));
        } else {
            exit(0);
        }
    }
    // concurrently initialise candidate sets
    streams_ = new cudaStream_t[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        cudaStreamCreate(&streams_[i]);
    }
}

void MMatch::kernel_initialise() {
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (data_size_+BLOCK_SIZE-1) / BLOCK_SIZE;

    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        Vertex vertex_id = equivalent_classes_[i][0];
        Label vertex_label = plain_query_graph_->labels_[vertex_id];
        Vertex vertex_degree = plain_query_graph_->degree(vertex_id);
        initialise_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[i]>>>(d_data_labels_, d_data_offset_, ptrs_d_candidate_bitmap_[i], data_size_, vertex_label, vertex_degree);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void MMatch::prepare_collect() {
    ptrs_d_candidate_array_ = new Vertex*[num_equivalent_classes_];
    num_candidates_array_ = new unsigned[num_equivalent_classes_];
}

void MMatch::exclusive_sum(unsigned* d_array, unsigned* sum_array, Vertex size, unsigned &sum, cudaStream_t stream) {
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, sum_array, size, stream);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, sum_array, size, stream);
    cudaFree(d_temp_storage);

    cudaMemcpy(&sum, sum_array + size - 1, sizeof(unsigned), cudaMemcpyDeviceToHost);
}

void MMatch::exclusive_sum(long long unsigned* d_array, long long unsigned* sum_array, long long unsigned size, long long unsigned &sum, cudaStream_t stream) {
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, sum_array, size, stream);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, sum_array, size, stream);
    cudaFree(d_temp_storage);

    cudaMemcpy(&sum, sum_array + size - 1, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
}

void MMatch::kernel_collect() {
    // prefix sum array
    unsigned* d_exclusive_sum = NULL;
    if (device_memory_manager_->allocate_memory(d_exclusive_sum, sizeof(unsigned)*(data_size_+1)*num_equivalent_classes_, "exclusive prefix sum array")) {
        cudaMemset(d_exclusive_sum, 0, sizeof(unsigned)*(data_size_+1)*num_equivalent_classes_);
    } else {
        exit(0);
    }

    // before collecting candidate arrays, compute the number of candidates
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        exclusive_sum(ptrs_d_candidate_bitmap_[i], d_exclusive_sum + i * (data_size_ + 1), data_size_ + 1, num_candidates_array_[i], streams_[i]);
        cout << "group " << i << ": " << num_candidates_array_[i] << endl;
        device_memory_manager_->allocate_memory(ptrs_d_candidate_array_[i], sizeof(Vertex) * num_candidates_array_[i], "candidate array of equivalent class " + to_string(i));
    }

    // collect candidates
    int BLOCK_SIZE = 1024;
    int GRID_SIZE = (data_size_+BLOCK_SIZE-1) / BLOCK_SIZE;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        scatter_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[i]>>>(ptrs_d_candidate_bitmap_[i], d_exclusive_sum + i * (data_size_ + 1), ptrs_d_candidate_array_[i], data_size_, 1);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    device_memory_manager_->free_memory(d_exclusive_sum, sizeof(unsigned)*(data_size_+1)*num_equivalent_classes_, "exclusive prefix sum array");
}

void MMatch::prepare_filter() {
    device_memory_manager_->copy_ptrs_to_device(d_ptrs_d_candidate_bitmap_, ptrs_d_candidate_bitmap_, num_equivalent_classes_, "bitmap pointers on device");

    ptrs_query_neighbors_d_count_ = new unsigned**[num_equivalent_classes_]; // each pointer is for one class
    ptrs_d_query_neighbors_count_ = new unsigned**[num_equivalent_classes_];
    ptrs_d_query_neighbors_count_threshold_ = new unsigned*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 0) { // a leaf
            ptrs_query_neighbors_d_count_[i] = NULL;
            ptrs_d_query_neighbors_count_[i] = NULL;
            ptrs_d_query_neighbors_count_threshold_[i] = NULL;
        } else {
            ptrs_query_neighbors_d_count_[i] = new unsigned*[num_equivalent_classes_];
            for (Vertex j = 0; j < num_equivalent_classes_; j++) {
                ptrs_query_neighbors_d_count_[i][j] = NULL;
            }

            unsigned* h_count_threshold = new unsigned[num_equivalent_classes_];
            memset(h_count_threshold, 0, sizeof(unsigned)*num_equivalent_classes_);

            for (auto forest_operator : forest_refine_plan_[i]) {
                Vertex successor = forest_operator.target_compressed_vertex_;
                h_count_threshold[successor] = forest_operator.count_;
                if (device_memory_manager_->allocate_memory(ptrs_query_neighbors_d_count_[i][successor], 
                        sizeof(unsigned)*(data_size_+1), "data neighbor count for (" + to_string(i) + ", " + to_string(successor) + ")")) {
                    cudaMemset(ptrs_query_neighbors_d_count_[i][successor], 0, sizeof(unsigned)*(data_size_+1));
                } else {
                    exit(0);
                }
            }
            for (auto core_operator : core_refine_plan_[i]) {
                if (!core_operator.is_optimized_) {
                    Vertex core_neighbor = core_operator.target_compressed_vertex_;
                    h_count_threshold[core_neighbor] = core_operator.count_;
                    if (device_memory_manager_->allocate_memory(ptrs_query_neighbors_d_count_[i][core_neighbor], 
                            sizeof(unsigned)*(data_size_+1), "data neighbor count for (" + to_string(i) + ", " + to_string(core_neighbor) + ")")) {
                        cudaMemset(ptrs_query_neighbors_d_count_[i][core_neighbor], 0, sizeof(unsigned)*(data_size_+1));
                    } else {
                        exit(0);
                    }
                }
            }
            device_memory_manager_->copy_ptrs_to_device(ptrs_d_query_neighbors_count_[i], ptrs_query_neighbors_d_count_[i], 
                num_equivalent_classes_, "data neighbor count pointers of " + to_string(i) + " on device");
            if (device_memory_manager_->allocate_memory(ptrs_d_query_neighbors_count_threshold_[i], sizeof(unsigned)*num_equivalent_classes_, "count threshold of " + to_string(i))) {
                cudaMemcpy(ptrs_d_query_neighbors_count_threshold_[i], h_count_threshold, sizeof(unsigned)*num_equivalent_classes_, cudaMemcpyHostToDevice);
            } else {
                exit(0);
            }
            delete[] h_count_threshold;
        }
    }
    //
    ptrs_d_candidate_bitmap_next_iterartion_ = new unsigned*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 2) {
            if (device_memory_manager_->allocate_memory(ptrs_d_candidate_bitmap_next_iterartion_[i], sizeof(unsigned)*(data_size_+1), 
                    "bitmap of equivalent class " + to_string(i) + " in next iteration")) {
                cudaMemcpyAsync(ptrs_d_candidate_bitmap_next_iterartion_[i], ptrs_d_candidate_bitmap_[i], 
                    sizeof(unsigned)*(data_size_+1), cudaMemcpyDeviceToDevice, streams_[i]);
            }
        } else {
            ptrs_d_candidate_bitmap_next_iterartion_[i] = NULL;
        }
    }
    //
    ptrs_d_candidate_array_next_iteration_ = new Vertex*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] != 0) {
            device_memory_manager_->allocate_memory(ptrs_d_candidate_array_next_iteration_[i], sizeof(Vertex) * num_candidates_array_[i], 
                "candidate array of equivalent class " + to_string(i) + " in the next iteration");
        } else {
            ptrs_d_candidate_array_next_iteration_[i] = NULL;
        }
    }
    //
    ptrs_d_invalid_candidate_array_ = new Vertex*[num_equivalent_classes_];
    ptrs_d_invalid_candidate_array_next_iteration_ = new Vertex*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 2) {
            device_memory_manager_->allocate_memory(ptrs_d_invalid_candidate_array_[i], sizeof(Vertex) * num_candidates_array_[i], 
                "invalid candidate array of equivalent class " + to_string(i));
            device_memory_manager_->allocate_memory(ptrs_d_invalid_candidate_array_next_iteration_[i], sizeof(Vertex) * num_candidates_array_[i], 
                "invalid candidate array of equivalent class " + to_string(i) + " in the next iteration");
        } else {
            ptrs_d_invalid_candidate_array_[i] = NULL;
            ptrs_d_invalid_candidate_array_next_iteration_[i] = NULL;
        }
    }
    num_invalid_candidates_array_ = new unsigned[num_equivalent_classes_];
    //
    device_memory_manager_->copy_ptrs_to_device(d_ptrs_d_query_neighbors_count_, ptrs_d_query_neighbors_count_, num_equivalent_classes_, 
        "device pointers of all data neighbor count pointers on device");
}

void MMatch::collect_successors(Vertex vertex_id, Vertex &num_successors, Vertex* &d_successors) {
    num_successors = forest_refine_plan_[vertex_id].size();
    Vertex* successors = new Vertex[num_successors];
    device_memory_manager_->allocate_memory(d_successors, sizeof(Vertex)*num_successors, "successors of " + to_string(vertex_id));
    for (Vertex i = 0; i < num_successors; i++) {
        successors[i] = forest_refine_plan_[vertex_id][i].target_compressed_vertex_;
    }
    cudaMemcpy(d_successors, successors, sizeof(Vertex)*num_successors, cudaMemcpyHostToDevice);
    delete[] successors;
}

void MMatch::kernel_filter_forest() {
    int BLOCK_SIZE;
    int GRID_SIZE;

    unsigned** d_flag = new unsigned*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        d_flag[i] = NULL;
    }
    
    for (int i = forest_order_.size() - 1; i >= 0; i--) {
        auto vertices_crt_layer = forest_order_[i];
        for (auto vertex_crt_layer : vertices_crt_layer) {
            if (equivalent_class_type_[vertex_crt_layer] != 0) { // not a leaf
                BLOCK_SIZE = 256;
                GRID_SIZE = (num_candidates_array_[vertex_crt_layer]*32+BLOCK_SIZE-1) / BLOCK_SIZE;

                Vertex num_successors;
                Vertex* d_successors;
                collect_successors(vertex_crt_layer, num_successors, d_successors);

                unsigned size_shared_memory = sizeof(Vertex)*BLOCK_SIZE + sizeof(unsigned)*BLOCK_SIZE/32*num_successors;
                count_neighbors_kernel<<<GRID_SIZE, BLOCK_SIZE, size_shared_memory, streams_[vertex_crt_layer]>>>(d_ptrs_d_candidate_bitmap_, d_data_offset_, 
                    d_data_edges_, ptrs_d_candidate_array_[vertex_crt_layer], num_candidates_array_[vertex_crt_layer], d_successors, num_successors, 
                    ptrs_d_query_neighbors_count_[vertex_crt_layer], vertex_crt_layer);

                BLOCK_SIZE = 1024;
                GRID_SIZE = (num_candidates_array_[vertex_crt_layer]+BLOCK_SIZE-1)/BLOCK_SIZE;
                check_re_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[vertex_crt_layer]>>>(ptrs_d_candidate_bitmap_[vertex_crt_layer], ptrs_d_candidate_array_[vertex_crt_layer], 
                    ptrs_d_query_neighbors_count_[vertex_crt_layer], ptrs_d_query_neighbors_count_threshold_[vertex_crt_layer], 
                    num_candidates_array_[vertex_crt_layer], d_successors, num_successors);

                if (device_memory_manager_->allocate_memory(d_flag[vertex_crt_layer], sizeof(unsigned) * (num_candidates_array_[vertex_crt_layer] + 1), 
                        "exclusive prefix sum array for " + to_string(vertex_crt_layer))) {
                    cudaMemset(d_flag[vertex_crt_layer], 0, sizeof(unsigned) * (num_candidates_array_[vertex_crt_layer] + 1));
                } else {
                    exit(0);
                }
                count_valid_candidate_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[vertex_crt_layer]>>>(ptrs_d_candidate_bitmap_[vertex_crt_layer], ptrs_d_candidate_array_[vertex_crt_layer], 
                    d_flag[vertex_crt_layer], num_candidates_array_[vertex_crt_layer]);

                Vertex num_valid_candidates = 0;
                exclusive_sum(d_flag[vertex_crt_layer], d_flag[vertex_crt_layer], num_candidates_array_[vertex_crt_layer] + 1, num_valid_candidates, streams_[vertex_crt_layer]);
                cout << "group " << vertex_crt_layer << " (remaining size): " << num_valid_candidates << endl;
                if (num_valid_candidates == 0) {
                    cout << "#embedding is 0 because of " << vertex_crt_layer << endl;
                    exit(0);
                }

                compact_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[vertex_crt_layer]>>>(d_flag[vertex_crt_layer], ptrs_d_candidate_array_[vertex_crt_layer], 
                    ptrs_d_candidate_array_next_iteration_[vertex_crt_layer], num_candidates_array_[vertex_crt_layer]);
                // change 2 to 0 so that count the number of valid candidates and compute prefix array
                cudaMemcpyAsync(ptrs_d_candidate_array_[vertex_crt_layer], ptrs_d_candidate_array_next_iteration_[vertex_crt_layer], 
                    sizeof(Vertex)*num_valid_candidates, cudaMemcpyDeviceToDevice, streams_[vertex_crt_layer]);
                GRID_SIZE = (data_size_+BLOCK_SIZE-1) / BLOCK_SIZE;
                change_flag_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[vertex_crt_layer]>>>(ptrs_d_candidate_bitmap_[vertex_crt_layer], data_size_);
                
                num_candidates_array_[vertex_crt_layer] = num_valid_candidates;
                device_memory_manager_->free_memory(d_successors, sizeof(Vertex)*num_successors, "successors of " + to_string(vertex_crt_layer));
                // sychronize bitmap
                if (equivalent_class_type_[vertex_crt_layer] == 2) {
                    cudaMemcpyAsync(ptrs_d_candidate_bitmap_next_iterartion_[vertex_crt_layer], ptrs_d_candidate_bitmap_[vertex_crt_layer], 
                        sizeof(unsigned)*(data_size_+1), cudaMemcpyDeviceToDevice, streams_[vertex_crt_layer]);
                }
            }
        }
        // next layer can begin only after current layer is finished because there exists dependency
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }
    //
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (d_flag[i] != NULL) {
            device_memory_manager_->free_memory(d_flag[i], "exclusive prefix sum array for " + to_string(i));
        }
    }
}

void MMatch::collect_core_neighbors(vector<Vertex> core_vertices, Vertex* &num_re_core_neighbors, Vertex* &num_op_core_neighbors, 
                    Vertex* &num_reversed_re_core_neighbors, Vertex** &ptrs_d_re_core_neighbors, Vertex** &ptrs_d_op_core_neighbors, 
                    Vertex** &ptrs_d_reversed_re_core_neighbors) {
    re_core_neighbors_.resize(num_equivalent_classes_);
    op_core_neighbors_.resize(num_equivalent_classes_);
    reversed_re_core_neighbors_.resize(num_equivalent_classes_);
    for (auto core_vertex : core_vertices) {
        for (auto operator_ : core_refine_plan_[core_vertex]) {
            if (operator_.is_optimized_) {
                op_core_neighbors_[core_vertex].push_back(operator_.target_compressed_vertex_);
            } else {
                re_core_neighbors_[core_vertex].push_back(operator_.target_compressed_vertex_);
                reversed_re_core_neighbors_[operator_.target_compressed_vertex_].push_back(core_vertex);
            }
        }
    }
    // 
    num_re_core_neighbors = new Vertex[num_equivalent_classes_];
    num_op_core_neighbors = new Vertex[num_equivalent_classes_];
    num_reversed_re_core_neighbors = new Vertex[num_equivalent_classes_];
    ptrs_d_re_core_neighbors = new Vertex*[num_equivalent_classes_];
    ptrs_d_op_core_neighbors = new Vertex*[num_equivalent_classes_];
    ptrs_d_reversed_re_core_neighbors = new Vertex*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        num_re_core_neighbors[i] = 0;
        num_op_core_neighbors[i] = 0;
        num_reversed_re_core_neighbors[i] = 0;
        ptrs_d_re_core_neighbors[i] = NULL;
        ptrs_d_op_core_neighbors[i] = NULL;
        ptrs_d_reversed_re_core_neighbors[i] = NULL;
    }
    for (auto core_vertex : core_vertices) {
        num_re_core_neighbors[core_vertex] = re_core_neighbors_[core_vertex].size();
        num_op_core_neighbors[core_vertex] = op_core_neighbors_[core_vertex].size();
        num_reversed_re_core_neighbors[core_vertex] = reversed_re_core_neighbors_[core_vertex].size();
        if (num_re_core_neighbors[core_vertex] != 0) {
            Vertex* crt_re_core_neighbors = new Vertex[num_re_core_neighbors[core_vertex]];

            copy(re_core_neighbors_[core_vertex].begin(), re_core_neighbors_[core_vertex].end(), crt_re_core_neighbors);

            device_memory_manager_->allocate_memory(ptrs_d_re_core_neighbors[core_vertex], sizeof(Vertex)*num_re_core_neighbors[core_vertex], "regular core neighbors of " + to_string(core_vertex));
            cudaMemcpy(ptrs_d_re_core_neighbors[core_vertex], crt_re_core_neighbors, sizeof(Vertex)*num_re_core_neighbors[core_vertex], cudaMemcpyHostToDevice);

            delete[] crt_re_core_neighbors;
        }
        if (num_op_core_neighbors[core_vertex] != 0) {
            Vertex* crt_op_core_neighbors = new Vertex[num_op_core_neighbors[core_vertex]];

            copy(op_core_neighbors_[core_vertex].begin(), op_core_neighbors_[core_vertex].end(), crt_op_core_neighbors);

            device_memory_manager_->allocate_memory(ptrs_d_op_core_neighbors[core_vertex], sizeof(Vertex)*num_op_core_neighbors[core_vertex], "optimized core neighbors of " + to_string(core_vertex));
            cudaMemcpy(ptrs_d_op_core_neighbors[core_vertex], crt_op_core_neighbors, sizeof(Vertex)*num_op_core_neighbors[core_vertex], cudaMemcpyHostToDevice);
            
            delete[] crt_op_core_neighbors;
        }
        if (num_reversed_re_core_neighbors[core_vertex] != 0) {
            Vertex* crt_reversed_re_core_neighbors = new Vertex[num_reversed_re_core_neighbors[core_vertex]];

            copy(reversed_re_core_neighbors_[core_vertex].begin(), reversed_re_core_neighbors_[core_vertex].end(), crt_reversed_re_core_neighbors);

            device_memory_manager_->allocate_memory(ptrs_d_reversed_re_core_neighbors[core_vertex], sizeof(Vertex)*num_reversed_re_core_neighbors[core_vertex], "reversed regular core neighbors of " + to_string(core_vertex));
            cudaMemcpy(ptrs_d_reversed_re_core_neighbors[core_vertex], crt_reversed_re_core_neighbors, sizeof(Vertex)*num_reversed_re_core_neighbors[core_vertex], cudaMemcpyHostToDevice);

            delete[] crt_reversed_re_core_neighbors;
        }
    }
}

void MMatch::perform_iteration(vector<Vertex> core_vertices, Vertex* num_re_core_neighbors, Vertex* num_op_core_neighbors, Vertex* num_reversed_re_core_neighbors, 
                    Vertex** ptrs_d_re_core_neighbors, Vertex** ptrs_d_op_core_neighbors, Vertex** ptrs_d_reversed_re_core_neighbors, bool is_pull) {
    int BLOCK_SIZE;
    int GRID_SIZE;
    // check regular neighbors here has a problem
    for (auto core_vertex : core_vertices) {
        BLOCK_SIZE = 256;
        unsigned size_shared_memory;
        if (num_re_core_neighbors[core_vertex] != 0 && is_pull) {
            GRID_SIZE = (num_candidates_array_[core_vertex]*32+BLOCK_SIZE-1)/BLOCK_SIZE;
            size_shared_memory = sizeof(Vertex)*BLOCK_SIZE + sizeof(unsigned)*BLOCK_SIZE/32*num_re_core_neighbors[core_vertex];

            count_neighbors_kernel<<<GRID_SIZE, BLOCK_SIZE, size_shared_memory, streams_[core_vertex]>>>(d_ptrs_d_candidate_bitmap_, d_data_offset_, 
                d_data_edges_, ptrs_d_candidate_array_[core_vertex], num_candidates_array_[core_vertex], ptrs_d_re_core_neighbors[core_vertex], 
                num_re_core_neighbors[core_vertex], ptrs_d_query_neighbors_count_[core_vertex], core_vertex);
        } else if (num_reversed_re_core_neighbors[core_vertex] != 0 && !is_pull) {
            GRID_SIZE = (num_invalid_candidates_array_[core_vertex]*32+BLOCK_SIZE-1)/BLOCK_SIZE;
            if (GRID_SIZE == 0) {
                continue;
            }
            size_shared_memory = sizeof(Vertex)*BLOCK_SIZE;

            alleviate_neighbors_kernel<<<GRID_SIZE, BLOCK_SIZE, size_shared_memory, streams_[core_vertex]>>>(d_ptrs_d_candidate_bitmap_, d_data_offset_, 
                d_data_edges_, ptrs_d_invalid_candidate_array_[core_vertex], num_invalid_candidates_array_[core_vertex], ptrs_d_reversed_re_core_neighbors[core_vertex], 
                num_reversed_re_core_neighbors[core_vertex], d_ptrs_d_query_neighbors_count_, core_vertex);
        }
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    //
    for (auto core_vertex : core_vertices) {
        BLOCK_SIZE = 1024;
        GRID_SIZE = (num_candidates_array_[core_vertex]+BLOCK_SIZE-1)/BLOCK_SIZE;
        check_re_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(ptrs_d_candidate_bitmap_[core_vertex], ptrs_d_candidate_array_[core_vertex], 
            ptrs_d_query_neighbors_count_[core_vertex], ptrs_d_query_neighbors_count_threshold_[core_vertex], 
            num_candidates_array_[core_vertex], ptrs_d_re_core_neighbors[core_vertex], num_re_core_neighbors[core_vertex]);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    // check optimized neighbors
    for (auto core_vertex : core_vertices) {
        if (num_op_core_neighbors[core_vertex] != 0) {
            BLOCK_SIZE = 1024;
            GRID_SIZE = (num_candidates_array_[core_vertex]+BLOCK_SIZE-1) / BLOCK_SIZE;
            check_op_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(d_ptrs_d_candidate_bitmap_, ptrs_d_candidate_array_[core_vertex], 
                ptrs_d_op_core_neighbors[core_vertex], num_candidates_array_[core_vertex], num_op_core_neighbors[core_vertex], ptrs_d_candidate_bitmap_next_iterartion_[core_vertex]);
            cudaMemcpyAsync(ptrs_d_candidate_bitmap_[core_vertex], ptrs_d_candidate_bitmap_next_iterartion_[core_vertex], 
                sizeof(unsigned)*(data_size_+1), cudaMemcpyDeviceToDevice, streams_[core_vertex]);
        }
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError()); // now we have splitted invalid data candidates of this iteration
}

void MMatch::decide_perform_mode(vector<Vertex> core_vertices, Vertex* num_re_core_neighbors, Vertex** ptrs_d_re_core_neighbors, unsigned** d_flag, bool &is_pull, bool &is_stop) {
    static int times = 0;
    int BLOCK_SIZE;
    int GRID_SIZE;

    Vertex* num_candidates_array_next_iteration = new Vertex[num_equivalent_classes_];
    for (auto core_vertex : core_vertices) {
        // copy candidate array for the computation of invalid candidate arrays
        cudaMemcpyAsync(ptrs_d_invalid_candidate_array_[core_vertex], ptrs_d_candidate_array_[core_vertex], 
                            num_candidates_array_[core_vertex]*sizeof(Vertex), cudaMemcpyDeviceToDevice, streams_[core_vertex]);
        // collect valid candidates in this iteration
        BLOCK_SIZE = 1024;
        GRID_SIZE = (num_candidates_array_[core_vertex]+BLOCK_SIZE-1)/BLOCK_SIZE;

        count_valid_candidate_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(ptrs_d_candidate_bitmap_[core_vertex], ptrs_d_candidate_array_[core_vertex], 
            d_flag[core_vertex], num_candidates_array_[core_vertex]);

        num_candidates_array_next_iteration[core_vertex] = 0;
        exclusive_sum(d_flag[core_vertex], d_flag[core_vertex], num_candidates_array_[core_vertex] + 1, 
            num_candidates_array_next_iteration[core_vertex], streams_[core_vertex]);
        cout << "group " << core_vertex << " (remaining size): " << num_candidates_array_next_iteration[core_vertex] << endl;
        if (num_candidates_array_next_iteration[core_vertex] == 0) {
            cout << "#embedding is 0 because of " << core_vertex << endl;
            exit(0);
        }

        compact_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(d_flag[core_vertex], ptrs_d_candidate_array_[core_vertex], 
            ptrs_d_candidate_array_next_iteration_[core_vertex], num_candidates_array_[core_vertex]);
        cudaMemcpyAsync(ptrs_d_candidate_array_[core_vertex], ptrs_d_candidate_array_next_iteration_[core_vertex], 
            sizeof(Vertex)*num_candidates_array_next_iteration[core_vertex], cudaMemcpyDeviceToDevice, streams_[core_vertex]);

        // collect invalid candidates in this iteration
        cudaMemsetAsync(d_flag[core_vertex], 0, sizeof(unsigned) * (num_candidates_array_[core_vertex] + 1), streams_[core_vertex]);
        count_invalid_candidate_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(ptrs_d_candidate_bitmap_[core_vertex], ptrs_d_invalid_candidate_array_[core_vertex], 
            d_flag[core_vertex], num_candidates_array_[core_vertex]);
        //
        num_invalid_candidates_array_[core_vertex] = 0;
        exclusive_sum(d_flag[core_vertex], d_flag[core_vertex], num_candidates_array_[core_vertex] + 1, 
            num_invalid_candidates_array_[core_vertex], streams_[core_vertex]);
        cout << "group " << core_vertex << " (removed size): " << num_invalid_candidates_array_[core_vertex] << endl;
        //
        if (num_invalid_candidates_array_[core_vertex] == 0) {
            continue;
        }
        compact_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(d_flag[core_vertex], ptrs_d_invalid_candidate_array_[core_vertex], 
            ptrs_d_invalid_candidate_array_next_iteration_[core_vertex], num_candidates_array_[core_vertex]);
        //
        cudaMemcpyAsync(ptrs_d_invalid_candidate_array_[core_vertex], ptrs_d_invalid_candidate_array_next_iteration_[core_vertex], 
            sizeof(Vertex)*num_invalid_candidates_array_[core_vertex], cudaMemcpyDeviceToDevice, streams_[core_vertex]);
        GRID_SIZE = (num_invalid_candidates_array_[core_vertex]+BLOCK_SIZE-1)/BLOCK_SIZE;
        clear_counter_kernel2<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(ptrs_d_invalid_candidate_array_[core_vertex], num_invalid_candidates_array_[core_vertex], 
            ptrs_d_query_neighbors_count_[core_vertex], ptrs_d_re_core_neighbors[core_vertex], num_re_core_neighbors[core_vertex]);
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    // decide perform mode of next iteration
    double pull_cost = 0;
    double push_cost = 0;
    for (auto core_vertex : core_vertices) {
        pull_cost += num_candidates_array_next_iteration[core_vertex] * num_re_core_neighbors[core_vertex];
        push_cost += num_invalid_candidates_array_[core_vertex] * num_re_core_neighbors[core_vertex];
    }
    if (pull_cost <= push_cost) {
        is_pull = true;
    } else {
        is_pull = false;
    }
    is_pull = true;
    // change 2 to 0
    GRID_SIZE = (data_size_+BLOCK_SIZE-1) / BLOCK_SIZE;
    for (auto core_vertex : core_vertices) {
        change_flag_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[core_vertex]>>>(ptrs_d_candidate_bitmap_[core_vertex], data_size_);
        num_candidates_array_[core_vertex] = num_candidates_array_next_iteration[core_vertex];
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
    //
    for (auto core_vertex : core_vertices) {
        cudaMemcpyAsync(ptrs_d_candidate_bitmap_next_iterartion_[core_vertex], ptrs_d_candidate_bitmap_[core_vertex], 
            sizeof(unsigned)*(data_size_+1), cudaMemcpyDeviceToDevice, streams_[core_vertex]);
    }
    //
    times++;
    if (times >= 3) {
        is_stop = true;
    }
}

void MMatch::update_auxiliary_structure(vector<Vertex> core_vertices, Vertex* num_re_core_neighbors, Vertex* num_reversed_re_core_neighbors, 
                    Vertex** ptrs_d_re_core_neighbors, Vertex** ptrs_d_reversed_re_core_neighbors, bool is_pull) {
    int BLOCK_SIZE;
    int GRID_SIZE;

    for (auto core_vertex : core_vertices) {
        BLOCK_SIZE = 256;
        unsigned size_shared_memory;
        if (num_re_core_neighbors[core_vertex] != 0 && is_pull) {
            GRID_SIZE = (num_candidates_array_[core_vertex]*32+BLOCK_SIZE-1) / BLOCK_SIZE;
            size_shared_memory = sizeof(Vertex)*BLOCK_SIZE + sizeof(unsigned)*BLOCK_SIZE/32*num_re_core_neighbors[core_vertex];
            count_neighbors_kernel<<<GRID_SIZE, BLOCK_SIZE, size_shared_memory, streams_[core_vertex]>>>(d_ptrs_d_candidate_bitmap_, d_data_offset_, 
                d_data_edges_, ptrs_d_candidate_array_[core_vertex], num_candidates_array_[core_vertex], ptrs_d_re_core_neighbors[core_vertex], 
                num_re_core_neighbors[core_vertex], ptrs_d_query_neighbors_count_[core_vertex], core_vertex);
        } else if (num_reversed_re_core_neighbors[core_vertex] != 0 && !is_pull) {
            GRID_SIZE = (num_invalid_candidates_array_[core_vertex]*32+BLOCK_SIZE-1) / BLOCK_SIZE;
            if (GRID_SIZE == 0) {
                continue;
            }
            size_shared_memory = sizeof(Vertex)*BLOCK_SIZE;
            alleviate_neighbors_kernel<<<GRID_SIZE, BLOCK_SIZE, size_shared_memory, streams_[core_vertex]>>>(d_ptrs_d_candidate_bitmap_, d_data_offset_, 
                d_data_edges_, ptrs_d_invalid_candidate_array_[core_vertex], num_invalid_candidates_array_[core_vertex], ptrs_d_reversed_re_core_neighbors[core_vertex], 
                num_reversed_re_core_neighbors[core_vertex], d_ptrs_d_query_neighbors_count_, core_vertex);
        }
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());
}

void MMatch::kernel_filter_core() {
    unsigned** d_flag = new unsigned*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        d_flag[i] = NULL;
    }

    // collect vertices in core
    vector<Vertex> core_vertices;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 2) {
            core_vertices.push_back(i);
        }
    }
    // allocate exclusive prefix sum arrays
    for (auto core_vertex : core_vertices) {
        if (device_memory_manager_->allocate_memory(d_flag[core_vertex], sizeof(unsigned) * (num_candidates_array_[core_vertex] + 1), 
                "exclusive prefix sum array for " + to_string(core_vertex))) {
            cudaMemset(d_flag[core_vertex], 0, sizeof(unsigned) * (num_candidates_array_[core_vertex] + 1));
        } else {
            exit(0);
        }
    }
    // collect neighbors in core for each query vertex in core
    Vertex* num_re_core_neighbors = NULL;
    Vertex* num_op_core_neighbors = NULL;
    Vertex* num_reversed_re_core_neighbors = NULL;
    Vertex** ptrs_d_re_core_neighbors = NULL;
    Vertex** ptrs_d_op_core_neighbors = NULL;
    Vertex** ptrs_d_reversed_re_core_neighbors = NULL;
    collect_core_neighbors(core_vertices, num_re_core_neighbors, num_op_core_neighbors, num_reversed_re_core_neighbors, 
        ptrs_d_re_core_neighbors, ptrs_d_op_core_neighbors, ptrs_d_reversed_re_core_neighbors);
    // perform mode
    bool is_pull = true;
    bool is_stop = false;
    while (!is_stop) {
        cout << "pull mode: " << is_pull << endl;
        perform_iteration(core_vertices, num_re_core_neighbors, num_op_core_neighbors, num_reversed_re_core_neighbors, 
            ptrs_d_re_core_neighbors, ptrs_d_op_core_neighbors, ptrs_d_reversed_re_core_neighbors, is_pull);
        decide_perform_mode(core_vertices, num_re_core_neighbors, ptrs_d_re_core_neighbors, d_flag, is_pull, is_stop);
    }
    update_auxiliary_structure(core_vertices, num_re_core_neighbors, num_reversed_re_core_neighbors, ptrs_d_re_core_neighbors, ptrs_d_reversed_re_core_neighbors, is_pull);

    // release exclusive prefix sum arrays
    for (auto core_vertex : core_vertices) {
        device_memory_manager_->free_memory(d_flag[core_vertex], sizeof(unsigned) * (num_candidates_array_[core_vertex] + 1), 
                "exclusive prefix sum array for " + to_string(core_vertex));
    }
    // release invalid candidate arrays
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 1) {
            device_memory_manager_->free_memory(ptrs_d_candidate_array_next_iteration_[i], 
                "candidate array of equivalent class " + to_string(i) + " in the next iteration");
        } else if (equivalent_class_type_[i] == 2) {
            device_memory_manager_->free_memory(ptrs_d_candidate_array_next_iteration_[i], 
                "candidate array of equivalent class " + to_string(i) + " in the next iteration");
            device_memory_manager_->free_memory(ptrs_d_invalid_candidate_array_[i], 
                "invalid candidate array of equivalent class " + to_string(i));
            device_memory_manager_->free_memory(ptrs_d_invalid_candidate_array_next_iteration_[i], 
                "invalid candidate array of equivalent class " + to_string(i) + " in the next iteration");
        }
    }
    // release bitmap backup
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 2) {
            device_memory_manager_->free_memory(ptrs_d_candidate_bitmap_next_iterartion_[i], sizeof(unsigned)*(data_size_+1), 
                    "bitmap of equivalent class " + to_string(i) + " in next iteration");
        }
    }
}

void MMatch::perform_refine_plan() {
    prepare_initialise();
    kernel_initialise();
    
    prepare_collect();
    kernel_collect();

    int aaa = 0;
    double bbb = 0;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] != 0) {
            aaa++;
            bbb += num_candidates_array_[i];
        }
    }
    bbb = bbb / aaa;
    cout << "LDF: " << bbb << endl;

    prepare_filter();
    kernel_filter_forest();
    kernel_filter_core();

    bbb = 0;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] != 0) {
            bbb += num_candidates_array_[i];
        }
    }
    bbb = bbb / aaa;
    cout << "afterfilter: " << bbb << endl;
}

void MMatch::collect_forest_edges(Vertex class_id, Vertex** vertex_first_class_arrays, Edge** vertex_class_offset, Vertex** vertex_second_class_arrays, Edge* vertex_num_edges) {
    int WARP_SIZE;
    int BLOCK_SIZE;
    int GRID_SIZE;

    unsigned* candidate_pos_in_array = NULL;
    if (device_memory_manager_->allocate_memory(candidate_pos_in_array, sizeof(unsigned)*(data_size_+1), 
        "index in array for " + to_string(class_id))) {
        cudaMemset(candidate_pos_in_array, 0, sizeof(unsigned)*(data_size_+1));
    }
    Vertex sum = 0;
    exclusive_sum(ptrs_d_candidate_bitmap_[class_id], candidate_pos_in_array, data_size_ + 1, sum, streams_[class_id]);
    if (sum != num_candidates_array_[class_id]) {
        cout << "error 1211 line" << endl;
    }

    auto crt_forest_refine_plan = forest_refine_plan_[class_id];
    for (auto operator_ : crt_forest_refine_plan) {
        Vertex target = operator_.target_compressed_vertex_;
        BLOCK_SIZE = 1024;
        GRID_SIZE = (data_size_+BLOCK_SIZE-1)/BLOCK_SIZE;
        clear_counter_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(ptrs_d_candidate_bitmap_[class_id], 
            ptrs_query_neighbors_d_count_[class_id][target], data_size_);

        unsigned* crt_neighbor_count = ptrs_query_neighbors_d_count_[class_id][target];

        exclusive_sum(crt_neighbor_count, crt_neighbor_count, data_size_ + 1, vertex_num_edges[target], streams_[class_id]);

        cout << "(" << class_id << ", " << target << "): " << vertex_num_edges[target] << endl;

        Vertex* d_neighbor_list = NULL;
        unsigned* d_compacted_count = NULL;
        if (device_memory_manager_->allocate_memory(d_neighbor_list, sizeof(Vertex)*vertex_num_edges[target], 
            "decompressed candidate edges for (" + to_string(class_id) + ", " + to_string(target) + ")")) {
            cudaMemset(d_neighbor_list, 0, sizeof(Vertex)*vertex_num_edges[target]);
        }
        if (device_memory_manager_->allocate_memory(d_compacted_count, sizeof(unsigned)*(num_candidates_array_[class_id]+1), 
            "compressed offset for (" + to_string(class_id) + ", " + to_string(target) + ")")) {
            cudaMemset(d_compacted_count, 0, sizeof(unsigned)*(num_candidates_array_[class_id]+1));
        }

        compact_kernel2<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(candidate_pos_in_array, ptrs_d_candidate_bitmap_[class_id], 
            crt_neighbor_count, d_compacted_count, data_size_);

        //two-step output scheme: re-examine the candidate edges and write them to the corresponding address of the hash table
        WARP_SIZE = 32;
        BLOCK_SIZE = 256;
        GRID_SIZE = (num_candidates_array_[class_id]*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
        re_examine_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(d_data_offset_, d_data_edges_, d_ptrs_d_candidate_bitmap_, 
            data_size_, num_candidates_array_[class_id], target, ptrs_d_candidate_array_[class_id], 
            d_compacted_count, d_neighbor_list);
        //
        vertex_first_class_arrays[target] = ptrs_d_candidate_array_[class_id];
        vertex_class_offset[target] = d_compacted_count;
        vertex_second_class_arrays[target] = d_neighbor_list;
    }
}

void MMatch::collect_covered_neighbors(Vertex class_id, Vertex* &d_covered_neighbors, Vertex &num_covered_neighbors) {
    vector<Vertex> covered_neighbors;
    for (auto operator_ : original_core_refine_plan_[class_id]) {
        if (!find_operator_set(operator_.target_compressed_vertex_, core_refine_plan_[class_id]).first) {
            covered_neighbors.push_back(operator_.target_compressed_vertex_);
        }
    }
    num_covered_neighbors = covered_neighbors.size();
    if (num_covered_neighbors == 0) {
        return;
    }
    for (auto covered_neighbor : covered_neighbors) {
        if (device_memory_manager_->allocate_memory(ptrs_query_neighbors_d_count_[class_id][covered_neighbor], 
                sizeof(unsigned)*(data_size_+1), "data neighbor count for (" + to_string(class_id) + ", " + to_string(covered_neighbor) + ")")) {
            cudaMemset(ptrs_query_neighbors_d_count_[class_id][covered_neighbor], 0, sizeof(unsigned)*(data_size_+1));
        } else {
            exit(0);
        }
    }
    cudaMemcpy(ptrs_d_query_neighbors_count_[class_id], ptrs_query_neighbors_d_count_[class_id], sizeof(unsigned*)*num_equivalent_classes_, cudaMemcpyHostToDevice);
    //
    Vertex* h_covered_neighbors = new Vertex[num_covered_neighbors];
    copy(covered_neighbors.begin(), covered_neighbors.end(), h_covered_neighbors);
    device_memory_manager_->allocate_memory(d_covered_neighbors, sizeof(Vertex)*num_covered_neighbors, "covered neighbors of " + to_string(class_id));
    cudaMemcpy(d_covered_neighbors, h_covered_neighbors, sizeof(Vertex)*num_covered_neighbors, cudaMemcpyHostToDevice);
}

void MMatch::collect_core_edges(Vertex class_id, Vertex** vertex_first_class_arrays, Edge** vertex_class_offset, Vertex** vertex_second_class_arrays, Edge* vertex_num_edges) {
    int WARP_SIZE;
    int BLOCK_SIZE;
    int GRID_SIZE;

    unsigned* candidate_pos_in_array = NULL;
    if (device_memory_manager_->allocate_memory(candidate_pos_in_array, sizeof(unsigned)*(data_size_+1), 
        "index in array for " + to_string(class_id))) {
        cudaMemset(candidate_pos_in_array, 0, sizeof(unsigned)*(data_size_+1));
    }
    Vertex sum = 0;
    exclusive_sum(ptrs_d_candidate_bitmap_[class_id], candidate_pos_in_array, data_size_ + 1, sum, streams_[class_id]);
    if (sum != num_candidates_array_[class_id]) {
        cout << "error 1296 line" << endl;
    }

    auto crt_forest_refine_plan = forest_refine_plan_[class_id];
    if (!crt_forest_refine_plan.empty()) {
        for (auto operator_ : crt_forest_refine_plan) {
            Vertex target = operator_.target_compressed_vertex_;
            //
            BLOCK_SIZE = 1024;
            GRID_SIZE = (data_size_+BLOCK_SIZE-1)/BLOCK_SIZE;
            clear_counter_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(ptrs_d_candidate_bitmap_[class_id], 
                ptrs_query_neighbors_d_count_[class_id][target], data_size_);

            unsigned* crt_neighbor_count = ptrs_query_neighbors_d_count_[class_id][target];
            exclusive_sum(crt_neighbor_count, crt_neighbor_count, data_size_ + 1, vertex_num_edges[target], streams_[class_id]);

            cout << "(" << class_id << ", " << target << "): " << vertex_num_edges[target] << endl;

            Vertex* d_neighbor_list = NULL;
            unsigned* d_compacted_count = NULL;
            if (device_memory_manager_->allocate_memory(d_neighbor_list, sizeof(Vertex)*vertex_num_edges[target], 
                "decompressed candidate edges for (" + to_string(class_id) + ", " + to_string(target) + ")")) {
                cudaMemset(d_neighbor_list, 0, sizeof(Vertex)*vertex_num_edges[target]);
            }
            if (device_memory_manager_->allocate_memory(d_compacted_count, sizeof(unsigned)*(num_candidates_array_[class_id]+1), 
                "compressed offset for (" + to_string(class_id) + ", " + to_string(target) + ")")) {
                cudaMemset(d_compacted_count, 0, sizeof(unsigned)*(num_candidates_array_[class_id]+1));
            }

            compact_kernel2<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(candidate_pos_in_array, ptrs_d_candidate_bitmap_[class_id], 
            crt_neighbor_count, d_compacted_count, data_size_);

            //two-step output scheme: re-examine the candidate edges and write them to the corresponding address of the hash table
            WARP_SIZE = 32;
            BLOCK_SIZE = 256;
            GRID_SIZE = (num_candidates_array_[class_id]*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
            re_examine_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(d_data_offset_, d_data_edges_, d_ptrs_d_candidate_bitmap_, 
                data_size_, num_candidates_array_[class_id], target, ptrs_d_candidate_array_[class_id], 
                d_compacted_count, d_neighbor_list);
            //
            vertex_first_class_arrays[target] = ptrs_d_candidate_array_[class_id];
            vertex_class_offset[target] = d_compacted_count;
            vertex_second_class_arrays[target] = d_neighbor_list;
        }
    }
    //
    if (!op_core_neighbors_[class_id].empty()) {
        Vertex num_covered_neighbors = 0;
        Vertex* d_covered_neighbors = NULL;
        collect_covered_neighbors(class_id, d_covered_neighbors, num_covered_neighbors);
        if (num_covered_neighbors != 0) {
            BLOCK_SIZE = 256;
            GRID_SIZE = (num_candidates_array_[class_id]*32+BLOCK_SIZE-1) / BLOCK_SIZE;
            unsigned size_shared_memory = sizeof(Vertex)*BLOCK_SIZE + sizeof(unsigned)*BLOCK_SIZE/32*num_covered_neighbors;
            count_neighbors_kernel<<<GRID_SIZE, BLOCK_SIZE, size_shared_memory, streams_[class_id]>>>(d_ptrs_d_candidate_bitmap_, d_data_offset_, 
                d_data_edges_, ptrs_d_candidate_array_[class_id], num_candidates_array_[class_id], d_covered_neighbors, 
                num_covered_neighbors, ptrs_d_query_neighbors_count_[class_id], class_id);
        }
    }
    //
    auto crt_core_refine_plan = original_core_refine_plan_[class_id];
    for (auto operator_ : crt_core_refine_plan) {
        Vertex target = operator_.target_compressed_vertex_;
        unsigned* crt_neighbor_count = ptrs_query_neighbors_d_count_[class_id][target];
        exclusive_sum(crt_neighbor_count, crt_neighbor_count, data_size_ + 1, vertex_num_edges[target], streams_[class_id]);

        cout << "(" << class_id << ", " << target << "): " << vertex_num_edges[target] << endl;

        Vertex* d_neighbor_list = NULL;
        unsigned* d_compacted_count = NULL;
        if (device_memory_manager_->allocate_memory(d_neighbor_list, sizeof(Vertex)*vertex_num_edges[target], 
            "decompressed candidate edges for (" + to_string(class_id) + ", " + to_string(target) + ")")) {
            cudaMemset(d_neighbor_list, 0, sizeof(Vertex)*vertex_num_edges[target]);
        }
        if (device_memory_manager_->allocate_memory(d_compacted_count, sizeof(unsigned)*(num_candidates_array_[class_id]+1), 
            "compressed offset for (" + to_string(class_id) + ", " + to_string(target) + ")")) {
            cudaMemset(d_compacted_count, 0, sizeof(unsigned)*(num_candidates_array_[class_id]+1));
        }

        BLOCK_SIZE = 1024;
        GRID_SIZE = (data_size_+BLOCK_SIZE-1)/BLOCK_SIZE;
        compact_kernel2<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(candidate_pos_in_array, ptrs_d_candidate_bitmap_[class_id], 
            crt_neighbor_count, d_compacted_count, data_size_);

        //two-step output scheme: re-examine the candidate edges and write them to the corresponding address of the hash table
        WARP_SIZE = 32;
        BLOCK_SIZE = 256;
        GRID_SIZE = (num_candidates_array_[class_id]*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
        re_examine_kernel<<<GRID_SIZE, BLOCK_SIZE, 0, streams_[class_id]>>>(d_data_offset_, d_data_edges_, d_ptrs_d_candidate_bitmap_, 
            data_size_, num_candidates_array_[class_id], target, ptrs_d_candidate_array_[class_id], 
            d_compacted_count, d_neighbor_list);
        //
        vertex_first_class_arrays[target] = ptrs_d_candidate_array_[class_id];
        vertex_class_offset[target] = d_compacted_count;
        vertex_second_class_arrays[target] = d_neighbor_list;
    }
}

void MMatch::collect_candidate_edges() {
    // allocate candidate edges for compressed query graph
    Vertex*** edges_first_class_arrays = new Vertex**[num_equivalent_classes_];
    Edge*** edges_class_offset = new Edge**[num_equivalent_classes_];
    Vertex*** edges_second_class_arrays = new Vertex**[num_equivalent_classes_];
    Edge** num_compressed_edges = new Edge*[num_equivalent_classes_];
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        edges_first_class_arrays[i] = new Vertex*[num_equivalent_classes_];
        edges_class_offset[i] = new Edge*[num_equivalent_classes_];
        edges_second_class_arrays[i] = new Vertex*[num_equivalent_classes_];
        num_compressed_edges[i] = new Edge[num_equivalent_classes_];
        for (Vertex j = 0; j < num_equivalent_classes_; j++) {
            edges_first_class_arrays[i][j] = NULL;
            edges_class_offset[i][j] = NULL;
            edges_second_class_arrays[i][j] = NULL;
            num_compressed_edges[i][j] = 0;
        }
    }
    // collect candidate edges for compressed query graph
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 0) { // a leaf
            continue;
        } else if (equivalent_class_type_[i] == 1) { // a forest class
            collect_forest_edges(i, edges_first_class_arrays[i], edges_class_offset[i], edges_second_class_arrays[i], num_compressed_edges[i]);
        } else { // a core class
            collect_core_edges(i, edges_first_class_arrays[i], edges_class_offset[i], edges_second_class_arrays[i], num_compressed_edges[i]);
        }
    }
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

#ifdef DEBUG
    // check bi-direction edges
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 2) {
            for (auto core_neighbor : original_core_refine_plan_[i]) {
                Vertex target = core_neighbor.target_compressed_vertex_;
                if (num_compressed_edges[i][target] != num_compressed_edges[target][i]) {
                    cout << "asymmetry" << endl;
                    cout << i << " -> " << target << ": " << num_compressed_edges[i][target] << endl;
                    cout << target << " -> " << i << ": " << num_compressed_edges[target][i] << endl;
                }
            }
        }
    }
    /*// check details of one specific bi-direction edges
    Vertex src = 0;
    Vertex dest = 5;
    // src -> dest
    Vertex* h_src = new Vertex[num_candidates_array_[src]];
    Edge* h_offset = new Edge[data_size_+1];
    Vertex* h_dest = new Vertex[num_compressed_edges[src][dest]];
    cudaMemcpy(h_src, edges_first_class_arrays[src][dest], sizeof(Vertex)*num_candidates_array_[src], cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offset, edges_class_offset[src][dest], sizeof(Edge)*(data_size_+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dest, edges_second_class_arrays[src][dest], sizeof(Vertex)*num_compressed_edges[src][dest], cudaMemcpyDeviceToHost);
    vector<pair<Vertex, Vertex>> srctodest;
    for (Vertex i = 0; i < num_candidates_array_[src]; i++) {
        Vertex source = h_src[i];
        Vertex begin = h_offset[source];
        Vertex end = h_offset[source+1];
        for (Vertex j = begin; j < end; j++) {
            Vertex destination = h_dest[j];
            srctodest.push_back(make_pair(source, destination));
        }
    }
    // dest -> src
    Vertex* h_inversed_src = new Vertex[num_candidates_array_[dest]];
    Edge* h_inversed_offset = new Edge[data_size_+1];
    Vertex* h_inversed_dest = new Vertex[num_compressed_edges[dest][src]];
    cudaMemcpy(h_inversed_src, edges_first_class_arrays[dest][src], sizeof(Vertex)*num_candidates_array_[dest], cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inversed_offset, edges_class_offset[dest][src], sizeof(Edge)*(data_size_+1), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inversed_dest, edges_second_class_arrays[dest][src], sizeof(Vertex)*num_compressed_edges[dest][src], cudaMemcpyDeviceToHost);
    vector<pair<Vertex, Vertex>> desttosrc;
    for (Vertex i = 0; i < num_candidates_array_[dest]; i++) {
        Vertex source = h_inversed_src[i];
        Vertex begin = h_inversed_offset[source];
        Vertex end = h_inversed_offset[source+1];
        for (Vertex j = begin; j < end; j++) {
            Vertex destination = h_inversed_dest[j];
            desttosrc.push_back(make_pair(source, destination));
        }
    }
    // bitmap
    unsigned* h_src_bitmap = new unsigned[data_size_+1];
    cudaMemcpy(h_src_bitmap, ptrs_d_candidate_bitmap_[src], sizeof(unsigned)*(data_size_+1), cudaMemcpyDeviceToHost);
    unsigned* h_dest_bitmap = new unsigned[data_size_+1];
    cudaMemcpy(h_dest_bitmap, ptrs_d_candidate_bitmap_[dest], sizeof(unsigned)*(data_size_+1), cudaMemcpyDeviceToHost);
    for (auto ele : srctodest) {
        if (find(desttosrc.begin(), desttosrc.end(), make_pair(ele.second, ele.first)) == desttosrc.end()) {
            cout << ele.first << " " << ele.second << " " << h_src_bitmap[ele.first] << " " << h_dest_bitmap[ele.second] << endl;
        }
    }
    cout << srctodest.size() << endl;
    cout << "--------------------------" << endl;
    for (auto ele : desttosrc) {
        if (find(srctodest.begin(), srctodest.end(), make_pair(ele.second, ele.first)) == srctodest.end()) {
            cout << ele.first << " " << ele.second << " " << h_dest_bitmap[ele.first] << " " << h_src_bitmap[ele.second] << endl;
        }
    }
    cout << desttosrc.size() << endl;*/
#endif
    // map query vertex to equivalent class 
    edges_first_vertex_arrays_ = new Vertex**[query_size_];
    edges_offset_ = new Edge**[query_size_];
    edges_second_vertex_arrays_ = new Vertex**[query_size_];
    num_candidate_edges_ = new Edge*[query_size_];
    for (Vertex i = 0; i < query_size_; i++) {
        edges_first_vertex_arrays_[i] = new Vertex*[query_size_];
        edges_offset_[i] = new Edge*[query_size_];
        edges_second_vertex_arrays_[i] = new Vertex*[query_size_];
        num_candidate_edges_[i] = new Edge[query_size_];
        // initialization
        for (Vertex j = 0; j < query_size_; j++) {
            edges_first_vertex_arrays_[i][j] = NULL;
            edges_offset_[i][j] = NULL;
            edges_second_vertex_arrays_[i][j] = NULL;
            num_candidate_edges_[i][j] = 0;
        }
        if (query_vertex_type_[i] == 0) { // leaf
            continue;
        } else if (query_vertex_type_[i] == 1) { // forest
            auto neighbors = plain_query_graph_->neighbors(i);
            Vertex crt_class = inversed_classes_[i].first;
            for (auto neighbor : neighbors) {
                Vertex neighbor_class = inversed_classes_[neighbor].first;
                if (find_operator_set(neighbor_class, forest_refine_plan_[crt_class]).first) {
                    edges_first_vertex_arrays_[i][neighbor] = edges_first_class_arrays[crt_class][neighbor_class];
                    edges_offset_[i][neighbor] = edges_class_offset[crt_class][neighbor_class];
                    edges_second_vertex_arrays_[i][neighbor] = edges_second_class_arrays[crt_class][neighbor_class];
                    num_candidate_edges_[i][neighbor] = num_compressed_edges[crt_class][neighbor_class];
                } else if (!find_operator_set(crt_class, forest_refine_plan_[neighbor_class]).first) {
                    cout << "query edge " << i << "->" << neighbor << " is ignored" << endl;
                    exit(0);
                }
            }
        } else if (query_vertex_type_[i] == 2) { // core
            auto neighbors = plain_query_graph_->neighbors(i);
            Vertex crt_class = inversed_classes_[i].first;
            for (auto neighbor : neighbors) {
                Vertex neighbor_class = inversed_classes_[neighbor].first;
                edges_first_vertex_arrays_[i][neighbor] = edges_first_class_arrays[crt_class][neighbor_class];
                edges_offset_[i][neighbor] = edges_class_offset[crt_class][neighbor_class];
                edges_second_vertex_arrays_[i][neighbor] = edges_second_class_arrays[crt_class][neighbor_class];
                num_candidate_edges_[i][neighbor] = num_compressed_edges[crt_class][neighbor_class];
            }
        }

    }
}

void MMatch::generate_join_order() {
    join_order_ = new Vertex[query_size_];
    id_to_pos_ = new Vertex[query_size_];
    bool* is_visited = new bool[query_size_];
    memset(is_visited, 0, sizeof(bool)*query_size_);
    //
    float max_num_core_candidates = 0;
    float max_num_forest_candidates = 0;
    for (Vertex i = 0; i < num_equivalent_classes_; i++) {
        if (equivalent_class_type_[i] == 2 && num_candidates_array_[i] > max_num_core_candidates) {
            max_num_core_candidates = num_candidates_array_[i];
        } else if (equivalent_class_type_[i] == 1 && num_candidates_array_[i] > max_num_forest_candidates) {
            max_num_forest_candidates = num_candidates_array_[i];
        }
    }
    // increase weight so that core is first joined; forest is second joined; and leaf is last joined
    vector<ScoredVertex> query_vertices(query_size_);
    for (Vertex i = 0; i < query_size_; i++) {
        Vertex equivalent_class = inversed_classes_[i].first;
        if (equivalent_class_type_[equivalent_class] == 0) {
            query_vertices[i] = ScoredVertex(i, (float)(num_candidates_array_[equivalent_class] + max_num_forest_candidates + max_num_core_candidates));
        } else if (equivalent_class_type_[equivalent_class] == 1) {
            query_vertices[i] = ScoredVertex(i, (float)(num_candidates_array_[equivalent_class] + max_num_core_candidates));
        } else {
            query_vertices[i] = ScoredVertex(i, (float)(num_candidates_array_[equivalent_class]));
        }
    }
    for (Vertex crt_pos = 0; crt_pos < query_size_; crt_pos++) {
        priority_queue<ScoredVertex, vector<ScoredVertex>, greater<ScoredVertex>> pq;
        if (crt_pos == 0) {
            if (has_core_) { // has core structure
                for (auto query_vertex : query_vertices) {
                    pq.push(query_vertex);
                }
            } else { // no core structure
                for (Vertex i = 0; i < query_size_; i++) {
                    if (inversed_classes_[i].first == forest_order_[0][0]) {
                        pq.push(query_vertices[i]);
                    }
                }
            }
        } else {
            for (Vertex j = 0; j < crt_pos; j++) {
                auto query_neighbors = plain_query_graph_->neighbors(join_order_[j]);
                for (auto query_neighbor : query_neighbors) {
                    if (!is_visited[query_neighbor]) {
                        pq.push(query_vertices[query_neighbor]);
                    }
                }
            }
        }
#ifdef DEBUG
        print_priority_queue(pq);
#endif
        Vertex matched_vertex = pq.top().vertex_id_;
        join_order_[crt_pos] = matched_vertex;
        id_to_pos_[matched_vertex] = crt_pos;
        is_visited[matched_vertex] = true;
    }
#ifdef DEBUG
    cout << "match order: ";
    for (Vertex i = 0; i < query_size_; i++) {
        cout << join_order_[i] << " ";
    }
    cout << endl;
#endif
}

void MMatch::collect_neighbors_on_induced_subgraph(Vertex subgraph_size, Vertex** &induced_edges_first_vertex_arrays, Edge** &induced_edges_offset, 
                Vertex** &induced_edges_second_vertex_arrays, Vertex* &induced_candidate_num, Vertex* &inversed_idx, Vertex** &d_induced_edges_first_vertex_arrays, 
                Edge** &d_induced_edges_offset, Vertex** &d_induced_edges_second_vertex_arrays, Vertex* &d_induced_candidate_num, Vertex* &d_inversed_idx, 
                Vertex &num_induced_neighbors) {
    Vertex matched_vertex = join_order_[subgraph_size];
    // induced neighbors
    vector<Vertex> induced_neighbors;
    for (Vertex i = 0; i < subgraph_size; i++) {
        if (plain_query_graph_->check_edge_existence(matched_vertex, join_order_[i])) {
            induced_neighbors.push_back(join_order_[i]);
        }
    }
    // locate data vertex in embedding
    num_induced_neighbors = induced_neighbors.size();
    inversed_idx = new Vertex[num_induced_neighbors];
    for (Vertex i = 0; i < num_induced_neighbors; i++) {
        inversed_idx[i] = id_to_pos_[induced_neighbors[i]];
    }
    // load data edges
    induced_edges_first_vertex_arrays = new Vertex*[num_induced_neighbors];
    induced_edges_offset = new Edge*[num_induced_neighbors];
    induced_edges_second_vertex_arrays = new Vertex*[num_induced_neighbors];
    induced_candidate_num = new Vertex[num_induced_neighbors];
    for (Vertex i = 0; i < num_induced_neighbors; i++) {
        Vertex induced_neighbor = induced_neighbors[i];
        induced_edges_first_vertex_arrays[i] = edges_first_vertex_arrays_[induced_neighbor][matched_vertex];
        induced_edges_offset[i] = edges_offset_[induced_neighbor][matched_vertex];
        induced_edges_second_vertex_arrays[i] = edges_second_vertex_arrays_[induced_neighbor][matched_vertex];
        induced_candidate_num[i] = num_candidates_array_[inversed_classes_[induced_neighbor].first];
    }
    device_memory_manager_->copy_ptrs_to_device(d_induced_edges_first_vertex_arrays, induced_edges_first_vertex_arrays, 
        num_induced_neighbors, "source vertex arrays of induced subgraph " + to_string(subgraph_size));
    device_memory_manager_->copy_ptrs_to_device(d_induced_edges_offset, induced_edges_offset, num_induced_neighbors, 
        "offset arrays of induced subgraph " + to_string(subgraph_size));
    device_memory_manager_->copy_ptrs_to_device(d_induced_edges_second_vertex_arrays, induced_edges_second_vertex_arrays, 
        num_induced_neighbors, "destination vertex arrays of induced subgraph " + to_string(subgraph_size));

    device_memory_manager_->allocate_memory(d_induced_candidate_num, sizeof(Vertex)*num_induced_neighbors, "size array of induced subgraph " + to_string(subgraph_size));
    cudaMemcpy(d_induced_candidate_num, induced_candidate_num, sizeof(Vertex)*num_induced_neighbors, cudaMemcpyHostToDevice);
    device_memory_manager_->allocate_memory(d_inversed_idx, sizeof(Vertex)*num_induced_neighbors, "position array of induced subgraph " + to_string(subgraph_size));
    cudaMemcpy(d_inversed_idx, inversed_idx, sizeof(Vertex)*num_induced_neighbors, cudaMemcpyHostToDevice);
}

string block_identifier(BlockChain block_chain) {
    return "block (" + to_string(block_chain.crt_layer_) + ", " + to_string(block_chain.crt_block_id_) + ")";
}

string block_identifier(Vertex layer, Vertex block_id) {
    return "block (" + to_string(layer) + ", " + to_string(block_id) + ")";
}

string block_identifier(BlockChain block_chain, Vertex separated_id) {
    return "block (" + to_string(block_chain.crt_layer_) + ", " + to_string(block_chain.crt_block_id_) + "-" + to_string(separated_id) + ")";
}

void copy_block_data_dtoh(BlockData &block_data) {
    block_data.h_partial_embeddings_ = new Vertex[block_data.block_data_size_*block_data.len_embedding_];
    block_data.h_count_ = new long long unsigned[block_data.block_data_size_+1];
    cudaMemcpy(block_data.h_partial_embeddings_, block_data.d_partial_embeddings_, sizeof(Vertex)*block_data.block_data_size_*block_data.len_embedding_, cudaMemcpyDeviceToHost);
    cudaMemcpy(block_data.h_count_, block_data.d_count_, sizeof(long long unsigned)*(block_data.block_data_size_+1), cudaMemcpyDeviceToHost);
    block_data.is_on_host_ = true;
}

void copy_block_data_htod(BlockData &block_data, BlockChain block_chain, Vertex separated_id, MemoryManager* device_memory_manager) {
    device_memory_manager->allocate_memory(block_data.d_partial_embeddings_, sizeof(Vertex)*block_data.block_data_size_*block_data.len_embedding_, 
        "splitted embeddings of " + block_identifier(block_chain, separated_id));
    device_memory_manager->allocate_memory(block_data.d_count_, sizeof(long long unsigned)*(block_data.block_data_size_+1), 
        "splitted bound of neighbor list length of " + block_identifier(block_chain, separated_id));
    cudaMemcpy(block_data.d_partial_embeddings_, block_data.h_partial_embeddings_, sizeof(Vertex)*block_data.block_data_size_*block_data.len_embedding_, cudaMemcpyHostToDevice);
    cudaMemcpy(block_data.d_count_, block_data.h_count_, sizeof(long long unsigned)*(block_data.block_data_size_+1), cudaMemcpyHostToDevice);
    block_data.is_on_device_ = true;
}

void delete_device_block_data(BlockData &block_data, BlockChain block_chain, MemoryManager* device_memory_manager) {
    device_memory_manager->free_memory(block_data.d_partial_embeddings_, sizeof(Vertex)*block_data.len_embedding_*block_data.block_data_size_, 
        "embeddings of " + block_identifier(block_chain));
    device_memory_manager->free_memory(block_data.d_count_, sizeof(long long unsigned)*(block_data.block_data_size_+1), 
        "bound of neighbor list length of " + block_identifier(block_chain));

    block_data.is_on_device_ = false;
}

void MMatch::split_last_block_data(BlockData &last_layer_data, BlockChain last_layer_block, Vertex num_subblock, long long unsigned size_subblock) {
    // alias for programming
    Vertex num_result_cols = last_layer_data.len_embedding_;
    long long unsigned num_result_rows = last_layer_data.block_data_size_;
    // mark the boundary of intervals excluding 0 and num_result_rows
    long long unsigned* pivots = new long long unsigned[num_subblock - 1];
    for (Vertex i = 1; i < num_subblock; i++) {
        pivots[i - 1] = i * size_subblock;
    }
    long long unsigned* d_pivots = NULL;
    device_memory_manager_->allocate_memory(d_pivots, sizeof(long long unsigned)*(num_subblock-1), "temporary bound");
    cudaMemcpy(d_pivots, pivots, sizeof(long long unsigned)*(num_subblock-1), cudaMemcpyHostToDevice);

    long long unsigned* d_intervals = NULL; // record position of splitted B_{ij}
    long long unsigned* d_prefix_sum = NULL; // record prefix sum to recompute d_count array of B_{ij}
    device_memory_manager_->allocate_memory(d_intervals, sizeof(long long unsigned)*(num_subblock-1), "temporary interval");
    cudaMemset(d_intervals, 0, sizeof(long long unsigned)*(num_subblock-1));
    device_memory_manager_->allocate_memory(d_prefix_sum, sizeof(long long unsigned)*(num_subblock-1), "temporary reduced prefix sum");
    cudaMemset(d_prefix_sum, 0, sizeof(long long unsigned)*(num_subblock-1));
    int BLOCK_SIZE = 1024;
    long long unsigned GRID_SIZE = (num_result_rows+BLOCK_SIZE-1)/BLOCK_SIZE;
    split_last_embeddings_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(last_layer_data.d_count_, d_pivots, num_subblock-1, d_intervals, d_prefix_sum, num_result_rows);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    long long unsigned* intervals = new long long unsigned[num_subblock - 1];
    cudaMemcpy(intervals, d_intervals, sizeof(long long unsigned)*(num_subblock - 1), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    long long unsigned* num_subresult_rows = new long long unsigned[num_subblock];
#endif

    // after splitting, the length of d_count becomes size_subblock + 1, we need to re-allocate and rewrite it
    for (Vertex i = 0; i < num_subblock; i++) {
        long long unsigned block_data_size = 0;
        if (i == 0) {
            block_data_size = intervals[0];
            last_layer_data.split_data_.push_back(
                new BlockData(last_layer_data.d_partial_embeddings_, last_layer_data.d_count_, block_data_size, num_result_cols));
#ifdef DEBUG
            cudaMemcpy(num_subresult_rows + i, last_layer_data.d_count_ + block_data_size, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
#endif
        } else {
            if (i == num_subblock - 1) {
                block_data_size = num_result_rows - intervals[i - 1];
            } else {
                block_data_size = intervals[i] - intervals[i - 1];
            }

            long long unsigned* d_splitted_count = NULL;
            device_memory_manager_->allocate_memory(d_splitted_count, sizeof(long long unsigned)*(block_data_size+1), 
                "splitted bound of neighbor list length of " + block_identifier(last_layer_block, i));
            cudaMemcpy(d_splitted_count, last_layer_data.d_count_ + intervals[i - 1], sizeof(long long unsigned)*(block_data_size+1), cudaMemcpyDeviceToDevice);

            GRID_SIZE = (block_data_size+BLOCK_SIZE)/BLOCK_SIZE;
            rewrite_count_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_splitted_count, block_data_size + 1, d_prefix_sum, i - 1);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
#ifdef DEBUG
            cudaMemcpy(num_subresult_rows + i, d_splitted_count + block_data_size, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
#endif
            //
            last_layer_data.split_data_.push_back(
                new BlockData(last_layer_data.d_partial_embeddings_ + intervals[i - 1] * num_result_cols, d_splitted_count, block_data_size, num_result_cols));
        }
    }
#ifdef DEBUG
    // check intervals
    long long unsigned sum = 0;
    long long unsigned sum_groundtruth = 0;
    cudaMemcpy(&sum_groundtruth, last_layer_data.d_count_ + num_result_rows, sizeof(long long unsigned), cudaMemcpyDeviceToHost);
    for (Vertex i = 0; i < num_subblock; i++) {
        sum += num_subresult_rows[i];
    }
    if (sum != sum_groundtruth) {
        cout << "the segmentation is wrong" << endl;
        cout << sum_groundtruth << " " << sum << endl;
        for (Vertex i = 0; i < num_subblock; i++) {
            cout << num_subresult_rows[i] << endl;
        }
        exit(0);
    }
    long long unsigned* prefix_sum = new long long unsigned[num_subblock-1];
    cudaMemcpy(prefix_sum, d_prefix_sum, sizeof(long long unsigned)*(num_subblock - 1), cudaMemcpyDeviceToHost);
    sum = 0;
    for (Vertex i = 0; i < num_subblock - 1; i++) {
        sum += num_subresult_rows[i];
        if (sum != prefix_sum[i]) {
            cout << "the segmentation is wrong" << endl;
            for (Vertex i = 0; i < num_subblock; i++) {
                cout << prefix_sum[i] << " " << num_subresult_rows[i] << endl;
            }
            exit(0);
        }
    }
    delete[] num_subresult_rows;
    delete[] prefix_sum;
#endif
    // clear temporary arrays
    device_memory_manager_->free_memory(d_pivots, sizeof(long long unsigned)*(num_subblock-1), "temporary bound");
    device_memory_manager_->free_memory(d_intervals, sizeof(long long unsigned)*(num_subblock-1), "temporary interval");
    device_memory_manager_->free_memory(d_prefix_sum, sizeof(long long unsigned)*(num_subblock-1), "temporary reduced prefix sum");
    delete[] pivots;
    delete[] intervals;
}

// when this function is performed, last_layer_data is destroyed
// start at layer 0
// receive a block B_{ij} at layer i, split B_{ij} to form blocks at layer i+1
void MMatch::extend_embeddings_recursively(BlockChain &last_layer_block, BlockData &last_layer_data) {
    // alias for programming
    Vertex num_result_cols = last_layer_data.len_embedding_;
    long long unsigned num_result_rows = last_layer_data.block_data_size_;

    if (num_result_cols == query_size_) { // the leaf, we collect result and destroy device arrays
        num_all_embeddings_ += num_result_rows;
        Vertex* partial_result = new Vertex[num_result_cols*5]; // avoid to store result because cpu main memory is not sufficient
        cudaMemcpy(partial_result, last_layer_data.d_partial_embeddings_, sizeof(Vertex)*num_result_cols*5, cudaMemcpyDeviceToHost);
        separated_embeddings_.push_back(partial_result);
        num_separated_embeddings_.push_back(num_result_rows);
        delete_device_block_data(last_layer_data, last_layer_block, device_memory_manager_);
        return;
    }

    int WARP_SIZE = 32;
    int BLOCK_SIZE = 256;
    long long unsigned GRID_SIZE = (num_result_rows*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
    precompute_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(last_layer_data.d_partial_embeddings_, induced_edges_first_vertex_arrays_[num_result_cols][0], induced_edges_offset_[num_result_cols][0], 
        induced_edges_second_vertex_arrays_[num_result_cols][0], last_layer_data.d_count_, num_result_rows, num_result_cols, inversed_idx_[num_result_cols][0], 
        induced_candidate_num_[num_result_cols][0]);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    long long unsigned num_new_result_rows = 0; // the number of embeddings originated from block B_{ij} at layer i+1
    exclusive_sum(last_layer_data.d_count_, last_layer_data.d_count_, num_result_rows + 1, num_new_result_rows);
    cudaDeviceSynchronize();
    checkCudaErrors(cudaGetLastError());

    cout << "the bound of " << block_identifier(last_layer_block) << ": " << num_new_result_rows << endl;
    if (num_new_result_rows == 0) { // this block cannot generate blocks at layer i+1
        delete_device_block_data(last_layer_data, last_layer_block, device_memory_manager_);
        return;
    }
    // evaluate the number of blocks at layer i+1
    Vertex num_subblock = 0;
    long long unsigned size_subblock = 0;
    device_memory_manager_->form_blocks(num_new_result_rows, num_result_cols + 1, num_subblock, size_subblock);
    // blocks at layer i+1
    Vertex left_bound = blocks_tree_[num_result_cols].size();
    last_layer_block.next_layer_block_left_bound_ = left_bound;
    last_layer_block.next_layer_block_right_bound_ = left_bound + num_subblock - 1;
    cout << block_identifier(last_layer_block) << " -> [" << last_layer_block.next_layer_block_left_bound_ << ", " << last_layer_block.next_layer_block_right_bound_ << "]" << endl;

    // embeddings on the last layer explode on this layer, hence, we should perform extend operation in batch
    // split block B_{ij}
    if (num_subblock != 1) {
        split_last_block_data(last_layer_data, last_layer_block, num_subblock, size_subblock);
    } else {
        last_layer_data.split_data_.push_back(&last_layer_data);
    }
    // extend operation is performed in the DFS way, we need to store all blocks except B_{ij}-0 because B_{ij}-0 is performed immediately
    // so that we can backtrack B_{ij}-k (k = 1, 2, ..., num_subblock - 1)
    for (Vertex i = 1; i < num_subblock; i++) {
        copy_block_data_dtoh(*(last_layer_data.split_data_[i]));
        // delete newly formed d_count arrays, they have been stored in host memory
        device_memory_manager_->free_memory(last_layer_data.split_data_[i]->d_count_, sizeof(unsigned)*(last_layer_data.split_data_[i]->block_data_size_+1), 
            "splitted bound of neighbor list length of " + block_identifier(last_layer_block, i));
    }
    // perform extend operation
    for (Vertex i = 0; i < num_subblock; i++) {
        // alias for programming
        BlockData crt_splitted_block_data = *last_layer_data.split_data_[i];
        long long unsigned crt_num_result_rows = crt_splitted_block_data.block_data_size_;
        Vertex crt_num_result_cols = num_result_cols;
        long long unsigned crt_num_new_result_rows = 0; // the bound of the number of embeddings of block at layer i+1
        cudaMemcpy(&crt_num_new_result_rows, crt_splitted_block_data.d_count_+crt_splitted_block_data.block_data_size_, sizeof(long long unsigned), cudaMemcpyDeviceToHost);

        // allocate intermediate arrays
        long long unsigned* d_new_count = NULL;
        Vertex* d_pre_allocated_neighbor_list = NULL;
        device_memory_manager_->allocate_memory(d_pre_allocated_neighbor_list, sizeof(Vertex)*crt_num_new_result_rows, "temporary pre-allocate neighbor list");
        cudaMemset(d_pre_allocated_neighbor_list, 0, sizeof(Vertex)*crt_num_new_result_rows);
        device_memory_manager_->allocate_memory(d_new_count, sizeof(long long unsigned)*(crt_num_result_rows+1), "temporary exact neighbor list length");
        cudaMemset(d_new_count, 0, sizeof(long long unsigned)*(crt_num_result_rows+1));
        // compute exact neighbor list
        unsigned size_shared_memory = sizeof(Vertex)*(3*BLOCK_SIZE+BLOCK_SIZE/WARP_SIZE);
        GRID_SIZE = (crt_num_result_rows*WARP_SIZE+BLOCK_SIZE-1)/BLOCK_SIZE;
        cout << " " << crt_num_result_rows << " " << crt_num_new_result_rows << endl;

        compute_kernel<<<GRID_SIZE, BLOCK_SIZE, size_shared_memory>>>(crt_splitted_block_data.d_partial_embeddings_, d_pre_allocated_neighbor_list, d_induced_edges_first_vertex_arrays_[num_result_cols], 
            d_induced_edges_offset_[num_result_cols], d_induced_edges_second_vertex_arrays_[num_result_cols], d_induced_candidate_num_[num_result_cols], d_inversed_idx_[num_result_cols],
            crt_splitted_block_data.d_count_, d_new_count, crt_num_result_rows, crt_num_result_cols, num_induced_neighbors_[num_result_cols]);
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
        //
#if DEBUG
        /*if (num_result_cols == 1) {
            long long unsigned* h_new_count = new long long unsigned[crt_num_result_rows+1];
            cudaMemcpy(h_new_count, d_new_count, sizeof(long long unsigned)*(crt_num_result_rows+1), cudaMemcpyDeviceToHost);
            Edge* h_edges_offset = new Edge[crt_num_result_rows+1];
            cudaMemcpy(h_edges_offset, induced_edges_offset_[1][0], sizeof(Edge)*(crt_num_result_rows+1), cudaMemcpyDeviceToHost);
            for (int ll = 0; ll < crt_num_result_rows; ll++) {
                cout << h_new_count[ll] << " ";
            }
            cout << endl;
            exit(0);
        }*/
#endif
        exclusive_sum(d_new_count, d_new_count, crt_num_result_rows + 1, crt_num_new_result_rows);
        cout << crt_num_new_result_rows << endl;

        if (crt_num_new_result_rows != 0) { // this splitted block can generate subblocks on the next layer
            // perform extend operation to form embeddings at layer i+1
            Vertex* d_new_partial_embeddings = NULL;
            device_memory_manager_->allocate_memory(d_new_partial_embeddings, sizeof(Vertex)*crt_num_new_result_rows*(crt_num_result_cols+1), 
                    "embeddings of " + block_identifier(crt_num_result_cols, left_bound + i));
            cudaMemset(d_new_partial_embeddings, 0, sizeof(Vertex)*crt_num_new_result_rows*(crt_num_result_cols+1));

            GRID_SIZE = (crt_num_result_rows*32+BLOCK_SIZE-1)/BLOCK_SIZE;
            extend_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(crt_splitted_block_data.d_partial_embeddings_, d_pre_allocated_neighbor_list, d_new_partial_embeddings, 
                crt_splitted_block_data.d_count_, d_new_count, crt_num_result_rows, crt_num_result_cols);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
            // prepare required data
            long long unsigned* d_count = NULL;
            device_memory_manager_->allocate_memory(d_count, sizeof(long long unsigned)*(crt_num_new_result_rows+1), 
                "bound of neighbor list length of " + block_identifier(crt_num_result_cols, left_bound + i));
            cudaMemset(d_count, 0, sizeof(long long unsigned)*(crt_num_new_result_rows+1));
            // extend operation has finished, lodge blocks at layer i+1
            blocks_tree_[crt_num_result_cols].push_back(BlockChain(crt_num_result_cols, left_bound + i, INVALID, INVALID));
            datas_tree_[crt_num_result_cols].push_back(BlockData(d_new_partial_embeddings, d_count, crt_num_new_result_rows, crt_num_result_cols + 1));
        }

        // clear invalid arrays
        device_memory_manager_->free_memory(d_new_count, sizeof(long long unsigned)*(crt_num_result_rows+1), "temporary exact neighbor list length");
        // here, crt_num_new_result_rows changes, but we compute memory size based on log, i.e., sizeof(Vertex)*crt_num_new_result_rows is not used
        device_memory_manager_->free_memory(d_pre_allocated_neighbor_list, sizeof(Vertex)*crt_num_new_result_rows, "temporary pre-allocate neighbor list");
        if (i == 0) { // delete thw whole block because it has been stored separately
            delete_device_block_data(last_layer_data, last_layer_block, device_memory_manager_);
        } else { // delete the separated traversed block of B_{ij}
            device_memory_manager_->free_memory(crt_splitted_block_data.d_partial_embeddings_, sizeof(Vertex)*crt_num_result_rows*crt_num_result_cols, 
                "splitted embeddings of " + block_identifier(last_layer_block, i));
            device_memory_manager_->free_memory(crt_splitted_block_data.d_count_, sizeof(unsigned)*(crt_num_result_rows + 1), 
                "splitted bound of neighbor list length of " + block_identifier(last_layer_block, i));
            delete[] crt_splitted_block_data.h_partial_embeddings_;
            delete[] crt_splitted_block_data.h_count_;
        }

        // recursively call this function, block at layer i+1 is regarded as input block
        if (crt_num_new_result_rows != 0) {
            cout << block_identifier(blocks_tree_[crt_num_result_cols][left_bound + i]) << ": " << crt_num_new_result_rows << endl;
            extend_embeddings_recursively(blocks_tree_[crt_num_result_cols][left_bound + i], datas_tree_[crt_num_result_cols][left_bound + i]);
        }
        // load next splitted block
        if (i != num_subblock - 1) {
            copy_block_data_htod(*last_layer_data.split_data_[i + 1], last_layer_block, i + 1, device_memory_manager_);
            cudaDeviceSynchronize();
            checkCudaErrors(cudaGetLastError());
        }
    }
}

void MMatch::join_candidate_edges() {
    // allocate cuda streams
    cudaStreamCreate(&h_to_d_);
    cudaStreamCreate(&perform_kernel_);
    cudaStreamCreate(&d_to_h_);
    cudaStreamCreate(&d_to_d_);
    // collect each induced subgraph of query graph
    num_induced_neighbors_ = new Vertex[query_size_];
    induced_edges_first_vertex_arrays_ = new Vertex**[query_size_];
    induced_edges_offset_ = new Edge**[query_size_];
    induced_edges_second_vertex_arrays_ = new Vertex**[query_size_];
    induced_candidate_num_ = new Vertex*[query_size_];
    inversed_idx_ = new Vertex*[query_size_];
    d_induced_edges_first_vertex_arrays_ = new Vertex**[query_size_];
    d_induced_edges_offset_ = new Edge**[query_size_];
    d_induced_edges_second_vertex_arrays_ = new Vertex**[query_size_];
    d_induced_candidate_num_ = new Vertex*[query_size_];
    d_inversed_idx_ = new Vertex*[query_size_];
    for (Vertex i = 1; i < query_size_; i++) {
        collect_neighbors_on_induced_subgraph(i, induced_edges_first_vertex_arrays_[i], induced_edges_offset_[i], induced_edges_second_vertex_arrays_[i], 
            induced_candidate_num_[i], inversed_idx_[i], d_induced_edges_first_vertex_arrays_[i], d_induced_edges_offset_[i], d_induced_edges_second_vertex_arrays_[i], 
            d_induced_candidate_num_[i], d_inversed_idx_[i], num_induced_neighbors_[i]);
    }
    // initialise BFS tree
    device_memory_manager_->record_memory_pool();
    blocks_tree_.resize(query_size_);
    datas_tree_.resize(query_size_);
    // there is only one node in the first layer
    // check if this node needs to be splitted
    long long unsigned num_result_rows = num_candidates_array_[inversed_classes_[join_order_[0]].first];
    Vertex num_subblock = 0;
    long long unsigned size_subblock = 0;
    device_memory_manager_->form_blocks(num_result_rows, 1, num_subblock, size_subblock);
    // for each subblock, perform DFS-extend operation
    for (Vertex i = 0; i < num_subblock; i++) {
        // prepare required data
        long long unsigned block_data_size = 0;
        if (i == num_subblock - 1) {
            block_data_size = num_result_rows - i * size_subblock;
        } else {
            block_data_size = size_subblock;
        }
        // device arrays
        Vertex* d_partial_embeddings = NULL;
        device_memory_manager_->allocate_memory(d_partial_embeddings, sizeof(Vertex)*block_data_size, "embeddings of " + block_identifier(0, i));
        cudaMemcpy(d_partial_embeddings, ptrs_d_candidate_array_[inversed_classes_[join_order_[0]].first] + i * size_subblock, sizeof(Vertex) * block_data_size, 
            cudaMemcpyDeviceToDevice);
        //
        long long unsigned* d_count = NULL;
        device_memory_manager_->allocate_memory(d_count, sizeof(long long unsigned)*(block_data_size + 1), "bound of neighbor list length of " + block_identifier(0, i));
        cudaMemset(d_count, 0, sizeof(unsigned)*(block_data_size + 1));
        // lodge this subblock
        blocks_tree_[0].push_back(BlockChain(0, i, INVALID, INVALID));
        datas_tree_[0].push_back(BlockData(d_partial_embeddings, d_count, block_data_size, 1));
        // perform
        extend_embeddings_recursively(blocks_tree_[0][i], datas_tree_[0][i]);
        // the tree rooted at this subblock has been traversed
    }
    
#ifdef DEBUG
    cout << "#embedding: " << num_all_embeddings_ << endl;
    Vertex num_dfs_leaves = num_separated_embeddings_.size();
    Vertex num_displayed_embeddings = 5;
    if (num_all_embeddings_ < num_displayed_embeddings) {
        num_displayed_embeddings = num_all_embeddings_;
    }
    if (num_dfs_leaves != 0) {
        if (num_separated_embeddings_[0] < num_displayed_embeddings) {
            num_displayed_embeddings = num_separated_embeddings_[0];
        }
        if (num_separated_embeddings_[num_dfs_leaves-1] < num_displayed_embeddings) {
            num_displayed_embeddings = num_separated_embeddings_[num_dfs_leaves-1];
        }
    }
    for (Vertex i = 0; i < num_displayed_embeddings; i++) {
        for (Vertex j = 0; j < query_size_; j++) {
            cout << separated_embeddings_[0][i*query_size_+j] << " ";
        }
        cout << endl;
    }
    // for (Vertex i = num_separated_embeddings_[num_dfs_leaves-1] - 5; i < num_separated_embeddings_[num_dfs_leaves-1]; i++) {
    for (Vertex i = 0; i < num_displayed_embeddings; i++) {
        for (Vertex j = 0; j < query_size_; j++) {
            cout << separated_embeddings_[num_dfs_leaves-1][i*query_size_+j] << " ";
        }
        cout << endl;
    }
#endif
}

void MMatch::match() {
    init_GPU(dev_id_);

    time_recorder_->event_start("copy graph");
    copy_graph_to_GPU();
    time_recorder_->event_end("copy graph");

    time_recorder_->event_start("total time");
    cout << "-----     decomposing query graph    -----" << endl;
    time_recorder_->event_start("decompose query graph");
    decompose_query_graph();
    time_recorder_->event_end("decompose query graph");

    cout << "-----     generating refine plan    -----" << endl;
    time_recorder_->event_start("generate refine plan");
    generate_refine_plan();
    time_recorder_->event_end("generate refine plan");

    cout << "-----     performing refine plan    -----" << endl;
    time_recorder_->event_start("perform refine plan");
    perform_refine_plan();
    time_recorder_->event_end("perform refine plan");
    
    // gather candidates for edges
    cout << "-----     collecting candidate edges    -----" << endl;
    time_recorder_->event_start("collect candidate edges");
    collect_candidate_edges();
    time_recorder_->event_end("collect candidate edges");

    cout << "-----     generating join order    -----" << endl;
    time_recorder_->event_start("generate join order");
    generate_join_order();
    time_recorder_->event_end("generate join order");

    // join all candidate edges to get result
    cout << "-----     joining candidate edges    -----" << endl;
    time_recorder_->event_start("join candidate edges");
    join_candidate_edges();
    time_recorder_->event_end("join candidate edges");

    time_recorder_->event_end("total time");

    // release();
    time_recorder_->print_all_events();
    // device_memory_manager_->print_memory_cost();
}