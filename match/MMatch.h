#ifndef MMATCH_H
#define MMATCH_H

#include "subgraph_match.h"
#include "embedding.h"
#include <queue>
#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cuda_runtime.h>
#include "../graph/graph.h"
#include "../utility/defines.h"
#include "../utility/inverted_index.h"
#include "../utility/time_recorder.h"
#include "../memory_manager/memory_manager.h"
#include "GSI.h"

using std::cout;
using std::endl;
using std::vector;
using std::queue;
using std::copy;
using std::pair;
using std::find;
using std::fill;
using std::sort;
using std::to_string;
using std::make_pair;

struct BasicOperator {
    Vertex target_compressed_vertex_;
    Vertex count_;
    bool is_optimized_;
};

extern bool basic_operator_sorter(BasicOperator const &u, BasicOperator const &v);

struct BlockChain {
    Vertex crt_layer_;
    Vertex crt_block_id_;
    Vertex next_layer_block_left_bound_;
    Vertex next_layer_block_right_bound_;
    BlockChain(Vertex crt_layer, Vertex crt_block_id, Vertex next_layer_block_left_bound, Vertex next_layer_block_right_bound) {
        crt_layer_ = crt_layer;
        crt_block_id_ = crt_block_id;
        next_layer_block_left_bound_ = next_layer_block_left_bound;
        next_layer_block_right_bound_ = next_layer_block_right_bound;
    }
};

struct BlockData {
    Vertex* h_partial_embeddings_;
    long long unsigned* h_count_;
    Vertex* d_partial_embeddings_;
    long long unsigned* d_count_; // here, prefix exclusive sum of d_count may beyond maximal value of unsigned
    vector<BlockData*> split_data_; // for computation of embeddings of next layer
    bool is_on_device_ = true;
    bool is_on_host_ = false;
    long long unsigned block_data_size_ = 0;
    Vertex len_embedding_ = 0;
    BlockData(Vertex* d_partial_embeddings, long long unsigned* d_count, long long unsigned block_data_size, Vertex len_embedding) {
        d_partial_embeddings_ = d_partial_embeddings;
        d_count_ = d_count;
        block_data_size_ = block_data_size;
        len_embedding_ = len_embedding;
    }
};

/*!
* An logic class that implements MMatch
* input: data graph and query graph
* output: embeddings
*/
class MMatch : public GPUSubgraphMatcher {
public:
    // inspect time cost and space cost
    TimeRecorder* time_recorder_;
    MemoryManager* device_memory_manager_;
    int dev_id_ = 0;
    // basic graph structure
    Vertex query_size_;
    Vertex data_size_;
    // InvertedIndex<Label, Vertex>* inverted_data_label_;
    UndirectedPlainGraph* plain_query_graph_;
    UndirectedCSRGraph* csr_data_graph_;
    // graph on device
    Label* d_data_labels_;
    Edge* d_data_offset_;
    Vertex* d_data_edges_;
    // decomposed query graph
    Vertex* query_vertex_type_; // 2-core: 2; forest: 1; leaf 0;
    //
    vector<vector<Vertex>> equivalent_classes_;
    vector<pair<Vertex, Vertex>> inversed_classes_;
    Vertex num_equivalent_classes_;
    Vertex* equivalent_class_type_;
    vector<vector<BasicOperator>> forest_refine_plan_;
    vector<vector<BasicOperator>> core_refine_plan_;
    vector<vector<Vertex>> forest_order_;
    bool has_core_;
    //
    cudaStream_t* streams_;
    unsigned** ptrs_d_candidate_bitmap_;
    Vertex** ptrs_d_candidate_array_;
    Vertex** ptrs_d_candidate_array_next_iteration_;
    unsigned* num_candidates_array_;
    //
    unsigned** d_ptrs_d_candidate_bitmap_;
    unsigned*** ptrs_query_neighbors_d_count_;
    unsigned*** ptrs_d_query_neighbors_count_;
    unsigned** ptrs_d_query_neighbors_count_threshold_;
    unsigned** ptrs_d_candidate_bitmap_next_iterartion_;
    vector<vector<Vertex>> re_core_neighbors_;
    vector<vector<Vertex>> op_core_neighbors_;
    vector<vector<Vertex>> reversed_re_core_neighbors_;
    Vertex** ptrs_d_invalid_candidate_array_;
    Vertex** ptrs_d_invalid_candidate_array_next_iteration_;
    unsigned* num_invalid_candidates_array_;
    unsigned*** d_ptrs_d_query_neighbors_count_;
    //
    vector<vector<BasicOperator>> original_core_refine_plan_;
    Vertex*** edges_first_vertex_arrays_ = NULL;
    Edge*** edges_offset_ = NULL;
    Vertex*** edges_second_vertex_arrays_ = NULL;
    Edge** num_candidate_edges_ = NULL;
    // join order
    Vertex* join_order_ = NULL;
    Vertex* id_to_pos_ = NULL;
    //
    Vertex num_all_embeddings_ = 0;
    vector<Vertex*> separated_embeddings_;
    vector<Vertex> num_separated_embeddings_;
    vector<vector<BlockChain>> blocks_tree_;
    vector<vector<BlockData>> datas_tree_;
    cudaStream_t h_to_d_;
    cudaStream_t perform_kernel_;
    cudaStream_t d_to_h_;
    cudaStream_t d_to_d_;
    // get neighbors on induced subgraph
    Vertex* num_induced_neighbors_ = NULL;
    // host
    Vertex*** induced_edges_first_vertex_arrays_ = NULL;
    Edge*** induced_edges_offset_ = NULL;
    Vertex*** induced_edges_second_vertex_arrays_ = NULL;
    Vertex** induced_candidate_num_ = NULL;
    Vertex** inversed_idx_ = NULL;
    // device
    Vertex*** d_induced_edges_first_vertex_arrays_ = NULL;
    Edge*** d_induced_edges_offset_ = NULL;
    Vertex*** d_induced_edges_second_vertex_arrays_ = NULL;
    Vertex** d_induced_candidate_num_ = NULL;
    Vertex** d_inversed_idx_ = NULL;

    MMatch(Graph* query_graph, Graph* data_graph, InvertedIndex<Label, Vertex>* inverted_data_label, int dev_id = 0) : GPUSubgraphMatcher(query_graph, data_graph) {
        // allocate profilers
        time_recorder_ = new TimeRecorder();
        cudaDeviceProp dev_props;
        cudaGetDeviceProperties(&dev_props, dev_id);
        device_memory_manager_ = new MemoryManager(dev_props.totalGlobalMem);
        dev_id_ = dev_id;
        // inverted_data_label_ = inverted_data_label;

        plain_query_graph_ = dynamic_cast<UndirectedPlainGraph*>(query_graph_);
        if (plain_query_graph_ == nullptr) {
            cout << "Error: transform abstract graph to plain graph; input query graph is not a plain graph" << endl;
            exit(0);
        }
        csr_data_graph_ = dynamic_cast<UndirectedCSRGraph*>(data_graph_);
        if (csr_data_graph_ == nullptr) {
            cout << "Error: transform abstract graph to CSR graph; input data graph is not a CSR graph" << endl;
            exit(0);
        }
        query_size_ = plain_query_graph_->num_vertices();
        data_size_ = csr_data_graph_->num_vertices();
    }
    void match() override;
    void copy_graph_to_GPU();
    void decompose_query_graph();
    void generate_refine_plan();
    void perform_refine_plan();
    void collect_candidate_edges();
    void generate_join_order();
    void join_candidate_edges();

    //
    bool exist_a_mapping(vector<Vertex> src_neighbors, vector<Vertex> dest_neighbors, vector<pair<Vertex, Vertex>> inversed_groups);
    void find_equivalent_classes();
    double compute_weight(Vertex root);
    Vertex select_min_spanning_tree();
    //
    void prepare_initialise();
    void kernel_initialise();
    void prepare_collect();
    void exclusive_sum(unsigned* d_array, unsigned* sum_array, Vertex size, unsigned &sum, cudaStream_t stream=0);
    void exclusive_sum(long long unsigned* d_array, long long unsigned* sum_array, long long unsigned size, long long unsigned &sum, cudaStream_t stream=0);
    void kernel_collect();
    void prepare_filter();
    void collect_successors(Vertex vertex_id, Vertex &num_successors, Vertex* &d_successors);
    void kernel_filter_forest();
    void collect_core_neighbors(vector<Vertex> core_vertices, Vertex* &num_re_core_neighbors, Vertex* &num_op_core_neighbors, 
                    Vertex* &num_reversed_re_core_neighbors, Vertex** &ptrs_d_re_core_neighbors, Vertex** &ptrs_d_op_core_neighbors, 
                    Vertex** &ptrs_d_reversed_re_core_neighbors);
    void perform_iteration(vector<Vertex> core_vertices, Vertex* num_re_core_neighbors, Vertex* num_op_core_neighbors, Vertex* num_reversed_re_core_neighbors, 
                    Vertex** ptrs_d_re_core_neighbors, Vertex** ptrs_d_op_core_neighbors, Vertex** ptrs_d_reversed_re_core_neighbors, bool is_pull);
    void decide_perform_mode(vector<Vertex> core_vertices, Vertex* num_re_core_neighbors, Vertex** ptrs_d_re_core_neighbors, unsigned** d_flag, bool &is_pull, bool &is_stop);
    void update_auxiliary_structure(vector<Vertex> core_vertices, Vertex* num_re_core_neighbors, Vertex* num_reversed_re_core_neighbors, 
                    Vertex** ptrs_d_re_core_neighbors, Vertex** ptrs_d_reversed_re_core_neighbors, bool is_pull);
    void kernel_filter_core();
    //
    void collect_forest_edges(Vertex class_id, Vertex** vertex_first_class_arrays, Edge** vertex_class_offset, 
                Vertex** vertex_second_class_arrays, Edge* vertex_num_edges);
    void collect_covered_neighbors(Vertex class_id, Vertex* &d_covered_neighbors, Vertex &num_covered_neighbors);
    void collect_core_edges(Vertex class_id, Vertex** vertex_first_class_arrays, Edge** vertex_class_offset, 
                Vertex** vertex_second_class_arrays, Edge* vertex_num_edges);
    //
    //
    void collect_neighbors_on_induced_subgraph(Vertex subgraph_size, Vertex** &induced_edges_first_vertex_arrays, Edge** &induced_edges_offset, 
                Vertex** &induced_edges_second_vertex_arrays, Vertex* &induced_candidate_num, Vertex* &inversed_idx, Vertex** &d_induced_edges_first_vertex_arrays, 
                Edge** &d_induced_edges_offset, Vertex** &d_induced_edges_second_vertex_arrays, Vertex* &d_induced_candidate_num, Vertex* &d_inversed_idx, 
                Vertex &num_induced_neighbors);
    void split_last_block_data(BlockData &last_layer_data, BlockChain last_layer_block, Vertex num_subblock, long long unsigned size_subblock);
    void extend_embeddings_recursively(BlockChain &last_layer_block, BlockData &last_layer_data);
    
    // void release();

#ifdef DEBUG

#endif
};

#endif