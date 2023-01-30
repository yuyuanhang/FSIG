#ifndef GSI_H
#define GSI_H

#include "subgraph_match.h"
#include "embedding.h"
#include <string>
#include <iostream>
#include <queue>
#include <cuda_runtime.h>
#include "../memory_manager/memory_manager.h"
#include "../graph/graph.h"
#include "../utility/defines.h"
#include "../utility/inverted_index.h"
#include "../utility/time_recorder.h"

using std::string;
using std::to_string;
using std::cout;
using std::endl;
using std::priority_queue;
using std::greater;

struct ScoredVertex{
    Vertex vertex_id_;
    float score_;
    ScoredVertex(Vertex vertex_id = INVALID, float score = INVALID) {
        vertex_id_ = vertex_id;
        score_ = score;
    }
    bool operator < (const ScoredVertex &v) const {
        return score_ < v.score_;
    }
    bool operator > (const ScoredVertex &v) const {
        return score_ > v.score_;
    }
};

#ifdef DEBUG
extern void print_priority_queue(priority_queue<ScoredVertex, vector<ScoredVertex>, greater<ScoredVertex>> pq);
#endif

/*!
* An logic class that implements GSI
* input: data graph and query graph
* output: embeddings
*/
class GSI : public GPUSubgraphMatcher {
public:
    // inspect time cost and space cost
    TimeRecorder* time_recorder_;
    MemoryManager* device_memory_manager_;
    int dev_id_ = 0;

    // data graph and query graph
    UndirectedPlainGraph* plain_query_graph_;
    UndirectedPCSRGraph* pcsr_data_graph_;
    Vertex query_size_;
    Vertex data_size_;
    Vertex* query_signature_table_;
    Vertex* data_signature_table_;
    Vertex** ptrs_d_candidate_array_;
    unsigned* size_candidate_sets_;
    Vertex* id_to_pos_;
    Vertex* pos_to_id_;
    Vertex* d_embeddings_;

    GSI(Graph* query_graph, Graph* data_graph, InvertedIndex<Label, Vertex>* inverted_data_label, Vertex* query_signature_table, 
            Vertex* data_signature_table, int dev_id = 0) : GPUSubgraphMatcher(query_graph, data_graph) {
        // allocate profilers
        time_recorder_ = new TimeRecorder();
        cudaDeviceProp dev_props;
        cudaGetDeviceProperties(&dev_props, dev_id);
        device_memory_manager_ = new MemoryManager(dev_props.totalGlobalMem);
        dev_id_ = dev_id;

        // specify data graph and query graph
        plain_query_graph_ = dynamic_cast<UndirectedPlainGraph*>(query_graph_);
        if (plain_query_graph_ == nullptr) {
            cout << "Error: transform abstract graph to plain graph; input query graph is not a plain graph" << endl;
            exit(0);
        }
        pcsr_data_graph_ = dynamic_cast<UndirectedPCSRGraph*>(data_graph_);
        if (pcsr_data_graph_ == nullptr) {
            cout << "Error: transform abstract graph to CSR graph; input data graph is not a PCSR graph" << endl;
            exit(0);
        }
        query_size_ = plain_query_graph_->num_vertices();
        data_size_ = pcsr_data_graph_->num_vertices();
        query_signature_table_ = query_signature_table;
        data_signature_table_ = data_signature_table;
    }

    void exclusive_sum(unsigned* d_array, unsigned* sum_array, Vertex size, unsigned &sum, cudaStream_t stream=0);
    bool signature_filter();
    void generate_join_order();
    void get_neighbors_on_induced_subgraph(vector<Vertex> &induced_subgraph_neighbors, Vertex order);
    void join_embeddings();
    void match() override;
    
    // void release();
};

#endif