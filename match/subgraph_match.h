/**
 * @file subgraph_match.h
 * @brief contains an abstract logic class SubgraphMatcher and its child classes
 * @details child classes specify the way that embeddings are enumerated
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#ifndef SUBGRAPH_MATCH_H
#define SUBGRAPH_MATCH_H

#include <iostream>
#include <cuda_runtime.h>
#include "embedding.h"
#include "../graph/graph.h"
#include "../utility/defines.h"

using std::cout;
using std::endl;

// GPU macro define
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        cout << "CUDA error at: " << file << ":" << line << endl;
        cout << cudaGetErrorString(err) << " " << func << endl;
        exit(0);
    }
}

/*!
* An abstract logic class
* input: data graph and query graph
* output: embeddings
*/
class GPUSubgraphMatcher {
public:
    Graph* query_graph_;
    Graph* data_graph_;
    Embedding* embedding_;

    GPUSubgraphMatcher(Graph* query_graph, Graph* data_graph) {
        query_graph_ = query_graph;
        data_graph_ = data_graph;
        embedding_ = new Embedding(query_graph_->num_vertices());
    }
    virtual void match() = 0;
    void init_GPU(int dev_id) {
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        if (num_devices == 0) {
            cout << "Error: no devices supporting CUDA" << endl;
            exit(0);
        }
        cudaSetDevice(dev_id);
        cudaDeviceProp dev_props;
        if (cudaGetDeviceProperties(&dev_props, dev_id) == 0) {
            cout << "Using device " << dev_id << " " << dev_props.name << endl;
            cout << "Global memory: " << (double)dev_props.totalGlobalMem/(1024*1024*1024) << "GB" << endl << \
                "Shared memory: " << (double)dev_props.sharedMemPerBlock/1024 << "KB" << endl << \
                "maxThreadsPerBlock: " << dev_props.maxThreadsPerBlock << endl << \
                "the number of SMs: " << dev_props.multiProcessorCount << endl << \
                "version: " << dev_props.major << "." << dev_props.minor << endl << \
                "clock: " << (double)dev_props.clockRate/(1024*1024) << "GHz" << endl;
        }

        int* warmup = NULL;
        cudaMalloc(&warmup, sizeof(int));
        cudaFree(warmup);

        // NOTICE: the memory alloced by cudaMalloc is different from the GPU heap(for new/malloc in kernel functions)
        size_t size = 0x7fffffff;
        size *= 2;
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, size);
        cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
        cout << "the size of heap limit: " << (double)size/(1024*1024) << "MB" << endl;
    }
};

#endif