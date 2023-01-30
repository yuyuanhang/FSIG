/**
 * @file graph_format_transformer.h
 * @brief contains an abstract logic class GraphFormatTransformer and its child classes
 * @details child classes specify the format of input graph and output graph, i.e., which graph to which graph
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#ifndef GRAPH_FORMAT_TRANSFORMER_H
#define GRAPH_FORMAT_TRANSFORMER_H

#include <vector>
#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <fstream>
#include "../graph/graph.h"
#include "../utility/defines.h"

using std::vector;
using std::copy;
using std::cout;
using std::endl;
using std::sort;
using std::less;
using std::find;
using std::queue;
using std::set;
using std::ofstream;
using std::ifstream;

/*!
* An abstract logic class that transforms the way graph is stored
*/
class GraphFormatTransformer {
public:
    virtual void transform(Graph* input_graph, Graph* &output_graph) = 0;
};

/*!
* A logic class that transforms plain graph to CSR graph
*/
class UndirectedPlainToCSRTransformer : public GraphFormatTransformer {
public:
    void transform(Graph* input_graph, Graph* &output_graph) override;
};

/*!
* A logic class that transforms plain graph to CSR graph
*/
class UndirectedPlainToPCSRTransformer : public GraphFormatTransformer {
public:
    void transform(Graph* input_graph, Graph* &output_graph) override;
    static uint32_t hash(const void * key, int len, uint32_t seed) {
        return MurmurHash2(key, len, seed);
    }
    static uint32_t MurmurHash2(const void * key, int len, uint32_t seed) {
        // 'm' and 'r' are mixing constants generated offline.
        // they are not really 'magic', they just happen to work well.
        const uint32_t m = 0x5bd1e995;
        const int r = 24;
        // initialize the hash to a 'random' value
        uint32_t h = seed ^ len;
        // mix 4 bytes at a time into the hash
        const unsigned char* data = (const unsigned char*) key;
        while (len >= 4) {
            uint32_t k = *(uint32_t*) data;
            k *= m;
            k ^= k >> r;
            k *= m;
            h *= m;
            h ^= k;
            data += 4;
            len -= 4;
        }
        // handle the last few bytes of the input array
        switch (len) {
            case 3:
                h ^= data[2] << 16;
            case 2:
                h ^= data[1] << 8;
            case 1:
              h ^= data[0];
              h *= m;
        };
        // do a few final mixes of the hash to ensure the last few bytes are well-incorporated.
        h ^= h >> 13;
        h *= m;
        h ^= h >> 15;
        return h;
    }
    static void build_signature(bool column_oriented, Vertex* &signature_table, UndirectedPlainGraph* graph);
};
#endif