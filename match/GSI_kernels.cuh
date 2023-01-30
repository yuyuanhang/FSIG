#ifndef GSI_KERNELS_CUH
#define GSI_KERNELS_CUH

#include "../utility/defines.h"

__constant__ unsigned c_query_signature_table[SIGNUM];

__global__ void filter_kernel(Vertex* d_data_signature_table, unsigned* d_bitmap, Vertex data_size) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size) {
        return; 
    }

    unsigned flag = (c_query_signature_table[0] == d_data_signature_table[idx]) ? 1 : 0;
    for (unsigned i = 1; i < SIGNUM; i++) {
        unsigned query_sig = c_query_signature_table[i];
        unsigned data_sig = d_data_signature_table[data_size * i + idx];
        
        if(flag) {
            flag = ((query_sig & data_sig) == query_sig) ? 1 : 0;
        }
    }
    d_bitmap[idx] = flag;
}

__global__ void scatter_kernel(unsigned* d_bitmap, Vertex* d_candidate_array, Vertex data_size) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size) {
        return; 
    }

    unsigned pos = d_bitmap[idx];
    if (pos != d_bitmap[idx + 1]) {
        d_candidate_array[pos] = idx;
    }
}

__global__ void GSI_generate_bitmap_kernel(unsigned* d_bitmap, Vertex* d_candidate_array, Vertex num_candidates) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) {
        return; 
    }
    
    Vertex vertex_id = d_candidate_array[idx];
    unsigned pos = vertex_id >> 5;
    unsigned residual = vertex_id & 0x1f;
    residual = 1 << residual;
    atomicOr(d_bitmap + pos, residual);
}

__constant__ Vertex* c_embeddings;
__constant__ unsigned* c_bitmap;
__constant__ Edge c_num_result_rows;
__constant__ Vertex c_num_result_cols;
__constant__ Vertex c_neighbor_pos;
__constant__ Edge* c_offset;
__constant__ Vertex* c_edges;
__constant__ Vertex c_num_keys;
__constant__ Vertex* c_degree_bound;

__device__ uint32_t MurmurHash2(const void * key, int len, uint32_t seed) {
    const uint32_t m = 0x5bd1e995;
    const int r = 24;
    // initialize the hash to a 'random' value
    uint32_t h = seed ^ len;
    // mix 4 bytes at a time into the hash
    const unsigned char * data = (const unsigned char *) key;
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

__global__ void preallocate_kernel(Vertex* d_degree_bound) {
    __shared__ unsigned buckets_in_block[1024];
    __shared__ unsigned identified_offset[64];

    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    Vertex lane_id = idx & 0x1f;
    Vertex group_id = idx >> 5;
    Vertex block_group = threadIdx.x >> 5;
    Vertex block_presum = threadIdx.x & 0xffffffe0;

    if (group_id >= c_num_result_rows) {
        return; 
    }

    Vertex* crt_embedding = c_embeddings + group_id * c_num_result_cols;
    Vertex data_neighbor = crt_embedding[c_neighbor_pos];

    if (lane_id == 0) {
        identified_offset[2 * block_group] = INVALID;
        identified_offset[2 * block_group + 1] = INVALID;
    }
    __syncthreads();

    unsigned target_bucket = MurmurHash2(&data_neighbor, 4, HASHSEED) % c_num_keys;
    while (identified_offset[block_group] == INVALID && target_bucket != INVALID) {
        buckets_in_block[block_presum + lane_id] = c_offset[32 * target_bucket + lane_id];
        if (lane_id < 30 && (lane_id&1) == 0) {
            if (buckets_in_block[block_presum + lane_id] == data_neighbor) {
                identified_offset[2 * block_group] = buckets_in_block[block_presum + lane_id + 1];
                identified_offset[2 * block_group + 1] = buckets_in_block[block_presum + lane_id + 3];
            }
        }
        __syncthreads();
        target_bucket = buckets_in_block[block_presum+30];
    }
    if (lane_id == 0) {
        d_degree_bound[group_id] = identified_offset[2 * block_group + 1] - identified_offset[2 * block_group];
    }
}

__global__ void load_target_data_kernel(Vertex* d_target_data_vertices, Vertex* d_filtered_degree) {
    __shared__ unsigned buckets_in_block[1024]; // load the whole bucket
    __shared__ unsigned mappings_in_block[1024]; // load the whole mapping
    __shared__ unsigned neighbors_in_block[1024]; // load the position in column_index array (neighbor list)
    __shared__ unsigned counts[32];
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    Vertex lane_id = idx & 0x1f;
    Vertex group_id = idx >> 5;
    Vertex block_group = threadIdx.x >> 5;
    Vertex block_presum = threadIdx.x & 0xffffffe0;

    if (group_id >= c_num_result_rows) {
        return; 
    }

    if (lane_id < c_num_result_cols) {
        mappings_in_block[block_presum + lane_id] = c_embeddings[group_id * c_num_result_cols + lane_id];
    }
    __syncthreads();
    Vertex data_neighbor = mappings_in_block[block_presum + c_neighbor_pos];

    if (lane_id == 0) {
        neighbors_in_block[block_presum] = INVALID;
    }
    __syncthreads();

    unsigned target_bucket = MurmurHash2(&data_neighbor, 4, HASHSEED) % c_num_keys;
    while (neighbors_in_block[block_presum] == INVALID && target_bucket != INVALID) {
        buckets_in_block[block_presum + lane_id] = c_offset[32 * target_bucket + lane_id];
        if (lane_id < 30 && (lane_id&1) == 0) {
            if (buckets_in_block[block_presum + lane_id] == data_neighbor) {
                neighbors_in_block[block_presum] = buckets_in_block[block_presum + lane_id + 1];
                neighbors_in_block[block_presum + 1] = buckets_in_block[block_presum + lane_id + 3];
            }
        }
        __syncthreads();
        target_bucket = buckets_in_block[block_presum+30];
    }

    if (neighbors_in_block[block_presum] == INVALID) { // not found
        if (lane_id == 0) {
            d_filtered_degree[group_id] = 0;
        }
        return;
    }

    Vertex* crt_target_data = d_target_data_vertices + c_degree_bound[group_id];
    Vertex size = neighbors_in_block[block_presum + 1] - neighbors_in_block[block_presum];
    Vertex* list = c_edges + neighbors_in_block[block_presum];
    Vertex pos = 0;
    Vertex loop = size >> 5;
    size = size & 0x1f;

    // pred is a flag that shows if the candidate is valid
    // presum is a exclusive sum of pred
    unsigned pred, presum;
    unsigned num_flushed_vertices = 0;
    counts[block_group] = 0;

    for (Vertex i = 0; i < loop; i++, pos+=32) {
        // load neighbor list
        buckets_in_block[block_presum + lane_id] = list[pos + lane_id];
        // check isomorphism
        unsigned j;
        for (j = 0; j < c_num_result_cols; j++) {
            if (mappings_in_block[block_presum + j] == buckets_in_block[block_presum + lane_id]) {
                break;
            }
        }

        pred = 0;
        if (j == c_num_result_cols) {
            unsigned pos = buckets_in_block[block_presum + lane_id] >> 5;
            unsigned residual = buckets_in_block[block_presum + lane_id] & 0x1f;
            residual = 1 << residual;
            if ((c_bitmap[pos] & residual) == residual) {
                pred = 1;
            }
        }

        presum = pred;
        // prefix sum in a warp to find positions
        for (unsigned stride = 1; stride < 32; stride <<= 1) {
            unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
            if (lane_id >= stride) {
                presum += tmp;
            }
            __syncthreads();
        }
        unsigned total = __shfl_sync(FULL_MASK, presum, 31);
        // transform inclusive prefix sum to exclusive prefix sum
        presum = __shfl_up_sync(FULL_MASK, presum, 1);
        if (lane_id == 0) {
            presum = 0;
        }
        // write to corresponding position
        if (pred == 1) {
            if (counts[block_group] + presum < 32) {
                neighbors_in_block[block_presum + counts[block_group] + presum] = buckets_in_block[block_presum + lane_id];
            }
        }
        // flush cache
        if (counts[block_group] + total >= 32) {
            crt_target_data[num_flushed_vertices + lane_id] = neighbors_in_block[block_presum + lane_id];
            num_flushed_vertices += 32;
            if (pred == 1) {
                unsigned pos = counts[block_group] + presum;
                if (pos >= 32) {
                    neighbors_in_block[block_presum + pos - 32] = buckets_in_block[block_presum + lane_id];
                }
            }
            counts[block_group] = counts[block_group] + total - 32;
        } else {
            counts[block_group] += total;
        }
    }
    presum = pred = 0;
    if (lane_id < size) {
        // load neighbor list
        buckets_in_block[block_presum + lane_id] = list[pos + lane_id];

        // check isomorphism
        unsigned i;
        for (i = 0; i < c_num_result_cols; i++) {
            if (mappings_in_block[block_presum + i] == buckets_in_block[block_presum + lane_id]) {
                break;
            }
        }
        if (i == c_num_result_cols) {
            unsigned pos = buckets_in_block[block_presum + lane_id] >> 5;
            unsigned residual = buckets_in_block[block_presum + lane_id] & 0x1f;
            residual = 1 << residual;
            if ((c_bitmap[pos] & residual) == residual) {
                pred = 1;
            }
        }
        presum = pred;
    }
    __syncthreads();
    // prefix sum in a warp to find positions
    for (unsigned stride = 1; stride < 32; stride <<= 1) {
        unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
        if (lane_id >= stride) {
            presum += tmp;
        }
        __syncthreads();
    }
    unsigned total = __shfl_sync(FULL_MASK, presum, 31);
    presum = __shfl_up_sync(FULL_MASK, presum, 1);
    if (lane_id == 0) {
        presum = 0;
    }
    if (pred == 1) {
        if (counts[block_group] + presum < 32) {
            neighbors_in_block[block_presum + counts[block_group] + presum] = buckets_in_block[block_presum + lane_id];
        }
    }
    if (counts[block_group] + total >= 32) {
        crt_target_data[num_flushed_vertices + lane_id] = neighbors_in_block[block_presum + lane_id];
        num_flushed_vertices += 32;
        if (pred == 1) {
            unsigned pos = counts[block_group] + presum;
            if (pos >= 32) {
                crt_target_data[num_flushed_vertices + pos - 32] = buckets_in_block[block_presum + lane_id];
            }
        }
        num_flushed_vertices += counts[block_group] + total - 32;
    } else {
        if (lane_id < counts[block_group] + total) {
            crt_target_data[num_flushed_vertices + lane_id] = neighbors_in_block[block_presum + lane_id];
        }
        num_flushed_vertices += counts[block_group] + total;
    }
    
    if (lane_id == 0) {
        d_filtered_degree[group_id] = num_flushed_vertices;
    }
}

__device__ unsigned GSI_binary_search(unsigned _key, unsigned* _array, unsigned _array_num) {
    if (_array_num == 0 || _array == NULL) {
        return INVALID;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key) {
        return _array_num - 1;
    }

    if (_last < _key || _first > _key) {
        return INVALID;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high) {
        mid = (high - low) / 2 + low;   // same to (low + high) / 2
        if (_array[mid] == _key) {
            return mid;
        }
        if (_array[mid] > _key) {
            high = mid - 1;
        }
        else {
            low = mid + 1;
        }
    }
    return INVALID;
}

__global__ void intersection_kernel(Vertex* d_target_data_vertices, Vertex* d_filtered_degree) {
    __shared__ unsigned buckets_in_block[1024]; // load list1
    __shared__ unsigned mappings_in_block[1024]; // load list2
    __shared__ unsigned neighbors_in_block[1024]; // intersection result
    __shared__ unsigned counts[32];
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    Vertex lane_id = idx & 0x1f;
    Vertex group_id = idx >> 5;
    Vertex block_group = threadIdx.x >> 5;
    Vertex block_presum = threadIdx.x & 0xffffffe0;

    if (group_id >= c_num_result_rows) {
        return; 
    }

    unsigned crt_filtered_degree = d_filtered_degree[group_id];
    if (crt_filtered_degree == 0) { //early termination
        return;
    }
    
    if (lane_id < c_num_result_cols) {
        mappings_in_block[block_presum + lane_id] = c_embeddings[group_id * c_num_result_cols + lane_id];
    }
    __syncthreads();
    Vertex data_neighbor = mappings_in_block[block_presum + c_neighbor_pos];

    if (lane_id == 0) {
        neighbors_in_block[block_presum] = INVALID;
    }
    __syncthreads();

    unsigned target_bucket = MurmurHash2(&data_neighbor, 4, HASHSEED) % c_num_keys;
    while (neighbors_in_block[block_presum] == INVALID && target_bucket != INVALID) {
        buckets_in_block[block_presum + lane_id] = c_offset[32 * target_bucket + lane_id];
        if (lane_id < 30 && (lane_id&1) == 0) {
            if (buckets_in_block[block_presum + lane_id] == data_neighbor) {
                neighbors_in_block[block_presum] = buckets_in_block[block_presum + lane_id + 1];
                neighbors_in_block[block_presum + 1] = buckets_in_block[block_presum + lane_id + 3];
            }
        }
        __syncthreads();
        target_bucket = buckets_in_block[block_presum+30];
    }
    
    if (neighbors_in_block[block_presum] == INVALID) { // not found
        if (lane_id == 0) {
            d_filtered_degree[group_id] = 0;
        }
        return;
    }

    Vertex* crt_target_data = d_target_data_vertices + c_degree_bound[group_id];
    Vertex list_num = neighbors_in_block[block_presum + 1] - neighbors_in_block[block_presum];
    Vertex* list = c_edges + neighbors_in_block[block_presum];
    Vertex pos1 = 0, pos2 = 0;
    Vertex pred, presum;
    int choice = 0;
    unsigned num_flushed_vertices = 0;
    counts[block_group] = 0;
    while (pos1 < crt_filtered_degree && pos2 < list_num) {
        if (choice <= 0) {
            buckets_in_block[block_presum + lane_id] = INVALID;
            if (pos1 + lane_id < crt_filtered_degree) {
                buckets_in_block[block_presum + lane_id] = crt_target_data[pos1 + lane_id];
            }
        }
        if (choice >= 0) {
            mappings_in_block[block_presum + lane_id] = INVALID;
            if(pos2 + lane_id < list_num) {
                mappings_in_block[block_presum + lane_id] = list[pos2 + lane_id];
            }
        }
        pred = 0;  //some threads may fail in the judgement below
        unsigned valid1 = (pos1 + 32 < crt_filtered_degree) ? 32 : (crt_filtered_degree - pos1);
        unsigned valid2 = (pos2 + 32 < list_num) ? 32 : (list_num - pos2);
        if (pos1 + lane_id < crt_filtered_degree) {
            pred = GSI_binary_search(buckets_in_block[block_presum + lane_id], mappings_in_block + block_presum, valid2);
            if (pred != INVALID) {
                pred = 1;
            } else {
                pred = 0;
            }
        }
        presum = pred;
        for (unsigned stride = 1; stride < 32; stride <<= 1) {
            unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
            if (lane_id >= stride) {
                presum += tmp;
            }
            __syncthreads();
        }
        unsigned total = __shfl_sync(FULL_MASK, presum, 31);
        presum = __shfl_up_sync(FULL_MASK, presum, 1);
        if (lane_id == 0) {
            presum = 0;
        }
        if (pred == 1) {
            if (counts[block_group] + presum < 32) {
                neighbors_in_block[block_presum + counts[block_group] + presum] = buckets_in_block[block_presum + lane_id];
            }
        }
        if (counts[block_group] + total >= 32) {
            crt_target_data[num_flushed_vertices + lane_id] = neighbors_in_block[block_presum + lane_id];
            num_flushed_vertices += 32;
            if (pred == 1) {
                unsigned pos = counts[block_group] + presum;
                if (pos >= 32) {
                    neighbors_in_block[block_presum + pos - 32] = buckets_in_block[block_presum + lane_id];
                }
            }
            counts[block_group] = counts[block_group] + total - 32;
        } else {
            counts[block_group] += total;
        }

        // set the next movement
        choice = buckets_in_block[block_presum + valid1 - 1] - mappings_in_block[block_presum + valid2 - 1];
        if (choice <= 0) {
            pos1 += 32;
        }
        if (choice >= 0) {
            pos2 += 32;
        }
    }

    // flush the buffer into global memory
    if (lane_id < counts[block_group]) {
        crt_target_data[num_flushed_vertices + lane_id] = neighbors_in_block[block_presum + lane_id];
    }
    __syncthreads();
    num_flushed_vertices += counts[block_group];

    if (lane_id == 0) {
        d_filtered_degree[group_id] = num_flushed_vertices;
    }
}

__global__ void link_kernel(Vertex* d_target_data_vertices, unsigned* d_degree_bound, unsigned* d_filtered_degree, Vertex* d_new_embeddings) {
    __shared__ Vertex cache[1024]; // store embeddings
    __shared__ unsigned swpos[32];

    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    Vertex lane_id = idx & 0x1f;
    Vertex group_id = idx >> 5;
    Vertex block_group = threadIdx.x >> 5;
    Vertex block_presum = threadIdx.x & 0xffffffe0;

    unsigned tmp_begin = 0, start = 0, size = 0;
    if (group_id >= c_num_result_rows) {
        return;
    }
    
    tmp_begin = d_degree_bound[group_id];
    start = d_filtered_degree[group_id];
    size = d_filtered_degree[group_id + 1] - start;
    start *= (c_num_result_cols + 1);

    if (lane_id == 0) {
        if (size > 0) {
            memcpy(cache + block_presum, c_embeddings + group_id * c_num_result_cols, sizeof(Vertex) * c_num_result_cols);
        }
    }
    __syncthreads();
    unsigned curr = 0;
    Vertex* record = cache + block_presum;
    // use a block to deal with tasks >=1024

    while (true) {
        if (threadIdx.x == 0) {
            swpos[0] = INVALID;
        }
        __syncthreads();
        if (size >= curr + 1024) {
            swpos[0] = block_group;
        }
        __syncthreads();
        if (swpos[0] == INVALID) {
            break;
        }
        Vertex* target_embedding = cache + 32 * swpos[0];
        if (swpos[0] == block_group) {
            swpos[1] = tmp_begin;
            swpos[2] = start;
            swpos[3] = curr;
            swpos[4] = size;
        }
        __syncthreads();
        while (swpos[3] + 1023 < swpos[4]) {
            unsigned pos = (c_num_result_cols + 1) * (swpos[3] + threadIdx.x);
            memcpy(d_new_embeddings + swpos[2] + pos, target_embedding, sizeof(Vertex) * c_num_result_cols);
            d_new_embeddings[swpos[2] + pos + c_num_result_cols] = d_target_data_vertices[swpos[1] + swpos[3] + threadIdx.x];
            if (threadIdx.x == 0) {
                swpos[3] += 1024;
            }
            __syncthreads();
        }
        if (swpos[0] == block_group) {
            curr = swpos[3];
        }
        __syncthreads();
    }
    __syncthreads();

    while (curr < size) {
        // this judgement is fine, only causes divergence in the end
        if (curr + lane_id < size) {
            unsigned pos = (c_num_result_cols + 1) * (curr + lane_id);
            memcpy(d_new_embeddings + start + pos, record, sizeof(Vertex) * c_num_result_cols);
            d_new_embeddings[start + pos + c_num_result_cols] = d_target_data_vertices[tmp_begin + curr + lane_id];
        }
        curr += 32;
    }
}

#endif