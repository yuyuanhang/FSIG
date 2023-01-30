#ifndef MMATCH_KERNELS_CUH
#define MMATCH_KERNELS_CUH

#include "../utility/defines.h"

__global__ void initialise_kernel(Label* d_data_labels, Edge* d_data_offset, unsigned* d_candidate_set, Vertex data_size, Label vertex_label, Vertex vertex_degree) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size) {
        return; 
    }
    Label data_label = d_data_labels[idx];
    Vertex data_degree = d_data_offset[idx+1] - d_data_offset[idx];
    if (data_label == vertex_label && data_degree >= vertex_degree) {
        d_candidate_set[idx] = 1;
    }
}

__global__ void scatter_kernel(unsigned* d_candidate_set, unsigned* d_exclusive_sum, Vertex* d_candidate_array, Vertex data_size, unsigned flag) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size) {
        return; 
    }
    if (d_candidate_set[idx] == flag) {
        d_candidate_array[d_exclusive_sum[idx]] = idx;
    }
}

__global__ void count_neighbors_kernel(unsigned** d_candidate_set, Edge* d_data_offset, Vertex* d_data_edges, unsigned* d_candidate_array, Vertex num_candidates,
											Vertex* query_neighbors, Vertex num_neighbors, unsigned** d_neighbors_count, Vertex vertex_id) {
	extern __shared__ Vertex shared_space[];
	Vertex* data_neighbors = shared_space; // the space is blockDim.x
	unsigned* count = (unsigned*)(data_neighbors+blockDim.x); // the space is num_warp * num_neighbors
	Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
	Vertex warp_id = threadIdx.x >> 5;
	Vertex lane_id = threadIdx.x & 0x1f;
	Vertex global_id = idx >> 5;
	if (global_id >= num_candidates) {
        return; 
    }

    Vertex data_candidate = d_candidate_array[global_id]; // each warp is for a candidate
    if (d_candidate_set[vertex_id][data_candidate] == 2) { // has been filtered by optimized operation
        return;
    }
    Vertex begin = d_data_offset[data_candidate]; // the position of neighbor list
    Vertex n_begin = warp_id << 5; // the position of neighbor list in shared memory 
    Vertex c_begin = warp_id * num_neighbors; // the position of count array in shared memory
    Vertex size = d_data_offset[data_candidate+1] - begin; // length of neighbor list, num of data neighbors
    Vertex loop = size >> 5;
    size = size & 0x1f;

    // initialise shared memory, here, we assume that the number of neighbors is less than 32
    if (lane_id < num_neighbors) {
    	count[c_begin + lane_id] = 0;
    }

    for (Vertex i = 0; i < loop; i++) {
    	// load data neighbors of this batch from global memory to shared memory
    	data_neighbors[n_begin+lane_id] = d_data_edges[begin+lane_id];
    	// iterate on each query neighbor
    	for (Vertex j = 0; j < num_neighbors; j++) {
    		Vertex query_neighbor = query_neighbors[j];
    		Vertex data_neighbor = data_neighbors[n_begin+lane_id];
    		// read the flag of this data neighbor
    		unsigned crt_count = d_candidate_set[query_neighbor][data_neighbor];

    		// aggregate the count
        	crt_count += __shfl_down_sync(FULL_MASK, crt_count, 16);
    		crt_count += __shfl_down_sync(FULL_MASK, crt_count, 8);
    		crt_count += __shfl_down_sync(FULL_MASK, crt_count, 4);
    		crt_count += __shfl_down_sync(FULL_MASK, crt_count, 2);
    		crt_count += __shfl_down_sync(FULL_MASK, crt_count, 1);
        	
        	// write this count
        	if (lane_id == 0) {
        		count[c_begin + j] += crt_count;
        	}
    	}
    	// this batch has been processed, move to next batch
    	begin += 32;
    }
    // process residual data neighbors
    if (lane_id < size) {
    	// load data neighbors of this batch from global memory to shared memory
    	data_neighbors[n_begin+lane_id] = d_data_edges[begin+lane_id];
    }
    // iterate on each query neighbor
    for (Vertex i = 0; i < num_neighbors; i++) {
    	Vertex query_neighbor = query_neighbors[i];
    	Vertex data_neighbor;
    	if (lane_id < size) {
    		data_neighbor = data_neighbors[n_begin+lane_id];
    	}
    	// read the flag of this data neighbor
    	unsigned crt_count = 0;
    	if (lane_id < size) {
            if (d_candidate_set[query_neighbor][data_neighbor] == 1) { // check op kernel leads that d_candidate_set flag can be 2
                crt_count = 1;
            } else {
                crt_count = 0;
            }
    	}

    	// aggregate the count
        crt_count += __shfl_down_sync(FULL_MASK, crt_count, 16);
    	crt_count += __shfl_down_sync(FULL_MASK, crt_count, 8);
    	crt_count += __shfl_down_sync(FULL_MASK, crt_count, 4);
    	crt_count += __shfl_down_sync(FULL_MASK, crt_count, 2);
    	crt_count += __shfl_down_sync(FULL_MASK, crt_count, 1);
        	
        // write this count
        if (lane_id == 0) {
        	count[c_begin + i] += crt_count;
        	d_neighbors_count[query_neighbor][data_candidate] = count[c_begin + i];
        }
    }
}

__global__ void alleviate_neighbors_kernel(unsigned** d_candidate_set, Edge* d_data_offset, Vertex* d_data_edges, unsigned* d_candidate_array, Vertex num_candidates,
                                            Vertex* query_neighbors, Vertex num_neighbors, unsigned*** d_neighbors_count, Vertex vertex_id) {
    extern __shared__ Vertex shared_space[];
    Vertex* data_neighbors = shared_space; // the space is blockDim.x
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    Vertex warp_id = threadIdx.x >> 5;
    Vertex lane_id = threadIdx.x & 0x1f;
    Vertex global_id = idx >> 5;
    if (global_id >= num_candidates) {
        return; 
    }

    Vertex data_candidate = d_candidate_array[global_id]; // each warp is for a candidate
    if (d_candidate_set[vertex_id][data_candidate] == 2) { // has been filtered by optimized operation
        return;
    }
    Vertex begin = d_data_offset[data_candidate]; // the position of neighbor list
    Vertex n_begin = warp_id << 5; // the position of neighbor list in shared memory
    Vertex size = d_data_offset[data_candidate+1] - begin; // length of neighbor list, num of data neighbors
    Vertex loop = size >> 5;
    size = size & 0x1f;

    for (Vertex i = 0; i < loop; i++) {
        // load data neighbors of this batch from global memory to shared memory
        data_neighbors[n_begin+lane_id] = d_data_edges[begin+lane_id];
        // iterate on each query neighbor
        for (Vertex j = 0; j < num_neighbors; j++) {
            Vertex query_neighbor = query_neighbors[j];
            Vertex data_neighbor = data_neighbors[n_begin+lane_id];
            // 
            if (d_candidate_set[query_neighbor][data_neighbor] == 1) {
                atomicSub(d_neighbors_count[query_neighbor][vertex_id]+data_neighbor, 1);
            }
        }
        // this batch has been processed, move to next batch
        begin += 32;
    }
    // process residual data neighbors
    if (lane_id < size) {
        // load data neighbors of this batch from global memory to shared memory
        data_neighbors[n_begin+lane_id] = d_data_edges[begin+lane_id];
    }
    // iterate on each query neighbor
    for (Vertex i = 0; i < num_neighbors; i++) {
        Vertex query_neighbor = query_neighbors[i];
        Vertex data_neighbor;
        if (lane_id < size) {
            data_neighbor = data_neighbors[n_begin+lane_id];
            if (d_candidate_set[query_neighbor][data_neighbor] == 1) {
                atomicSub(d_neighbors_count[query_neighbor][vertex_id]+data_neighbor, 1);
            }
        }
    }
}

__global__ void check_op_kernel(unsigned** d_candidate_set, Vertex* d_candidate_array, Vertex* d_neighbors, 
                                    Vertex num_candidates, Vertex num_neighbors, unsigned* d_candidate_set_next_iteration) {
	Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_candidates) {
        return; 
    }
    Vertex data_vertex = d_candidate_array[idx];
    for (Vertex i = 0; i < num_neighbors; i++) {
    	Vertex query_neighbor = d_neighbors[i];
    	if (d_candidate_set[query_neighbor][data_vertex] == 0) {
    		d_candidate_set_next_iteration[data_vertex] = 2;
    		break;
    	}
    }
}

__global__ void check_re_kernel(unsigned* d_candidate_set, Vertex* d_candidate_array, unsigned** d_neighbors_count, Vertex* target_count, 
                                    Vertex num_candidates, Vertex* query_neighbors, Vertex num_neighbors) {
	Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_candidates) {
        return; 
    }

    Vertex data_vertex = d_candidate_array[idx];

    if (d_candidate_set[data_vertex] == 2) {
    	return;
    }

    bool is_valid = true;
    for (Vertex i = 0; i < num_neighbors; i++) {
        Vertex query_neighbor = query_neighbors[i];
    	if (d_neighbors_count[query_neighbor][data_vertex] < target_count[query_neighbor]) {
    		is_valid = false;
    		break;
    	}
    }
    if (!is_valid) {
    	d_candidate_set[data_vertex] = 2;
    }
}

__global__ void count_valid_candidate_kernel(unsigned* d_candidate_set, Vertex* d_candidate_array, unsigned* d_flag, Vertex num_candidates) {
	Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_candidates) {
        return; 
    }
    Vertex data_vertex = d_candidate_array[idx];
    if (d_candidate_set[data_vertex] == 2) {
    	d_flag[idx] = 0;
    } else if (d_candidate_set[data_vertex] == 1) {
    	d_flag[idx] = 1;
    }
}

__global__ void count_invalid_candidate_kernel(unsigned* d_candidate_set, Vertex* d_candidate_array, unsigned* d_flag, Vertex num_candidates) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) {
        return; 
    }
    Vertex data_vertex = d_candidate_array[idx];
    if (d_candidate_set[data_vertex] == 2) {
        d_flag[idx] = 1;
    } else if (d_candidate_set[data_vertex] == 1) {
        d_flag[idx] = 0;
    }
}

__global__ void compact_kernel(unsigned* d_exclusive_sum, Vertex* d_candidate_array, Vertex* d_candidate_array_next_iteration, Vertex num_candidates) {
	Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) {
        return; 
    }
    Vertex data_candidate = d_candidate_array[idx];
    unsigned crt_presum = d_exclusive_sum[idx];
    unsigned next_presum = d_exclusive_sum[idx+1];

    if (crt_presum != next_presum) {
        d_candidate_array_next_iteration[crt_presum] = data_candidate;
    }
}

__global__ void change_flag_kernel(unsigned* d_candidate_set, Vertex data_size) {
	Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= data_size) {
        return; 
    }
    if (d_candidate_set[idx] == 2) {
    	d_candidate_set[idx] = 0;
    }
}

__global__ void clear_counter_kernel2(Vertex* d_candidate_array, Vertex num_candidates, unsigned** d_neighbors_count, 
                            Vertex* query_neighbors, Vertex num_neighbors) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) {
        return; 
    }
    Vertex data_vertex = d_candidate_array[idx];
    for (Vertex i = 0; i < num_neighbors; i++) {
        Vertex query_neighbor = query_neighbors[i];
        d_neighbors_count[query_neighbor][data_vertex] = 0;
    }
}

__global__ void clear_counter_kernel(unsigned* d_candidate_set, unsigned* d_neighbors_count, Vertex data_size) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size) {
        return; 
    }
    if (d_candidate_set[idx] == 0) {
        d_neighbors_count[idx] = 0;
    }
}

__global__ void compact_kernel2(unsigned* candidate_pos_in_array, unsigned* d_candidate_set, unsigned* neighbor_count, Vertex* compacted_count, Vertex data_size) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_size) {
        return; 
    }
    Vertex crt_pos = candidate_pos_in_array[idx];
    unsigned is_valid = d_candidate_set[idx];

    if (is_valid) {
        compacted_count[crt_pos+1] = neighbor_count[idx+1];
    }
}

__global__ void re_examine_kernel(Edge* d_data_offset, Vertex* d_data_edges, unsigned** d_candidate_set, Vertex data_size, Vertex num_candidates, 
                    Vertex neighbor, Vertex* d_candidate_array, Edge* d_edge_offset, Vertex* d_neighbor_list) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;
    Vertex group_id = idx / 32;
    Vertex lane_id = idx % 32;
    if (group_id >= num_candidates) {
        return; 
    }
    Vertex crt_candidate = d_candidate_array[group_id];
    
    Edge write_pos = d_edge_offset[group_id];
    Edge begin = d_data_offset[crt_candidate];
    Vertex degree = d_data_offset[crt_candidate+1] - begin;
    Vertex loop = degree / 32;
    degree = degree % 32;

    // each thread get adjacent vertex of mu as mv
    Vertex base = 0;
    unsigned pred = 0, presum = 0;
    Vertex data_neighbor;
    for (Vertex i = 0; i < loop; i++, base += 32) {
        data_neighbor = d_data_edges[begin+base+lane_id];
        pred = d_candidate_set[neighbor][data_neighbor];
        presum = pred;
        
        for (unsigned stride = 1; stride < 32; stride <<= 1) {
            unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
            if (lane_id >= stride) {
                presum += tmp;
            }
        }
        // broadcast to all threads in the warp
        unsigned total = __shfl_sync(FULL_MASK, presum, 31);
        // transform inclusive prefixSum to exclusive prefixSum
        presum = __shfl_up_sync(FULL_MASK, presum, 1);
        // NOTICE: for the first element, the original presum value is copied
        if (lane_id == 0) {
            presum = 0;
        }
        // write to corresponding position
        if (pred == 1) {
            d_neighbor_list[write_pos+presum] = data_neighbor;
        }
        write_pos += total;
    }
    presum = pred = 0;
    if (lane_id < degree) {
        data_neighbor = d_data_edges[begin+base+lane_id];
        pred = d_candidate_set[neighbor][data_neighbor];
        presum = pred;
    }

    // prefix sum in a warp
    for (unsigned stride = 1; stride < 32; stride <<= 1) {
        unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
        if (lane_id >= stride) {
            presum += tmp;
        }
    }
    // transform inclusive prefixSum to exclusive prefixSum
    presum = __shfl_up_sync(FULL_MASK, presum, 1);
    // NOTICE: for the first element, the original presum value is copied
    if (lane_id == 0) {
        presum = 0;
    }
    // write to corresponding position
    // NOTICE: warp divergence exists(even we use compact, the divergence also exists in the compact operation)
    if (pred == 1) {
        d_neighbor_list[write_pos+presum] = data_neighbor;
    }
}

__global__ void precompute_kernel(Vertex* d_embedding, Vertex* edge_first_vertex, Edge* edge_offset, Vertex* edge_second_vertex,
	long long unsigned* d_count, long long unsigned num_embeddings, Vertex len_embeddings, Vertex pivot_idx, Vertex num_candidates) {
	long long unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long unsigned group_id = idx / 32;
    Vertex lane_id = idx % 32;
    if (group_id >= num_embeddings) {
        return; 
    }

    Vertex* crt_embedding = d_embedding + group_id * len_embeddings;
    Vertex data_pivot = crt_embedding[pivot_idx];

    // list length
   	Vertex size = num_candidates;
    Vertex loop = size / 32;
    size = size % 32;

    unsigned is_find;
    unsigned is_find_warp = 0;
    Vertex i;
    for (i = 0; i < loop; i++) {
    	is_find = 0;
    	if (edge_first_vertex[i*32+lane_id] == data_pivot) {
    		is_find = 1;
    	}
    	is_find_warp = __any_sync(FULL_MASK, is_find);
    	if (is_find_warp) {
    		break;
    	}
    }
    if (!is_find_warp) {
    	if (lane_id < size) {
    		if (edge_first_vertex[i*32+lane_id] == data_pivot) {
    			is_find = 1;
    		}
    	}
    	is_find_warp = __any_sync(FULL_MASK, is_find);
    	if (!is_find_warp) {
    		d_count[group_id] = 0;
    		return;
    	}
    }

    if (is_find == 1) {
    	d_count[group_id] = edge_offset[i*32+lane_id+1] - edge_offset[i*32+lane_id];
    }
}

__device__ unsigned binary_search(unsigned _key, unsigned* _array, unsigned _array_num) {
	//MAYBE: return the right border, or use the left border and the right border as parameters
    if (_array_num == 0 || _array == NULL) {
		return 0;
    }

    unsigned _first = _array[0];
    unsigned _last = _array[_array_num - 1];

    if (_last == _key) {
        return 1;
    }

    if (_last < _key || _first > _key) {
		return 0;
    }

    unsigned low = 0;
    unsigned high = _array_num - 1;
    unsigned mid;
    while (low <= high) {
        mid = (high - low) / 2 + low; // (low + high) / 2
        if (_array[mid] == _key) {
            return 1;
        }
        if (_array[mid] > _key) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
	return 0;
}

__global__ void compute_kernel(Vertex* d_embedding, Vertex* d_new_partial_embeddings, Vertex** edge_first_vertex, 
	Edge** edge_offset, Vertex** edge_second_vertex, Vertex* num_candidates, Vertex* pivot_idxs, long long unsigned* d_count, 
	long long unsigned* d_new_count, long long unsigned num_embeddings, Vertex len_embeddings, Vertex num_neighbors) {
	extern __shared__ Vertex shared_space[];
	Vertex* list1 = shared_space; // store the first neighbor list; a warp occupies 32 elements
	Vertex* list2 = list1+blockDim.x; // store the second neighbor list;
	Vertex* intersection_cache = list2+blockDim.x; // store intersection result
	Vertex* count = intersection_cache+blockDim.x; //count the size after all intersection

	long long unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long unsigned group_id = idx / 32;
    Vertex warp_id = threadIdx.x / 32;
    Vertex lane_id = idx % 32;

    if (group_id >= num_embeddings) {
    	return;
    }

    // initialise shared memory
    list1[threadIdx.x] = 0;
    list2[threadIdx.x] = 0;
    intersection_cache[threadIdx.x] = 0;
    if (lane_id == 0) {
        count[warp_id] = 0;
    }
    __syncthreads();

    unsigned len_list1 = 0;
    unsigned len_list2 = 0;
    Vertex* crt_embedding = d_embedding + group_id * len_embeddings;
    long long unsigned write_pos = d_count[group_id];
    Vertex* g_list1 = NULL;
    Vertex* g_list2 = NULL;
    if (d_count[group_id] == d_count[group_id + 1]) {
		d_new_count[group_id] = 0;
    	return;
    }

    for (Vertex i = 0; i < num_neighbors; i++) {
    	// find compressed neighbor list
    	Vertex data_pivot = crt_embedding[pivot_idxs[i]];
    	unsigned read_pos = 0;

    	Vertex size = num_candidates[i];
    	Vertex loop = size / 32;
    	size = size % 32;

    	unsigned is_find = 0;
    	unsigned is_find_warp = 0;
    	Vertex j;
    	for (j = 0; j < loop; j++) {
    		is_find = 0;
    		if (edge_first_vertex[i][j*32+lane_id] == data_pivot) {
    			is_find = 1;
    		}
    		is_find_warp = __any_sync(FULL_MASK, is_find);
    		if (is_find_warp) {
    			break;
    		}
    	}
    	if (!is_find_warp) {
    		if (lane_id < size) {
    			if (edge_first_vertex[i][j*32+lane_id] == data_pivot) {
    				is_find = 1;
    			}
    		}
    		is_find_warp = __any_sync(FULL_MASK, is_find);
    		if (!is_find_warp) {
    			d_new_count[group_id] = 0;
    			return;
    		}
    	}
    	// record reading position
    	if (is_find == 1) {
    		list1[warp_id*32] = edge_offset[i][j*32+lane_id];
    		list1[warp_id*32+1] = edge_offset[i][j*32+lane_id+1] - list1[warp_id*32];
    	}

    	__syncthreads();

  		// sychronize reading position
    	read_pos = list1[warp_id*32];
    	// list1 is always intermediate intersection result stored into d_new_partial_embeddings
    	// if i = 0, d_new_partial_embeddings has not been initialized 
    	if (i == 0) {
    		// initialise d_new_partial_embeddings
    		size = list1[warp_id*32+1];
    		loop = size / 32;
    		size = size % 32;

    		for (Vertex j = 0; j < loop; j++) {
    			d_new_partial_embeddings[write_pos+j*32+lane_id] = edge_second_vertex[i][read_pos+j*32+lane_id];
    		}
    		if (lane_id < size) {
    			d_new_partial_embeddings[write_pos+loop*32+lane_id] = edge_second_vertex[i][read_pos+loop*32+lane_id];
    		}
    		count[warp_id] = list1[warp_id*32+1];
    	} else { // do intersection
    		// get list, list1 is intersection result, list2 is a new neighbor list to be intersected
    		g_list1 = d_new_partial_embeddings + write_pos;
    		g_list2 = edge_second_vertex[i] + read_pos;

    		// list length, not be 0
    		len_list1 = count[warp_id];
    		len_list2 = list1[warp_id*32+1];
    		count[warp_id] = 0; // reset to record the length of intersection cache

    		unsigned pos1 = 0, pos2 = 0;
    		int choice = 0;
    		unsigned bgroup = warp_id*32;
    		unsigned new_count = 0;
    		while(pos1 < len_list1 && pos2 < len_list2) {
        		if(choice <= 0) { // load list1
            		list1[bgroup + lane_id] = INVALID;
            		if (pos1 + lane_id < len_list1) {
            			list1[bgroup + lane_id] = g_list1[pos1 + lane_id];
            		}
        		}
        		if (choice >= 0) { // load list2
            		list2[bgroup + lane_id] = INVALID;
            		if (pos2 + lane_id < len_list2) {
                		list2[bgroup + lane_id] = g_list2[pos2 + lane_id];
                	}
        		}
        		is_find = 0;  //some threads may fail in the judgement below
        		unsigned is_full1 = (pos1 + 32 < len_list1) ? 32 : (len_list1 - pos1);
        		unsigned is_full2 = (pos2 + 32 < len_list2) ? 32 : (len_list2 - pos2);
        		unsigned presum = 0;
        		if (pos1 + lane_id < len_list1) {
        			is_find = binary_search(list1[bgroup + lane_id], list2 + bgroup, is_full2);
        		}
        		// compute prefix sum
        		presum = is_find;
        		for (unsigned stride = 1; stride < 32; stride <<= 1) {
            		unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
            		if (lane_id >= stride) {
                		presum += tmp;
            		}
        		}
        		unsigned total = __shfl_sync(FULL_MASK, presum, 31);  //broadcast to all threads in the warp
        		presum = __shfl_up_sync(FULL_MASK, presum, 1);
        		if (lane_id == 0) {
            		presum = 0;
        		}
        		if (is_find == 1) {
        			// load into cache
            		if(count[warp_id] + presum < 32) {
                		intersection_cache[bgroup + count[warp_id] + presum] = list1[bgroup + lane_id];
            		}
        		}
        		// cache area overflow
        		if (count[warp_id] + total >= 32) {
        			// clear cache area
            		d_new_partial_embeddings[write_pos+new_count+lane_id] = intersection_cache[bgroup+lane_id];
            		new_count += 32;
            		// write overflow part into cache area
            		if (is_find == 1) {
                		unsigned pos = count[warp_id] + presum;
                		if (pos >= 32) {
                   			intersection_cache[bgroup + pos - 32] = list1[bgroup + lane_id];
                		}
            		}
            		count[warp_id] = count[warp_id] + total - 32;
        		}
        		else {
            		count[warp_id] += total;
        		}
        		// set the next movement
        		choice = list1[bgroup + is_full1 - 1] - list2[bgroup + is_full2 - 1];
        		if (choice <= 0) {
            		pos1 += 32;
        		}
        		if (choice >= 0) {
            		pos2 += 32;
        		}
    		}
    		// flush residual cache area
    		if (lane_id < count[warp_id]) {
        		d_new_partial_embeddings[write_pos+new_count+lane_id] = intersection_cache[bgroup+lane_id];
    		}
    		new_count += count[warp_id];
    		count[warp_id] = new_count;
    	}
    }

    // check isomorphism
    Vertex size = count[warp_id];
    Vertex loop = size / 32;
    size = size % 32;
    unsigned new_count = 0;
    for (Vertex i = 0; i < loop; i++) {
    	unsigned is_valid = 1;
    	Vertex crt_candidate = d_new_partial_embeddings[write_pos+i*32+lane_id];
    	for (Vertex j = 0; j < len_embeddings; j++) {
    		if (crt_candidate == crt_embedding[j]) {
    			is_valid = 0;
    			break;
    		}
    	}
    	unsigned presum = is_valid;
        for (unsigned stride = 1; stride < 32; stride <<= 1) {
           	unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
           	if (lane_id >= stride) {
              	presum += tmp;
           	}
        }
        unsigned total = __shfl_sync(FULL_MASK, presum, 31);  //broadcast to all threads in the warp
        presum = __shfl_up_sync(FULL_MASK, presum, 1);
        if (lane_id == 0) {
           	presum = 0;
        }
        if (is_valid) {
        	d_new_partial_embeddings[write_pos+new_count+presum] = crt_candidate;
        }
        new_count += total;
    }
    unsigned is_valid = 1;
    Vertex crt_candidate;
    if (lane_id < size) {
    	crt_candidate = d_new_partial_embeddings[write_pos+loop*32+lane_id];
    	for (Vertex j = 0; j < len_embeddings; j++) {
    		if (crt_candidate == crt_embedding[j]) {
    			is_valid = 0;
    			break;
    		}
    	}
    } else {
    	is_valid = 0;
    }

    __syncthreads();

    unsigned presum = is_valid;
    for (unsigned stride = 1; stride < 32; stride <<= 1) {
       	unsigned tmp = __shfl_up_sync(FULL_MASK, presum, stride);
       	if (lane_id >= stride) {
          	presum += tmp;
       	}
    }
    unsigned total = __shfl_sync(FULL_MASK, presum, 31);  //broadcast to all threads in the warp
    presum = __shfl_up_sync(FULL_MASK, presum, 1);
    if (lane_id == 0) {
       	presum = 0;
    }
    if (is_valid) {
    	d_new_partial_embeddings[write_pos+new_count+presum] = crt_candidate;
    }

    new_count += total;
    if (lane_id == 0) {
        d_new_count[group_id] = new_count;
    }
}

__global__ void extend_kernel(Vertex* d_embedding, Vertex* d_new_partial_embeddings, Vertex* d_new_embedding, long long unsigned* d_count, 
	long long unsigned* d_new_count, long long unsigned num_embeddings, Vertex len_embeddings) {
    __shared__ Vertex cached_embedding[1024];
    __shared__ long long unsigned task_pos[32];

    long long unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long unsigned lane_id = idx & 0x1f;
    long long unsigned group_id = idx >> 5;
    long long unsigned block_group = threadIdx.x >> 5;
    long long unsigned block_presum = threadIdx.x & 0xffffffe0;

    long long unsigned read_pos = 0, write_pos = 0, size = 0;
    if (group_id >= num_embeddings) {
        return;
    }
    
    read_pos = d_count[group_id];
    write_pos = d_new_count[group_id];
    size = d_new_count[group_id + 1] - write_pos;
    write_pos *= (len_embeddings + 1);

    if (lane_id == 0) {
        if (size > 0) {
            memcpy(cached_embedding + block_presum, d_embedding + group_id * len_embeddings, sizeof(Vertex) * len_embeddings);
        }
    }
    __syncthreads();

    long long unsigned crt_pos = 0;
    Vertex* crt_embedding = cached_embedding + block_presum;
    
    // process big task
    while (true) {
        if (threadIdx.x == 0) {
            task_pos[0] = INVALID;
        }
        __syncthreads();

        if (size >= crt_pos + 1024) {
            task_pos[0] = block_group; // task block
        }
        __syncthreads();

        if (task_pos[0] == INVALID) {
            break;
        }

        Vertex* target_embedding = cached_embedding + 32 * task_pos[0];
        if (task_pos[0] == block_group) { // share task parameters
            task_pos[1] = read_pos;
            task_pos[2] = write_pos;
            task_pos[3] = crt_pos;
            task_pos[4] = size;
        }
        __syncthreads();

        while (task_pos[3] + 1023 < task_pos[4]) {
            long long unsigned pos = (len_embeddings + 1) * (task_pos[3] + threadIdx.x); // thread pos
            memcpy(d_new_embedding + task_pos[2] + pos, target_embedding, sizeof(Vertex) * len_embeddings); // copy old embedding
            d_new_embedding[task_pos[2] + pos + len_embeddings] = d_new_partial_embeddings[task_pos[1] + task_pos[3] + threadIdx.x]; // copy extended data vertex

            if (threadIdx.x == 0) {
                task_pos[3] += 1024;
            }
            __syncthreads();
        }
        if (task_pos[0] == block_group) {
            crt_pos = task_pos[3];
        }
        __syncthreads();
    }
    __syncthreads();

    // process own task
    while (crt_pos < size) {
        if (crt_pos + lane_id < size) {
            long long unsigned pos = (len_embeddings + 1) * (crt_pos + lane_id);
            memcpy(d_new_embedding + write_pos + pos, crt_embedding, sizeof(Vertex) * len_embeddings);
            d_new_embedding[write_pos + pos + len_embeddings] = d_new_partial_embeddings[read_pos + crt_pos + lane_id];
        }
        crt_pos += 32;
    }
}

__global__ void split_last_embeddings_kernel(long long unsigned* d_count, long long unsigned* d_pivots, Vertex num_pivots, long long unsigned* d_intervals, 
    long long unsigned* d_prefix_sum, Vertex num_result_rows) {
    long long unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_result_rows) {
        return;
    }

    long long unsigned crt_prefix_sum = d_count[idx];
    long long unsigned next_prefix_sum = d_count[idx + 1];
    for (Vertex i = 0; i < num_pivots; i++) {
        if (crt_prefix_sum <= d_pivots[i] && next_prefix_sum > d_pivots[i]) {
            d_intervals[i] = idx;
            d_prefix_sum[i] = crt_prefix_sum;
        }
    }
}

__global__ void rewrite_count_kernel(long long unsigned* d_count, Vertex num_result_rows, long long unsigned* d_prefix_sum, Vertex interval_id) {
    Vertex idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_result_rows) {
        return;
    }

    d_count[idx] = d_count[idx] - d_prefix_sum[interval_id];
}

#endif