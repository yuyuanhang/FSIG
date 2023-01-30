#include "GSI.h"
#include "GSI_kernels.cuh"
#include <cub/cub.cuh>

#ifdef DEBUG
void print_priority_queue(priority_queue<ScoredVertex, vector<ScoredVertex>, greater<ScoredVertex>> pq) {
    for (; !pq.empty(); pq.pop()) {
        ScoredVertex const& s = pq.top();
        cout << "{ " << s.vertex_id_ << ", " << s.score_ << " } ";
    }
    cout << endl;
}
#endif

void GSI::exclusive_sum(unsigned* d_array, unsigned* sum_array, Vertex size, unsigned &sum, cudaStream_t stream) {
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, sum_array, size, stream);
    checkCudaErrors(cudaPeekAtLastError());

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    checkCudaErrors(cudaPeekAtLastError());

    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_array, sum_array, size, stream);
    checkCudaErrors(cudaPeekAtLastError());
    cudaFree(d_temp_storage);

    cudaMemcpy(&sum, sum_array + size - 1, sizeof(unsigned), cudaMemcpyDeviceToHost);
}

bool GSI::signature_filter() {
	ptrs_d_candidate_array_ = new Vertex*[query_size_];
    size_candidate_sets_ = new Vertex[query_size_];
    string description;

    Vertex* d_data_signature_table = NULL;
    description = string("signature table of data graph");
    if (device_memory_manager_->allocate_memory(d_data_signature_table, data_size_ * SIGNUM * sizeof(Vertex), description)) {
    	cudaMemcpy(d_data_signature_table, data_signature_table_, data_size_ * SIGNUM * sizeof(Vertex), cudaMemcpyHostToDevice);
    } else {
    	exit(0);
    }

    unsigned* d_bitmap = NULL;
    description = string("bitmap of data vertices");
    if (device_memory_manager_->allocate_memory(d_bitmap, sizeof(unsigned) * (data_size_ + 1), description)) {
    	cudaMemset(d_bitmap, 0, sizeof(unsigned) * (data_size_ + 1));
    } else {
    	exit(0);
    }

    int BLOCK_SIZE = 1024;
	int GRID_SIZE = (data_size_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (Vertex i = 0; i < query_size_; i++) {
        // load the signature of query graph
        cudaMemcpyToSymbol(c_query_signature_table, query_signature_table_ + i * SIGNUM, sizeof(Vertex)*SIGNUM);

        filter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_data_signature_table, d_bitmap, data_size_);
        cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());
    	
    	Vertex size_candidate_set = 0;
    	exclusive_sum(d_bitmap, d_bitmap, data_size_ + 1, size_candidate_set);
		checkCudaErrors(cudaGetLastError());

        unsigned* d_candidate_array = NULL;
        description = string("candidate array ") + to_string(i);
        if (device_memory_manager_->allocate_memory(d_candidate_array, sizeof(unsigned)*size_candidate_set, description)) {
    		cudaMemset(d_candidate_array, 0, sizeof(unsigned)*size_candidate_set);
    	} else {
    		exit(0);
    	}
        scatter_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_bitmap, d_candidate_array, data_size_);
        cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

        ptrs_d_candidate_array_[i] = d_candidate_array;
        size_candidate_sets_[i] = size_candidate_set;
#ifdef DEBUG
		cout << "query vertex " << i << ": " << size_candidate_set << endl;
#endif
    }
#ifdef DEBUG
	double average = 0;
	for (Vertex i = 0; i < query_size_; i++) {
		average += size_candidate_sets_[i];
	}
	average = average / query_size_;
	cout << "average_cvs_size: " << average << endl;
#endif
    description = string("bitmap of data vertices");
    device_memory_manager_->free_memory(d_bitmap, sizeof(unsigned) * (data_size_ + 1), description);
    description = string("signature table of data graph");
	device_memory_manager_->free_memory(d_data_signature_table, data_size_ * SIGNUM * sizeof(Vertex), description);

	//get the num of candidates and compute scores
	//bool success = score_node(_score, _qnum);

	return true;
}

void GSI::generate_join_order() {
	bool* is_visited = new bool[query_size_];
	memset(is_visited, 0, sizeof(bool)*query_size_);

	vector<ScoredVertex> query_vertices(query_size_);
    for (Vertex i = 0; i < query_size_; i++) {
    	query_vertices[i] = ScoredVertex(i, (float)size_candidate_sets_[i]);
    }
    for (Vertex crt_pos = 0; crt_pos < query_size_; crt_pos++) {
    	priority_queue<ScoredVertex, vector<ScoredVertex>, greater<ScoredVertex>> pq;
    	if (crt_pos == 0) {
    		for (auto query_vertex : query_vertices) {
    			pq.push(query_vertex);
    		}
    	} else {
    		for (Vertex j = 0; j < crt_pos; j++) {
    			auto query_neighbors = plain_query_graph_->neighbors(pos_to_id_[j]);
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
    	pos_to_id_[crt_pos] = matched_vertex;
    	id_to_pos_[matched_vertex] = crt_pos;
    	is_visited[matched_vertex] = true;
    	// adjust query neighbors' priority
    	auto query_neighbors = plain_query_graph_->neighbors(pos_to_id_[crt_pos]);
    	for (auto query_neighbor : query_neighbors) {
    		query_vertices[query_neighbor].score_ *= 0.9;
    	}
    }
#ifdef DEBUG
	cout << "match order: ";
	for (Vertex i = 0; i < query_size_; i++) {
		cout << pos_to_id_[i] << " ";
	}
	cout << endl;
#endif
}
 
void GSI::get_neighbors_on_induced_subgraph(vector<Vertex> &induced_subgraph_neighbors, Vertex order) {
	induced_subgraph_neighbors.clear();
	Vertex crt_match_vertex = pos_to_id_[order];
	for (Vertex i = 0; i < order; i++) {
		Vertex pre_match_vertex = pos_to_id_[i];
		if (plain_query_graph_->check_edge_existence(crt_match_vertex, pre_match_vertex)) {
			induced_subgraph_neighbors.push_back(pre_match_vertex);
		}
	}
}

void GSI::join_embeddings() {
	// materialize initial partial embeddings
	Vertex start_vertex = pos_to_id_[0];
	Edge num_result_rows = size_candidate_sets_[start_vertex];
	Vertex num_result_cols = 1;
	d_embeddings_ = ptrs_d_candidate_array_[start_vertex];

	// a bitmap of candidate array. 
	// because we don't compute presum, it can be accessed by bit operations, not just by vertex id
	unsigned* d_bitmap = NULL;
	unsigned num_unsigned = (data_size_ + 31) / 32; // a unsigned type occupies 32 bits
	if (device_memory_manager_->allocate_memory(d_bitmap, sizeof(unsigned)*num_unsigned, "bitmap of data vertices")) {
    	cudaMemset(d_bitmap, 0, sizeof(unsigned)*num_unsigned);
    } else {
    	exit(0);
    }

    cout << "#embeddings: " << num_result_rows << endl;
	// join one vertex at a time to match induced subgraph
	for (Vertex i = 1; i < query_size_; i++) {
		cout << "joining " << i << "th vertex" << endl;
		Vertex target_vertex = pos_to_id_[i];
		Label target_label = plain_query_graph_->labels_[target_vertex];

		vector<Vertex> induced_subgraph_neighbors;
		get_neighbors_on_induced_subgraph(induced_subgraph_neighbors, i);

		// convert candidates from array to bitmap
        int BLOCK_SIZE = 1024;
        int GRID_SIZE = (size_candidate_sets_[target_vertex]+BLOCK_SIZE-1)/BLOCK_SIZE;
        GSI_generate_bitmap_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_bitmap, ptrs_d_candidate_array_[target_vertex], size_candidate_sets_[target_vertex]);
        cudaDeviceSynchronize();
		checkCudaErrors(cudaGetLastError());

        device_memory_manager_->free_memory(ptrs_d_candidate_array_[target_vertex], 
        	sizeof(Vertex)*size_candidate_sets_[target_vertex], string("candidate array ") + to_string(target_vertex));

		Vertex* d_degree_bound = NULL;
		Vertex* d_filtered_degree = NULL;
		if (device_memory_manager_->allocate_memory(d_degree_bound, sizeof(Vertex)*(num_result_rows+1), "maximal number of new mappings")) {
    		cudaMemset(d_degree_bound, 0, sizeof(Vertex)*(num_result_rows+1));
    	} else {
    		exit(0);
    	}
		if (device_memory_manager_->allocate_memory(d_filtered_degree, sizeof(Vertex)*(num_result_rows+1), "the number of new mappings")) {
    		cudaMemset(d_filtered_degree, 0, sizeof(Vertex)*(num_result_rows+1));
    	} else {
    		exit(0);
    	}
		
		// load device pointers and variables of embeddings
		cudaMemcpyToSymbol(c_embeddings, &d_embeddings_, sizeof(Vertex*));
		cudaMemcpyToSymbol(c_bitmap, &d_bitmap, sizeof(unsigned*));
		cudaMemcpyToSymbol(c_num_result_rows, &num_result_rows, sizeof(Edge));
		cudaMemcpyToSymbol(c_num_result_cols, &num_result_cols, sizeof(Vertex));

		Edge num_new_result_rows = 0;
	    GRID_SIZE = (num_result_rows*32+BLOCK_SIZE-1)/BLOCK_SIZE;
	    Vertex* d_target_data_vertices = NULL;
	    // iterate on neighbors on induced subgraph
	    for (Vertex j = 0; j < induced_subgraph_neighbors.size(); j++) {
	    	cout << "intersecting " << j << "th vertex" << endl;
	    	Vertex induced_subgraph_neighbor = induced_subgraph_neighbors[j];
	        cudaMemcpyToSymbol(c_neighbor_pos, id_to_pos_ + induced_subgraph_neighbor, sizeof(Vertex));
	        Label induced_subgraph_neighbor_label = plain_query_graph_->labels_[induced_subgraph_neighbor];

	        Label edge_label = pcsr_data_graph_->get_edge_label(target_label, induced_subgraph_neighbor_label);
	        PCSRUnit* crt_unit = pcsr_data_graph_->units_[edge_label];
	        
	        Edge* d_offset = NULL;
	        Vertex* d_edges = NULL;
	        
	        if (device_memory_manager_->allocate_memory(d_offset, sizeof(Edge)*crt_unit->num_keys_*32, "offset array of one partitioned graph")) {
    			cudaMemcpy(d_offset, crt_unit->offset_, sizeof(Edge)*crt_unit->num_keys_*32, cudaMemcpyHostToDevice);
    		} else {
    			exit(0);
    		}
    		if (device_memory_manager_->allocate_memory(d_edges, sizeof(Vertex)*crt_unit->num_edges_, "edge array of one partitioned graph")) {
    			cudaMemcpy(d_edges, crt_unit->edges_, sizeof(Vertex)*crt_unit->num_edges_, cudaMemcpyHostToDevice);
    		} else {
    			exit(0);
    		}
    		// load device pointers and variables of partitioned graph
	        cudaMemcpyToSymbol(c_offset, &d_offset, sizeof(Edge*));
	        cudaMemcpyToSymbol(c_edges, &d_edges, sizeof(Vertex*));
	        cudaMemcpyToSymbol(c_num_keys, &(crt_unit->num_keys_), sizeof(Vertex));
	        cudaMemcpyToSymbol(c_degree_bound, &d_degree_bound, sizeof(Vertex*));

	        BLOCK_SIZE = 256;
	        GRID_SIZE = (num_result_rows*32+BLOCK_SIZE-1)/BLOCK_SIZE;
	        if (j == 0) {
	            preallocate_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_degree_bound);
	            cudaDeviceSynchronize();
	            checkCudaErrors(cudaGetLastError());

	            exclusive_sum(d_degree_bound, d_degree_bound, num_result_rows + 1, num_new_result_rows);

	            cout << "#embeddings bound: " << num_new_result_rows << endl;

	            if (!device_memory_manager_->allocate_memory(d_target_data_vertices, sizeof(Vertex) * num_new_result_rows, "initial intersection list")) {
    				exit(0);
    			}
	            load_target_data_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_target_data_vertices, d_filtered_degree);
	            cudaDeviceSynchronize();
	            checkCudaErrors(cudaGetLastError());
	        }
	        else {
	            intersection_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_target_data_vertices, d_filtered_degree);
	            cudaDeviceSynchronize();
	            checkCudaErrors(cudaGetLastError());
	        }

	        device_memory_manager_->free_memory(d_offset, sizeof(Edge)*crt_unit->num_keys_*32, "offset array of one partitioned graph");
	        device_memory_manager_->free_memory(d_edges, sizeof(Vertex)*crt_unit->num_edges_, "edge array of one partitioned graph");
	    }

		Edge num_new_filtered_result_rows = 0;
	    exclusive_sum(d_filtered_degree, d_filtered_degree, num_result_rows + 1, num_new_filtered_result_rows);
	    cout << "#embeddings: " << num_new_filtered_result_rows << endl;
	    if (num_new_filtered_result_rows == 0) {
	    	exit(0);
	    }
	
		unsigned* d_new_embeddings = NULL; 
		if (!device_memory_manager_->allocate_memory(d_new_embeddings, sizeof(Vertex)*num_new_filtered_result_rows*(num_result_cols+1), "embeddings of induced subgraph " + to_string(i))) {
    		exit(0);
    	}

		link_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_target_data_vertices, d_degree_bound, d_filtered_degree, d_new_embeddings);
		checkCudaErrors(cudaGetLastError());
		cudaDeviceSynchronize();

	    if (i == 1) {
	    	device_memory_manager_->free_memory(d_embeddings_, sizeof(Vertex) * num_result_rows * num_result_cols, "candidate array " + to_string(pos_to_id_[0]));
	    } else {
	    	device_memory_manager_->free_memory(d_embeddings_, sizeof(Vertex) * num_result_rows * num_result_cols, "embeddings of induced subgraph " + to_string(i - 1));
	    }
		d_embeddings_ = d_new_embeddings;
	
		device_memory_manager_->free_memory(d_target_data_vertices, sizeof(Vertex) * num_new_result_rows, "initial intersection list");
		device_memory_manager_->free_memory(d_degree_bound, sizeof(Vertex)*(num_result_rows+1), "maximal number of new mappings");
		device_memory_manager_->free_memory(d_filtered_degree, sizeof(Vertex)*(num_result_rows+1), "the number of new mappings");
		
		num_result_cols++;
		num_result_rows = num_new_filtered_result_rows;
	}
	device_memory_manager_->free_memory(d_bitmap, sizeof(unsigned)*num_unsigned, "bitmap of data vertices");

#ifdef DEBUG
	Vertex sum = num_result_rows;
    Vertex* h_embeddings = new Vertex[sum*query_size_];
    cudaMemcpy(h_embeddings, d_embeddings_, sum*query_size_*sizeof(Vertex), cudaMemcpyDeviceToHost);
    for (Vertex i = 0; i < 5; i++) {
        for (Vertex j = 0; j < query_size_; j++) {
            cout << h_embeddings[i*query_size_+j] << " ";
        }
        cout << endl;
    }
    for (Vertex i = sum - 5; i < sum; i++) {
        for (Vertex j = 0; j < query_size_; j++) {
            cout << h_embeddings[i*query_size_+j] << " ";
        }
        cout << endl;
    }
#endif
}

void GSI::match() {
	init_GPU(dev_id_);

	time_recorder_->event_start("total time");

	cout << "-----     filtering candidate vertices    -----" << endl;
    time_recorder_->event_start("filter data vertices");
	bool success = signature_filter();
    time_recorder_->event_end("filter data vertices");

	//initialize the mapping structure
	id_to_pos_ = new Vertex[query_size_];
    pos_to_id_ = new Vertex[query_size_];
	memset(id_to_pos_, INVALID, sizeof(Vertex)*query_size_);
	memset(pos_to_id_, INVALID, sizeof(Vertex)*query_size_);
	cout << "-----     generating join order    -----" << endl;
	time_recorder_->event_start("generate join order");
	generate_join_order();
	time_recorder_->event_end("generate join order");

	cout << "-----     joining embeddings    -----" << endl;
	time_recorder_->event_start("join embeddings");
	join_embeddings();
	time_recorder_->event_end("join embeddings");

	time_recorder_->event_end("total time");

	time_recorder_->print_all_events();
	device_memory_manager_->print_memory_cost();
}