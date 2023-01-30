#ifndef MEMORY_MANAGER_H
#define MEMORY_MANAGER_H

#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cuda_runtime.h>

using std::string;
using std::vector;
using std::pair;
using std::cout;
using std::endl;

class MemoryManager
{
public:
	long long unsigned total_memory_size_ = 0;
	long long unsigned available_memory_size_ = 0;
	long long unsigned memory_size_for_enumeration_ = 0;
	vector<pair<string, long long unsigned>> log_;
	long long unsigned used_memory_peak_ = 0;

	MemoryManager(long long unsigned total_memory_size) : total_memory_size_(total_memory_size), available_memory_size_(total_memory_size) {
		
	}
	~MemoryManager() {}
	bool allocate_memory(unsigned* &ptr, long long unsigned size, string description) {
		// cout << "allocate memory " << description << " " << (double)size/(1024*1024) << "MB" << endl;
		if (available_memory_size_ < size) {
			cout << "remaining memory: " << (double)available_memory_size_/(1024*1024) << "MB" << endl;
			cout << "required memory: " << (double)size/(1024*1024) << "MB" << endl;
			cout << "out of memory on device" << endl;
			print_memory_cost();
			if (!check_memory()) {
				exit(0);
			}
			return false;
		} else {
			if (cudaMalloc(&ptr, size) == cudaSuccess) {
				available_memory_size_ -= size;
			    log_.push_back(make_pair(description, size));
			    if (total_memory_size_ - available_memory_size_ > used_memory_peak_) {
			    	used_memory_peak_ = total_memory_size_ - available_memory_size_;
			    }
			    if (!check_memory()) {
			    	print_memory_cost();
					exit(0);
				}
			    return true;
			} else {
				cout << available_memory_size_ << " " << size << endl;
				size_t free = 0;
				size_t total = 0;
				cudaMemGetInfo(&free, &total);
				cout << free << " " << total << endl;
				print_memory_cost();
				checkCudaErrors(cudaMalloc(&ptr, size)); // memory fragmentation?
				return false;
			}
		}
	}
	bool allocate_memory(long long unsigned* &ptr, long long unsigned size, string description) {
		// cout << "allocate memory " << description << " " << (double)size/(1024*1024) << "MB" << endl;
		if (available_memory_size_ < size) {
			cout << "remaining memory: " << (double)available_memory_size_/(1024*1024) << "MB" << endl;
			cout << "required memory: " << (double)size/(1024*1024) << "MB" << endl;
			cout << "out of memory on device" << endl;
			print_memory_cost();
			if (!check_memory()) {
				exit(0);
			}
			return false;
		} else {
			if (cudaMalloc(&ptr, size) == cudaSuccess) {
				available_memory_size_ -= size;
			    log_.push_back(make_pair(description, size));
			    if (total_memory_size_ - available_memory_size_ > used_memory_peak_) {
			    	used_memory_peak_ = total_memory_size_ - available_memory_size_;
			    }
			    if (!check_memory()) {
			    	print_memory_cost();
					exit(0);
				}
			    return true;
			} else {
				cout << available_memory_size_ << " " << size << endl;
				size_t free = 0;
				size_t total = 0;
				cudaMemGetInfo(&free, &total);
				cout << free << " " << total << endl;
				print_memory_cost();
				checkCudaErrors(cudaMalloc(&ptr, size));
				return false;
			}
		}
	}
	bool allocate_memory(int* &ptr, long long unsigned size, string description) {
		if (available_memory_size_ < size) {
			cout << "out of memory on device" << endl;
			print_memory_cost();
			if (!check_memory()) {
				exit(0);
			}
			return false;
		} else {
			if (cudaMalloc(&ptr, size) == cudaSuccess) {
				available_memory_size_ -= size;
			    log_.push_back(make_pair(description, size));
			    if (total_memory_size_ - available_memory_size_ > used_memory_peak_) {
			    	used_memory_peak_ = total_memory_size_ - available_memory_size_;
			    }
			    if (!check_memory()) {
			    	print_memory_cost();
					exit(0);
				}
			    return true;
			} else {
				cout << available_memory_size_ << " " << size << endl;
				size_t free = 0;
				size_t total = 0;
				cudaMemGetInfo(&free, &total);
				cout << free << " " << total << endl;
				print_memory_cost();
				checkCudaErrors(cudaMalloc(&ptr, size));
				return false;
			}
		}
	}
	bool free_memory(unsigned* ptr, long long unsigned size, string description) {
		// cout << "free memory " << description << " " << (double)size/(1024*1024) << "MB" << endl;
		if (cudaFree(ptr) == cudaSuccess) {
			bool is_find = false;
			for (int i = 0; i < log_.size(); i++) {
				if (log_[i].first == description) {
					available_memory_size_ += log_[i].second;
					log_.erase(log_.begin() + i);
					is_find = true;
					break;
				}
			}
			if (!check_memory()) {
			   	print_memory_cost();
				exit(0);
			}
			if (is_find) {
				return true;
			} else {
				cout << "cannot find pointer in log" << endl;
				return false;
			}
		} else {
			cout << "cuda function call fails" << endl;
			return false;
		}
	}
	bool free_memory(long long unsigned* ptr, long long unsigned size, string description) {
		// cout << "free memory " << description << " " << (double)size/(1024*1024) << "MB" << endl;
		if (cudaFree(ptr) == cudaSuccess) {
			bool is_find = false;
			for (int i = 0; i < log_.size(); i++) {
				if (log_[i].first == description) {
					available_memory_size_ += log_[i].second;
					log_.erase(log_.begin() + i);
					is_find = true;
					break;
				}
			}
			if (!check_memory()) {
			   	print_memory_cost();
				exit(0);
			}
			if (is_find) {
				return true;
			} else {
				cout << "cannot find pointer in log" << endl;
				return false;
			}
		} else {
			cout << "cuda function call fails" << endl;
			return false;
		}
	}
	bool free_memory(unsigned* ptr, string description) {
		// cout << "free memory " << description << endl;
		if (cudaFree(ptr) == cudaSuccess) {
			bool is_find = false;
			for (int i = 0; i < log_.size(); i++) {
				if (log_[i].first == description) {
					available_memory_size_ += log_[i].second;
					log_.erase(log_.begin() + i);
					is_find = true;
					break;
				}
			}
			if (!check_memory()) {
			   	print_memory_cost();
				exit(0);
			}
			if (is_find) {
				return true;
			} else {
				cout << "cannot find pointer in log" << endl;
				return false;
			}
		} else {
			cout << "cuda function call fails" << endl;
			return false;
		}
	}
	template<class T>
	void copy_ptrs_to_device(T** &d_ptrs, T** h_ptrs, int num_ptrs, string description) {
		cudaMalloc(&d_ptrs, sizeof(T*)*num_ptrs);
		available_memory_size_ -= sizeof(T*)*num_ptrs;
	    log_.push_back(make_pair(description, sizeof(T*)*num_ptrs));
		cudaMemcpy(d_ptrs, h_ptrs, sizeof(T*)*num_ptrs, cudaMemcpyHostToDevice);
	}
	void record_memory_pool() { memory_size_for_enumeration_ = available_memory_size_; }
	void form_blocks(long long unsigned num_embedding, unsigned len_embedding, unsigned &block_num, long long unsigned &block_size) {
		// In the computation, d_partial_embedding is extended to form d_new_partial_embedding. the required space includes:
		// last layer: d_partial_embedding, d_count 
		// intermediate variables: d_new_count, d_pre_allocated_neighbor_list
		// next_layer: d_new_partial_embedding, d_count
		// set the number of rows of d_partial_embedding as n_{i}, the number of columns of d_partial_embedding as l_{i}
		// set the number of rows of d_new_partial_embedding as n_{i+1}, the number of columns of d_partial_embedding as l_{i+1}
		// the bound of required memory cost is n_{i}*l_{i} + n_{i} + n_{i} + n_{i+1}*l_{i+1} + n_{i+1} + n_{i+1} (we ignore the extra one element in prefix sum array)
		// here, an extra memory usage is the splitted d_count array
		// also, an extra memory cost is used in exclusive_sum function
		// hence, each layer requires n_{i}*(l_{i}+2) spaces.
		// we divide the space evenly to form a pipeline
		long long unsigned limited_space = memory_size_for_enumeration_ / 2;
		long long unsigned required_space = num_embedding * (len_embedding * sizeof(unsigned) + 3 * sizeof(long long unsigned));
		// cout << limited_space << " " << required_space << " " << num_embedding << " " << len_embedding << " " << sizeof(long long unsigned) << endl;
		if (required_space > limited_space) { // subblocks are formed
			block_size = limited_space / (len_embedding * sizeof(unsigned) + 3 * sizeof(long long unsigned)); // the number of embeddings that can be accommodated in a subblock
			block_num = (num_embedding + block_size - 1) / block_size; // the number of subblocks
		} else {
			block_num = 1;
			block_size = num_embedding;
		}
	}
	void print_memory_cost() {
		cout << "available memory: " << (double)total_memory_size_/(1024*1024*1024) << "GB" << endl;
		cout << "allocated memory for enumeration: " << (double)memory_size_for_enumeration_/2/(1024*1024*1024) << "GB" << endl;
		cout << "memory cost: " << (double)used_memory_peak_/(1024*1024) << "MB" << endl;
		for (auto log : log_) {
			double cost = (double)log.second/(1024*1024);
			if (cost >= 1) {
				cout << log.first << ": " << (double)log.second/(1024*1024) << "MB" << endl;
			}
		}
	}
	bool check_memory() {
		long long unsigned used_memory = 0;
		for (auto log : log_) {
			used_memory += log.second;
		}
		if (used_memory + available_memory_size_ == total_memory_size_) {
			return true;
		} else {
			return false;
		}
	}
};

#endif