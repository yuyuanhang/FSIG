/**
 * @file graph.cc
 * @brief contains an abstract data class graph and its child classes
 * @details child classes specify the format of Graph, i.e., the way that graph is stored
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#include "graph.h"

void UndirectedPlainGraph::add_vertex(Vertex vertex_id, Label vertex_label) {
	if (labels_.size() <= vertex_id) {
		labels_.resize(vertex_id+1, INVALID_LABEL);
		set_num_vertices(vertex_id+1);
	}
	if (num_labels() <= vertex_label) {
		set_num_labels(vertex_label+1);
	}
	labels_[vertex_id] = vertex_label;
}

void UndirectedPlainGraph::add_edge(Vertex first_vertex, Vertex second_vertex) {
	if (edges_.size() <= first_vertex) {
		edges_.resize(first_vertex+1);
	}
	if (edges_.size() <= second_vertex) {
		edges_.resize(second_vertex+1);
	}
	edges_[first_vertex].push_back(second_vertex);
	edges_[second_vertex].push_back(first_vertex);
	set_num_edges(num_edges()+1);
}

Edge UndirectedPlainGraph::vertices_to_edge_id(Vertex first_vertex, Vertex second_vertex) {
    Edge sum = 0;

	for (Vertex i = 0; i < first_vertex; i++) {
    	sum += edges_[i].size();
	}

    vector<Vertex> neighbors = this->neighbors(first_vertex);
    for (Vertex i = 0; i < neighbors.size(); i++) {
        Vertex neighbor = neighbors[i];
        if (neighbor == second_vertex) {
            return sum + i;
        }
    }
}

bool UndirectedPlainGraph::check_edge_existence(Vertex first_vertex, Vertex second_vertex) {
	vector<Vertex> neighbors = this->neighbors(first_vertex);
	if (find(neighbors.begin(), neighbors.end(), second_vertex) == neighbors.end()) {
        return false;
    } else {
    	return true;
    }
}

void UndirectedPlainGraph::set_min_degree() {
	min_degree = num_vertices();
	for (Vertex i = 0; i < num_vertices(); i++) {
		if (edges_[i].size() < min_degree) {
			min_degree = edges_[i].size();
		}
	}
}

#ifdef DEBUG
void UndirectedPlainGraph::print_graph() {
	cout << "|V|: " << num_vertices() << endl;
	cout << "|E|: " << num_edges() << endl;
	cout << "|L|: " << num_labels() << endl;
}
#endif
