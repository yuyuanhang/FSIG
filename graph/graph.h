/**
 * @file graph.h
 * @brief contains an abstract data class Graph and its child classes
 * @details child classes specify the format of graph, i.e., the way that graph is stored
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "../utility/defines.h"

using std::vector;
using std::cout;
using std::endl;
using std::find;

/*!
* An abstract data class
*/
class Graph {
public:
    Vertex num_vertices_;
    Edge num_edges_;
    Label num_labels_;
    bool is_directed_ = false;
    bool is_labeled_ = true;
    
    Graph() { num_vertices_ = 0; num_edges_ = 0; num_labels_ = 0; }
    virtual ~Graph() = default;
    Vertex num_vertices() const { return num_vertices_; }
    Edge num_edges() const { return num_edges_; }
    Label num_labels() const { return num_labels_; }
    bool is_directed() const { return is_directed_; }
    void set_num_vertices(Vertex num_vertices) { num_vertices_ = num_vertices; }
    void set_num_edges(Edge num_edges) { num_edges_ = num_edges; }
    void set_num_labels(Label num_labels) { num_labels_ = num_labels; }
    void set_is_directed(bool is_directed) { is_directed_ = is_directed; }
};

/*!
* A dynamic graph class that provides interfaces to add vertices and edges
*/
class UndirectedPlainGraph : public Graph {
public:
    vector<Label> labels_;
    vector<vector<Vertex>> edges_;
    Vertex min_degree;

    UndirectedPlainGraph() { set_is_directed(false); }
    virtual ~UndirectedPlainGraph() = default;
    void add_vertex(Vertex vertex_id, Label vertex_label);
    void add_edge(Vertex first_vertex, Vertex second_vertex);
    Vertex degree(Vertex vertex_id) { return edges_[vertex_id].size(); }
    vector<Vertex> neighbors(Vertex vertex_id) { return edges_[vertex_id]; }
    Edge vertices_to_edge_id(Vertex first_vertex, Vertex second_vertex);
    bool check_edge_existence(Vertex first_vertex, Vertex second_vertex);
    void set_min_degree();
    
#ifdef DEBUG
    void print_graph();
#endif
};

/*!
* A static graph class that provides interfaces to access vertices and edges
*/
class UndirectedCSRGraph : public Graph {
public:
    Label* labels_;
    Edge* offset_;
    Vertex* edges_;
    Vertex min_degree;

    UndirectedCSRGraph() { set_is_directed(false); }
    ~UndirectedCSRGraph() { delete[] labels_; delete[] offset_; delete[] edges_; }
    Vertex degree(Vertex vertex_id) { return offset_[vertex_id+1] - offset_[vertex_id]; }
    Vertex* neighbors(Vertex vertex_id) { return edges_ + offset_[vertex_id]; }
};

/*!
* A static graph class that provides interfaces to access vertices and edges
*/
struct PCSRUnit {
    Edge* offset_ = NULL;  // the size is 32 * key_num
    Vertex* edges_ = NULL;
    Vertex num_keys_; // also the group number
    Vertex num_edges_;
};

class UndirectedPCSRGraph : public Graph {
public:
    Label* labels_;
    PCSRUnit** units_;
    Vertex num_units_;

    UndirectedPCSRGraph() {
        set_is_directed(false);
    }
    ~UndirectedPCSRGraph() {};
    static Label get_edge_label(Label first_label, Label second_label) {
        Label larger_label;
        Label less_label;
        if (first_label >= second_label) {
            larger_label = first_label;
            less_label = second_label;
        } else {
            larger_label = second_label;
            less_label = first_label;
        }
        if (larger_label == 0) {
            return less_label;
        } else {
            return ((larger_label - 1) * larger_label) / 2 + less_label;
        }
    }
};

#endif