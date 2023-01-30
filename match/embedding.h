/**
 * @file embedding.h
 * @brief contains an data class embedding
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include "../utility/defines.h"
#include "../utility/inverted_index.h"

/*!
* A static data class
*/
class Embedding {
public:
    Vertex num_vertices_;
    Edge num_instances_;
    Vertex* vertices_order_;
    Vertex* instances_;
    InversedFunction<Vertex>* inversed_embedding_;

    Embedding(Vertex num_vertices) {
        num_vertices_ = num_vertices;
        num_instances_ = 0;
        vertices_order_ = new Vertex[num_vertices];
        instances_ = NULL;
        inversed_embedding_ = new InversedFunction<Vertex>(num_vertices);
    }
    ~Embedding() { delete[] vertices_order_; delete[] instances_; }
    Vertex num_vertices() const { return num_vertices_; }
    Edge num_instances() const { return num_instances_; }
    Vertex* vertices_order() const { return vertices_order_; }
    void set_num_instances(Vertex num_instances) { num_instances_ = num_instances; }
    void insert_vertex(Vertex order, Vertex vertex) {
        vertices_order_[order] = vertex;
        inversed_embedding_->add_x(order, vertex);
    }
    Vertex get_position(Vertex vertex) {
        return inversed_embedding_->inversed_f_[vertex];
    }
};

#endif