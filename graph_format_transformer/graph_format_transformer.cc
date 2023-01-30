/**
 * @file graph_format_transformer.cc
 * @brief contains an abstract data class GraphFormatTransformer and its child classes
 * @details child classes specify the format of input graph and output graph, i.e., which graph to which graph
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#include "graph_format_transformer.h"

void UndirectedPlainToCSRTransformer::transform(Graph* input_graph, Graph* &output_graph) {
	UndirectedPlainGraph* plain_graph = dynamic_cast<UndirectedPlainGraph*>(input_graph);
	if (plain_graph == nullptr) {
        cout << "Error: transform plain graph to CSR graph; input graph is not a plain graph" << endl;
        exit(0);
    }

    UndirectedCSRGraph* CSR_graph = new UndirectedCSRGraph();
    CSR_graph->set_num_vertices(plain_graph->num_vertices());
    CSR_graph->set_num_edges(plain_graph->num_edges());
    CSR_graph->set_num_labels(plain_graph->num_labels());

    CSR_graph->labels_ = new Label[CSR_graph->num_vertices()];
    copy(plain_graph->labels_.begin(), plain_graph->labels_.end(), CSR_graph->labels_);

    CSR_graph->offset_ = new Edge[CSR_graph->num_vertices()+1];
    CSR_graph->edges_ = new Vertex[2*CSR_graph->num_edges()];
    CSR_graph->offset_[0] = 0;
	for (Vertex i = 0; i < plain_graph->num_vertices(); i++) {
		sort(plain_graph->edges_[i].begin(), plain_graph->edges_[i].end(), less<Edge>());
		CSR_graph->offset_[i+1] = CSR_graph->offset_[i] + plain_graph->edges_[i].size();
		copy(plain_graph->edges_[i].begin(), plain_graph->edges_[i].end(), CSR_graph->edges_+CSR_graph->offset_[i]);
	}

	output_graph = dynamic_cast<Graph*>(CSR_graph);
}

void UndirectedPlainToPCSRTransformer::transform(Graph* input_graph, Graph* &output_graph) {
    UndirectedPlainGraph* plain_graph = dynamic_cast<UndirectedPlainGraph*>(input_graph);
    if (plain_graph == nullptr) {
        cout << "Error: transform plain graph to PCSR graph; input graph is not a plain graph" << endl;
        exit(0);
    }

    UndirectedPCSRGraph* PCSR_graph = new UndirectedPCSRGraph();
/*#ifdef READ
    ifstream pcsr_reader("human_pcsr.bin", std::ios::out | std::ios::binary);

    pcsr_reader.read((char*)&PCSR_graph->num_vertices_, sizeof(Vertex));
    pcsr_reader.read((char*)&PCSR_graph->num_edges_, sizeof(Edge));
    pcsr_reader.read((char*)&PCSR_graph->num_labels_, sizeof(Label));

    PCSR_graph->labels_ = new Label[PCSR_graph->num_vertices()];
    pcsr_reader.read((char*)PCSR_graph->labels_, sizeof(Label)*PCSR_graph->num_vertices_);

    pcsr_reader.read((char*)&PCSR_graph->num_units_, sizeof(Vertex));
    PCSR_graph->units_ = new PCSRUnit*[PCSR_graph->num_units_];
    for (Vertex i = 0; i < PCSR_graph->num_units_; i++) {
        PCSR_graph->units_[i] = new PCSRUnit();

        pcsr_reader.read((char*)&PCSR_graph->units_[i]->num_keys_, sizeof(Vertex));
        pcsr_reader.read((char*)&PCSR_graph->units_[i]->num_edges_, sizeof(Vertex));

        PCSR_graph->units_[i]->offset_ = new Edge[32*PCSR_graph->units_[i]->num_keys_];
        PCSR_graph->units_[i]->edges_ = new Vertex[PCSR_graph->units_[i]->num_edges_];
        pcsr_reader.read((char*)PCSR_graph->units_[i]->offset_, sizeof(Edge)*32*PCSR_graph->units_[i]->num_keys_);
        pcsr_reader.read((char*)PCSR_graph->units_[i]->edges_, sizeof(Vertex)*PCSR_graph->units_[i]->num_edges_);
    }

    pcsr_reader.close();
    output_graph = dynamic_cast<Graph*>(PCSR_graph);
    return;
#endif*/

    PCSR_graph->set_num_vertices(plain_graph->num_vertices());
    PCSR_graph->set_num_edges(plain_graph->num_edges());
    PCSR_graph->set_num_labels(plain_graph->num_labels());
    PCSR_graph->labels_ = new Label[PCSR_graph->num_vertices()];
    copy(plain_graph->labels_.begin(), plain_graph->labels_.end(), PCSR_graph->labels_);
    for (Vertex i = 0; i < plain_graph->num_vertices(); i++) {
        sort(plain_graph->edges_[i].begin(), plain_graph->edges_[i].end(), less<Edge>());
    }

    PCSR_graph->num_units_ = (PCSR_graph->num_labels() * (PCSR_graph->num_labels() + 1)) / 2;
    PCSR_graph->units_ = new PCSRUnit*[PCSR_graph->num_units_];
    set<Vertex>* keys_set = new set<Vertex>[PCSR_graph->num_units_]; // vertex set, accelerate insertion
    vector<Vertex>* keys = new vector<Vertex>[PCSR_graph->num_units_];
    vector<vector<vector<Vertex>>> keys_neighbors_in_unit(PCSR_graph->num_units_);
#ifdef DEBUG
    cout << "partitioning graph and collecting edges for each partitioned graph..." << endl;
#endif
    // partition data graph based on edge label (if have) or (vertex label, vertex label)
    for (Vertex i = 0; i < PCSR_graph->num_vertices(); i++) {
        Label first_label = plain_graph->labels_[i];
        auto neighbors = plain_graph->neighbors(i);
        for (Vertex j = 0; j < neighbors.size(); j++) {
            Label second_label = plain_graph->labels_[neighbors[j]];
            Label edge_label = PCSR_graph->get_edge_label(first_label, second_label);
            auto insert_first = keys_set[edge_label].insert(i);
            if (insert_first.second) {
                keys[edge_label].push_back(i);
                keys_neighbors_in_unit[edge_label].push_back(vector<Vertex>());
            }
            keys_neighbors_in_unit[edge_label][keys[edge_label].size() - 1].push_back(neighbors[j]);
        }
    }
#ifdef DEBUG
    cout << "writing edges for each partitioned graph..." << endl;
#endif
    // write edges to units
    for (Vertex i = 0; i < PCSR_graph->num_units_; i++) {
        // write parameters and allocate space
        PCSR_graph->units_[i] = new PCSRUnit();
        PCSRUnit* crt_unit = PCSR_graph->units_[i];
        crt_unit->num_keys_ = keys[i].size();
        Edge num_edges_crt_unit = 0;
        for (Vertex j = 0; j < keys[i].size(); j++) {
            num_edges_crt_unit += keys_neighbors_in_unit[i][j].size();
        }
        crt_unit->num_edges_ = num_edges_crt_unit;
        crt_unit->offset_ = new Edge[32 * keys[i].size()];  // the size is 32 * key_num
        crt_unit->edges_ = new Vertex[num_edges_crt_unit];
        // initialise offset array
        for (Vertex j = 0; j < keys[i].size() * 16; j++) {
            crt_unit->offset_[2*j] = INVALID;
            crt_unit->offset_[2*j+1] = 0;
        }
        // initialise edge array
        for (Vertex j = 0; j < num_edges_crt_unit; j++) {
            crt_unit->edges_[j] = INVALID;
        }
        // assign keys into buckets
        vector<unsigned>* buckets = new vector<unsigned>[keys[i].size()];
        for (Vertex j = 0; j < keys[i].size(); j++) {
            Vertex vertex_id = keys[i][j];
            unsigned pos = hash(&vertex_id, 4, HASHSEED) % keys[i].size();
            buckets[pos].push_back(vertex_id);
        }
        // collect empty buckets for overflow bucket to use
        queue<Vertex> empty_buckets;
        for (Vertex j = 0; j < keys[i].size(); j++) {
            if (buckets[j].empty()) {
                empty_buckets.push(j);
            }
        }
        // dump buckets into array
        // first, record vertex id
        for (Vertex j = 0; j < keys[i].size(); j++) {
            if (buckets[j].empty()) {
                continue;
            }
            Vertex crt_bucket_size = buckets[j].size();
            Vertex num_required_slots = crt_bucket_size / 15;
            Vertex num_residual_keys = crt_bucket_size % 15;
            Vertex target_slot_id = j;
            for (Vertex k = 0; k < num_required_slots; k++) {
                for (Vertex l = 0; l < 15; l++) {
                    crt_unit->offset_[32*target_slot_id+2*l] = buckets[j][k*15+l];
                }
                Vertex empty_slot = empty_buckets.front();
                empty_buckets.pop();
                crt_unit->offset_[32*target_slot_id+30] = empty_slot;
                target_slot_id = empty_slot;
            }
            for (Vertex l = 0; l < num_residual_keys; l++) {
                crt_unit->offset_[32*target_slot_id+2*l] = buckets[j][num_required_slots*15+l];
            }
        }
        // then write neighbor lists
        Edge crt_pos = 0;
        for (Vertex j = 0; j < keys[i].size(); j++) {
            for (Vertex k = 0; k < 15; k++) {
                Vertex vertex_id = crt_unit->offset_[32*j+2*k];
                if (vertex_id == INVALID) {
                    break;
                }
                Vertex idx = find(keys[i].begin(), keys[i].end(), vertex_id) - keys[i].begin();
                auto neighbor_list = keys_neighbors_in_unit[i][idx];
                crt_unit->offset_[32*j+2*k+1] = crt_pos;
                crt_unit->offset_[32*j+2*(k+1)+1] = crt_pos + neighbor_list.size();
                for (Vertex l = 0; l < neighbor_list.size(); l++) {
                    crt_unit->edges_[crt_pos++] = neighbor_list[l];
                }
            }
        }
        delete[] buckets;
    }
    delete[] keys;
#ifdef DEBUG
    Edge sum_partitioned_edges = 0;
    for (Vertex i = 0; i < PCSR_graph->num_units_; i++) {
        sum_partitioned_edges += PCSR_graph->units_[i]->num_edges_;
    }
    cout << "partition graphs includes " << sum_partitioned_edges << " edges" << endl;
#endif
/*#ifdef STORE
    ofstream pcsr_file("human_pcsr.bin", std::ios::out | std::ios::binary);

    pcsr_file.write((char*)&PCSR_graph->num_vertices_, sizeof(Vertex));
    pcsr_file.write((char*)&PCSR_graph->num_edges_, sizeof(Edge));
    pcsr_file.write((char*)&PCSR_graph->num_labels_, sizeof(Label));
    pcsr_file.write((char*)PCSR_graph->labels_, sizeof(Label)*plain_graph->num_vertices());
    pcsr_file.write((char*)&PCSR_graph->num_units_, sizeof(Vertex));
    for (Vertex i = 0; i < PCSR_graph->num_units_; i++) {
        pcsr_file.write((char*)&PCSR_graph->units_[i]->num_keys_, sizeof(Vertex));
        pcsr_file.write((char*)&PCSR_graph->units_[i]->num_edges_, sizeof(Vertex));
        pcsr_file.write((char*)PCSR_graph->units_[i]->offset_, sizeof(Edge)*32*PCSR_graph->units_[i]->num_keys_);
        pcsr_file.write((char*)PCSR_graph->units_[i]->edges_, sizeof(Vertex)*PCSR_graph->units_[i]->num_edges_);
    }

    pcsr_file.close();
#endif*/
    output_graph = dynamic_cast<Graph*>(PCSR_graph);
}

void UndirectedPlainToPCSRTransformer::build_signature(bool column_oriented, Vertex* &signature_table, UndirectedPlainGraph* graph) {
    // allocate signature table
    unsigned num_bits = sizeof(Vertex)*8;
    unsigned len_signature = SIGLEN / num_bits;
    signature_table = new Vertex[len_signature * graph->num_vertices()];
    memset(signature_table, 0, sizeof(Vertex) * len_signature * graph->num_vertices());

    unsigned size_slot = 2;
    unsigned num_slots_ = (SIGLEN - num_bits) / size_slot;
    for(Vertex i = 0; i < graph->num_vertices(); i++) {
        // first, write vertex label into signature
        Label vertex_label = graph->labels_[i];
        signature_table[i * len_signature] = vertex_label;

        // then, write neighbors of this vertex into signature
        auto neighbors = graph->neighbors(i);
        for (Vertex j = 0; j < neighbors.size(); j++) {
            Vertex neighbor = neighbors[j];

            // hash neighbor's label
            Label neighbor_label = graph->labels_[neighbor];
            Vertex hash_pos = hash(&neighbor_label, 4, HASHSEED) % num_slots_;

            // sizeof(Vertex) / size_slot represents the number of slots in a Vertex
            // a denotes index in Vertex array; b denotes index in a Vertex
            unsigned a = size_slot * hash_pos / num_bits;
            unsigned b = size_slot * hash_pos % num_bits;
            unsigned t = signature_table[i * len_signature + 1 + a];
            // fetch corresponding bits via bit operations
            unsigned c = 3 << (size_slot * b);
            c = c & t;
            // these bits are in low-end
            c = c >> (size_slot * b);
            // change flag
            switch (c) {
                case 0:
                    c = 1;
                    break;
                case 1:
                    c = 3;
                    break;
                default:  // c == 3
                    c = 3;
                    break;
            }
            // write back
            c = c << (size_slot * b);
            t = t | c;
            signature_table[i * len_signature + 1 + a] = t;
        }
    }

    if (column_oriented) {
        //change to column oriented for data graph
        Vertex* new_table = new Vertex[len_signature * graph->num_vertices()];
        unsigned base = 0;
        for (unsigned j = 0; j < len_signature; j++) {
            for(Vertex k = 0; k < graph->num_vertices(); k++) {
                new_table[base++] = signature_table[k * len_signature + j];
            }
        }
        delete[] signature_table;
        signature_table = new_table;
    }
}
