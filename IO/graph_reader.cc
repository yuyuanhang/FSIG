/**
 * @file graph_reader.cc
 * @brief contains an abstract logic class GraphReader and its child classes
 * @details child classes specify the format of file, i.e., the way that vertices and edges are stored
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#include "graph_reader.h"

void UndirectedSurveyPlainReader::read(string file_path, Graph* &read_graph, bool is_labeled) {
	UndirectedPlainGraph* plain_graph = new UndirectedPlainGraph();

	ifstream survey_file(file_path);
    if (!survey_file.is_open()) {
        cout << "Error: cannot open file; undirected survey file" << endl;
        exit(0);
    }

    char type; survey_file >> type;
    if (type == 't') {
    	Vertex num_vertices;
    	Edge num_edges;
    	survey_file >> num_vertices >> num_edges;
    } else {
    	cout << "Error: graph file format is not correct; undirected survey file" << endl;
        exit(0);
    }

    while (!survey_file.eof()) {
    	survey_file >> type;
		if (type == 'v') {
			Vertex vertex_id;
			Label vertex_label;
			Vertex vertex_degree;

    		survey_file >> vertex_id >> vertex_label >> vertex_degree;
            if (is_labeled) {
                plain_graph->add_vertex(vertex_id, vertex_label);
            } else {
                plain_graph->add_vertex(vertex_id, 0);
            }
    	} else if (type == 'e') {
    		Vertex first_vertex;
    		Vertex second_vertex;

    		survey_file >> first_vertex >> second_vertex;
			plain_graph->add_edge(first_vertex, second_vertex);
    	}
    }
    plain_graph->set_min_degree();
    survey_file.close();
    read_graph = dynamic_cast<Graph*>(plain_graph);
}
