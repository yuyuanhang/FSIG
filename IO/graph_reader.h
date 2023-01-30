/**
 * @file graph_reader.h
 * @brief contains an abstract logic class GraphReader and its child classes
 * @details child classes specify the format of file, i.e., the way that vertices and edges are stored
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#ifndef GRAPH_READER_H
#define GRAPH_READER_H

#include <string>
#include <fstream>
#include <iostream>
#include "../graph/graph.h"
#include "../utility/defines.h"

using std::string;
using std::ifstream;
using std::cout;
using std::endl;

/**
* @brief An abstract logic class that reads graph from a file
*/
class GraphReader {
public:
	/**
	 * @param[in]    file_path : the path of graph file
	 * @param[in]    read_graph : the returned graph 
	*/
	virtual void read(string file_path, Graph* &read_graph, bool is_labeled) = 0;
};

/*!
* An logic class that reads graph from a survey format file and stores graph as a plain graph
*/
class UndirectedSurveyPlainReader : public GraphReader {
public:
	void read(string file_path, Graph* &read_graph, bool is_labeled) override;
};

#endif