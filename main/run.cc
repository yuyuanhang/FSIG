/**
 * @file run.cc
 * @brief performs subgraph matching
 * @details the process includes (1) read graph files; (2) determine the way graphs are store;
 * @author Yu Yuanhang
 * @version 1.0
 * @date 2022.07.22
 */

#include <string>
#include <iostream>
#include <vector>
#include "../graph/graph.h"
#include "../IO/graph_reader.h"
#include "../graph_format_transformer/graph_format_transformer.h"
#include "../utility/inverted_index.h"
#include "../match/subgraph_match.h"
#include "../match/MMatch.h"
#include "../match/GSI.h"
#include "../utility/time_recorder.h"

using std::cout;
using std::vector;
using std::string;
using std::endl;

/**
* @brief        performs subgraph matching
* @param[in]    argv[1] : the path of data graph file
* @param[in]    argv[2] : the path of query graph file 
* @param[in]    argv[3] (optional) : the type of used algorithm (default : 1) 1: MMatch; 2: GSI
* @param[in]    argv[4] (optional) : if the input graph is a labeled graph (default : 1)
* @param[in]    argv[5] (optional) : the path of file that stores the result of subgraph matching (default : ans.txt)
* @param[in]    argv[6] (optional) : the id of used device (default : 0)        
* @return       
*   0 execute successfully \n
*   -1 input parameters error \n
*/
int main(int argc, const char * argv[]) {
    string output_file = "ans.txt";
    int dev_id = 0;
    int algorithm_type = 1;
    bool is_labeled = 1;

    if (argc < 3) {
        cout << "invalid arguments!" << endl;
        return -1;
    }
    string data_file = argv[1];
    string query_file = argv[2];
    if (argc >= 4) {
        algorithm_type = atoi(argv[3]);
    }
    if (argc >= 5) {
        if (atoi(argv[4]) == 0) {
            is_labeled = false;
        }
    }
    if (argc >= 6) {
        output_file = argv[5];
    }
    if (argc >= 7) {
        dev_id = atoi(argv[6]);
    }

    // record time point
    TimeRecorder* time_recorder = new TimeRecorder();

    // read graph files, use GraphReader abstract logic class and its child classes 
    Graph* abstract_data_graph;
    Graph* abstract_query_graph;
    UndirectedSurveyPlainReader* survey_graph_reader = new UndirectedSurveyPlainReader();
    time_recorder->event_start("read data");
    survey_graph_reader->read(data_file, abstract_data_graph, is_labeled);
    time_recorder->event_end("read data");
    time_recorder->event_start("read query");
    survey_graph_reader->read(query_file, abstract_query_graph, is_labeled);
    time_recorder->event_end("read query");

    UndirectedPlainGraph* plain_data_graph = dynamic_cast<UndirectedPlainGraph*>(abstract_data_graph);
    if (plain_data_graph == nullptr) {
        cout << "Error: transform abstract graph to plain graph; data graph is not a plain graph" << endl;
        return -1;
    }
    UndirectedPlainGraph* plain_query_graph = dynamic_cast<UndirectedPlainGraph*>(abstract_query_graph);
    if (plain_query_graph == nullptr) {
        cout << "Error: transform abstract graph to plain graph; query graph is not a plain graph" << endl;
        return -1;
    }

#ifdef DEBUG
    plain_data_graph->print_graph();
    plain_query_graph->print_graph();
#endif

    /*
    * store the graph, it mainly includes two indices:
    * (1) CSR / PCSR format to store sparse graph
    * (2) inverted index to access vertices with a specific label
    */
    GPUSubgraphMatcher* matcher;
    GraphFormatTransformer* transformer;

    if (algorithm_type == 1) {
        transformer = new UndirectedPlainToCSRTransformer();
    } else if (algorithm_type == 2) {
        transformer = new UndirectedPlainToPCSRTransformer();
    }

    time_recorder->event_start("organise graph");
    transformer->transform(dynamic_cast<Graph*>(plain_data_graph), abstract_data_graph);
    time_recorder->event_end("organise graph");

    time_recorder->print_all_events();

    if (algorithm_type == 1) {
        UndirectedCSRGraph* CSR_data_graph = dynamic_cast<UndirectedCSRGraph*>(abstract_data_graph);
        if (CSR_data_graph == NULL) {
            cout << "Error: transform plain graph to CSR graph;" << endl;
            return -1;
        }
        InvertedIndex<Label, Vertex>* inverted_data_label = new InvertedIndex<Label, Vertex>(CSR_data_graph->num_vertices(), CSR_data_graph->num_labels(), CSR_data_graph->labels_);
        matcher = new MMatch(plain_query_graph, CSR_data_graph, inverted_data_label, dev_id);
    } else if (algorithm_type == 2) {
        UndirectedPCSRGraph* PCSR_data_graph = dynamic_cast<UndirectedPCSRGraph*>(abstract_data_graph);
        if (PCSR_data_graph == NULL) {
            cout << "Error: transform plain graph to PCSR graph;" << endl;
            return -1;
        }
        InvertedIndex<Label, Vertex>* inverted_data_label = new InvertedIndex<Label, Vertex>(PCSR_data_graph->num_vertices(), PCSR_data_graph->num_labels(), PCSR_data_graph->labels_);

        Vertex* query_signature_table = NULL;
        Vertex* data_signature_table = NULL;
        UndirectedPlainToPCSRTransformer::build_signature(false, query_signature_table, plain_query_graph);
        UndirectedPlainToPCSRTransformer::build_signature(true, data_signature_table, plain_data_graph);

        matcher = new GSI(plain_query_graph, PCSR_data_graph, inverted_data_label, query_signature_table, data_signature_table, dev_id);
    }
    
    matcher->match();

    return 0;
}
