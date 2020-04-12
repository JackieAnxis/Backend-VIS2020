import os
import json
import networkx as nx
from flask import Flask
from flask import request
from flask_cors import CORS
from models.get_data import get_test_data
from networkx.readwrite import json_graph
from embeddings.get_cluster import get_cluster_label
from MT.main import generate
from deformation.fuse import fuse_main
from models.S3 import search_similar_structures
from deformation.regal_alignment import get_regal_correspondence
from subgraph.main import get_subgraph

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/whole-graph')
def whole_graph():
    name = 'price'
    # name = 'VIS'
    # name = 'power-662-bus'
    # name = 'bn-mouse-kasthuri'
    # name = 'bn-mouse_visual-cortex_2'
    # name = 'email-Eu-core'
    # name = 'bio-DM-LC'
    # name = 'road-euroroad'

    # data_path = './data/' + name + '/graph.json'
    data_path = './data/' + name + '/graph-with-pos.json'
    # cluster_label = get_cluster_label()
    with open(data_path) as graph_data:
        return {
            "name": name,
            "data": json.loads(graph_data.read()),
            # "cluster": cluster_label
        }

@app.route('/sub-graph', methods=['POST'])
def sub_graph():
  settings = json.loads(request.data)
  markers = settings['markers']
  r = 10
  graph = get_subgraph(markers, r)
  return graph

@app.route('/test')
def test():
    return get_test_data()


@app.route('/compute', methods=['POST'])
def compute():
    data = json.loads(request.data)
    # # TODO: compute
    # os.chdir('./deformation/')
    # res = generate(json_graph.node_link_graph(data['source']), json_graph.node_link_graph(
    #     data['source_modified']), json_graph.node_link_graph(data['target']))
    # os.chdir('../')
    # with open('./data/test/_source.json', 'w') as file:
    #     json.dump(data['source_modified'], file)
    #
    # return {
    #     'target_generated': json_graph.node_link_data(res)
    # }

@app.route('/search', methods=['POST'])
def search():
    settings = json.loads(request.data)
    dataset = settings['dataset']
    # embedding_method = 'graphwave'
    embedding_method = 'xnetmf'
    # embedding_method = 'graphwave2'
    # embedding_method = 'role2vec'
    # embedding_method = 'walklets'
    # embedding_method = 'walklets2'
    path_prefix = './data/'
    settings["embedding_path"] = path_prefix + dataset + '/' + embedding_method + '.csv'
    settings["edgelist_path"] = path_prefix + dataset + '/graph.edgelist'
    s = search_similar_structures(settings)
    connected_components = s.get_knn_connected_components()

    exemplar_compound_graph = s.get_exemplar_compound_graph()
    knn_exemplar_maps = s.get_knn_nodes_to_exemplar_maps()

    connected_components_subgraph = []
    G = nx.read_edgelist(settings['edgelist_path'], nodetype=int,
                     data=(('weight', float),))

    origin_nodes = set(map(int, settings['search_nodes']))
    for i, node_list in enumerate(connected_components):
        node_set = set(node_list)
        if len(node_set & origin_nodes) * 2 > len(origin_nodes):
            continue
        connected_components_subgraph.append({
            'graph': json_graph.node_link_data(G.subgraph(node_list)),
            'node_maps': knn_exemplar_maps[i]
        })

    print(exemplar_compound_graph)
    return {
        'suggestions': connected_components_subgraph
    }
    
@app.route('/apply-deformation', methods=['POST'])
def apply_deformation():
    settings = json.loads(request.data)
    # markers = settings['autoMarkers']
    markers = settings['manualMarkers']
    # Method1: using knn components mapping
    # correspondence = get_knn_sim_correspondence(settings['nodeMaps'], settings['sourceGraph'], settings['targetGraph'])
    # print('correspondence:')
    # print(correspondence)

    # Method2: using REGAL embedding similarity matrix
    correspondence, id_pairs = get_regal_correspondence(settings['sourceGraph'], settings['targetGraph'])
    print('correspondence:')
    print(correspondence)
    print('correspondence id pair:')
    print(id_pairs)

    # Method3: using markers and others
    # correspondence = [] # (generate in deformation trasfer)
    # whole_graph_data = json_graph.node_link_graph(settings['wholeGraphData'])
    source_graph = json_graph.node_link_graph(settings['sourceGraph'])
    deformed_source_graph = json_graph.node_link_graph(settings['_sourceGraph'])
    target_graph = json_graph.node_link_graph(settings['targetGraph'])
    deformed_target_graph_network = generate(source_graph, deformed_source_graph, target_graph, markers)
    deformed_target_graph = json_graph.node_link_data(deformed_target_graph_network)
    return deformed_target_graph

@app.route('/apply-deformation-wholegraph', methods=['POST'])
def apply_deformation_wholegraph():
    settings = json.loads(request.data)
    whole_graph_data = json_graph.node_link_graph(settings['wholeGraphData'])
    deformed_target_graphs = settings['deformedTargetGraph']
    deformed_target_graphs_values = list(deformed_target_graphs)
    for i in range(0, len(deformed_target_graphs_values)):
      deformed_target_graphs_values[i] = json_graph.node_link_graph(deformed_target_graphs_values[i])
    new_G = json_graph.node_link_data(fuse_main(whole_graph_data, deformed_target_graphs_values))
    return new_G

if __name__ == '__main__':
    app.run()