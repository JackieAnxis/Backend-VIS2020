import os
import math
import json
import random
import networkx as nx
from flask import Flask
from flask import request
from flask_cors import CORS
from models.get_data import get_test_data
from networkx.readwrite import json_graph
from embeddings.get_cluster import get_cluster_label
from MT.main import generate
from models.S3 import search_similar_structures
from subgraph.main import get_subgraph
from models.utils import load_json_graph
from models.layout import layout, nx_spring_layout

app = Flask(__name__)
CORS(app)

@app.route('/user-study/time', methods=['POST'])
def submit_time():
    req = json.loads(request.data)
    id = req['id']
    time = req['time']
    with open(f'./data/user_study/results/{id}.json', 'w') as f:
        json.dump(time, f)
    return {
        'success': True
    }

@app.route('/user-study/generate', methods=['POST'])
def generate_auto():
    req = json.loads(request.data)
    source = json_graph.node_link_graph(req['source'])
    source_modified = json_graph.node_link_graph(req['sourceModified'])
    target = json_graph.node_link_graph(req['target'])
    #### markers ####
    # [[source id, target id], [], []]
    if 'markers' in req:
        target_generated,_ = generate(source, source_modified, target, markers=req['markers'])
    else:
        target_generated,_ = generate(source, source_modified, target)
    return json_graph.node_link_data(target_generated)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/user-graph/<int:index>')
def user_graph(index):
    GRAPH_COUNT_IN_EACH_CASE = 4

    prefix = './data/user_study/'
    # index = 3
    # settings = json.loads(request.data)
    # index = int(settings['index'])
    print(index)
    if index == 0:
        cases = []
        name = 'email_star'
        case = {'name': name}
        exemplar = load_json_graph(prefix + name + '/0.json')
        targets = []
        for i in range(1, GRAPH_COUNT_IN_EACH_CASE):
            g = load_json_graph(prefix + name + '/' + str(i) + '.json')
            raw = json.loads(json.dumps(json_graph.node_link_data((g))))
            raw['id'] = i
            targets.append(raw)
        
        raw_exemplar = json.loads(json.dumps(json_graph.node_link_data(exemplar)))
        raw_exemplar['id'] = 0 # tutorial exemplar id is 0
        case['exemplar'] = raw_exemplar
        case['targets'] = targets
        cases.append(case)

        return {
            "index": index,
            "cases": cases,
            "mode_sequence": [0, 1]
        }

    exemplar_index = index % GRAPH_COUNT_IN_EACH_CASE
    # mode_sequence_choices = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    mode_sequence_choices = [[0, 1], [0, 1]]
    mode_sequence = mode_sequence_choices[index % len(mode_sequence_choices)]
    dataset_names = ['brain', 'highschool_circle', 'highschool_complex', 'road', 'vis']
    random.shuffle(dataset_names)
    shuffle_dataset_names  = dataset_names
    cases = []
    for name in shuffle_dataset_names:
        case = { 'name': name }
        print(prefix + name + '/' + str(exemplar_index) + '.json')
        exemplar = load_json_graph(prefix + name + '/' + str(exemplar_index) + '.json')
        targets = []
        for i in range(GRAPH_COUNT_IN_EACH_CASE):
            if i != exemplar_index:
                g = load_json_graph(prefix + name + '/' + str(i) + '.json')
                raw = json.loads(json.dumps(json_graph.node_link_data((g))))
                raw['id'] = i
                targets.append(raw)
        raw_exemplar = json.loads(json.dumps(json_graph.node_link_data(exemplar)))
        raw_exemplar['id'] = exemplar_index
        case['exemplar'] = raw_exemplar
        case['targets'] = targets
        cases.append(case)

    return {
        "index": index,
        "cases": cases,
        "mode_sequence": mode_sequence
    }



@app.route('/whole-graph')
def whole_graph():
    # name = 'price'
    # name='finan512'
    name = 'finan512_small'
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
    embedding_method = 'graphwave'
    # embedding_method = 'xnetmf'
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
    app.run(
        host='0.0.0.0',
        port=7777,
    )