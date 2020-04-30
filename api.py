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

target_nodes = [[16384, 16385, 16386, 16387, 16388, 16389, 16390, 16391, 16128, 16141, 16269, 16270, 16271, 16152, 16286, 16163,
      16297, 16174, 16308, 16057, 16185, 16319, 16196, 16381, 16071, 16072, 16073, 16202, 16330, 16375, 16383, 16341,
      16214, 16215, 16216, 16347, 16382, 16351, 16352, 16353, 16110, 16111, 16112, 16369, 16254, 16370, 16372, 16373,
      16374, 16255, 16376, 16377, 16378, 16379, 16380, 16253, 16126, 16127],
     [73731, 73732, 73733, 73991, 73992, 73995, 73996, 73997, 73998, 73999, 74000, 74001, 73874, 73747, 73748, 73749,
      73875, 73876, 74002, 74003, 74004, 74005, 74006, 74007, 74008, 74009, 74010, 74011, 73762, 73890, 73891, 73892,
      74013, 74014, 73773, 74012, 73907, 73784, 73918, 73795, 73929, 73806, 73678, 73940, 73817, 73692, 73693, 73694,
      73823, 73951, 73962, 73835, 73836, 73837, 73968, 73972, 73973, 73974],
     [43651, 43779, 43913, 43790, 43665, 43666, 43667, 43796, 43924, 43935, 43808, 43809, 43810, 43941, 43945, 43946,
      43947, 43704, 43705, 43706, 43964, 43965, 43968, 43969, 43970, 43971, 43972, 43973, 43974, 43847, 43976, 43977,
      43720, 43721, 43722, 43848, 43849, 43975, 43978, 43979, 43980, 43981, 43982, 43983, 43986, 43735, 43863, 43864,
      43865, 43987, 43746, 43880, 43757, 43984, 43891, 43985, 43768, 43902],
     [53504, 53761, 53630, 53760, 53764, 53762, 53758, 53631, 53763, 53766, 53767, 53759, 53765, 53517, 53645, 53646,
      53647, 53528, 53662, 53539, 53629, 53673, 53550, 53684, 53561, 53433, 53695, 53572, 53447, 53448, 53449, 53578,
      53706, 53717, 53590, 53591, 53592, 53723, 53727, 53728, 53729, 53486, 53487, 53488, 53745, 53746, 53748, 53749,
      53750, 53751, 53752, 53753, 53754, 53755, 53756, 53757, 53502, 53503],
     [57984, 57985, 57986, 57987, 57988, 57989, 57990, 57863, 57736, 57737, 57738, 57864, 57865, 57994, 57995, 57996,
      57997, 57998, 57999, 58002, 58003, 58000, 58001, 57751, 57879, 57880, 57881, 57762, 57896, 57773, 57907, 57784,
      57991, 57992, 57918, 57993, 57667, 57795, 57929, 57806, 57681, 57682, 57683, 57812, 57940, 57951, 57824, 57825,
      57826, 57957, 57961, 57962, 57963, 57720, 57721, 57722, 57980, 57981],
     # [41472, 41473, 41474, 41605, 41609, 41610, 41611, 41368, 41369, 41370, 41628, 41629, 41632, 41633, 41634, 41635,
     #  41636, 41637, 41638, 41511, 41384, 41385, 41386, 41512, 41513, 41643, 41642, 41645, 41644, 41646, 41647, 41650,
     #  41651, 41648, 41649, 41399, 41527, 41528, 41529, 41410, 41544, 41421, 41555, 41432, 41639, 41566, 41640, 41315,
     #  41443, 41641, 41577, 41454, 41329, 41330, 41331, 41460, 41588, 41599],
     # [8706, 8840, 8717, 8851, 8728, 8862, 8611, 8739, 8873, 8750, 8625, 8626, 8627, 8756, 8884, 8895, 8768, 8769, 8770,
     #  8901, 8944, 8905, 8906, 8907, 8664, 8665, 8666, 8924, 8925, 8928, 8929, 8930, 8931, 8932, 8933, 8934, 8807, 8680,
     #  8681, 8682, 8808, 8809, 8823, 8935, 8936, 8937, 8938, 8939, 8940, 8941, 8942, 8943, 8695, 8824, 8825, 8945, 8946,
     #  8947],
     # [6275, 6403, 6537, 6414, 6289, 6290, 6291, 6420, 6548, 6559, 6432, 6433, 6434, 6565, 6569, 6570, 6571, 6328, 6329,
     #  6330, 6588, 6589, 6592, 6593, 6594, 6595, 6596, 6597, 6598, 6471, 6600, 6601, 6344, 6345, 6346, 6472, 6473, 6599,
     #  6602, 6603, 6604, 6605, 6606, 6607, 6608, 6359, 6487, 6488, 6489, 6610, 6611, 6370, 6609, 6504, 6381, 6515, 6392,
     #  6526],
     # [69891, 70025, 69902, 70119, 70036, 69785, 69913, 70047, 69924, 69799, 69800, 69801, 69930, 70058, 70069, 69942,
     #  69943, 69944, 70075, 70079, 70080, 70081, 69838, 69839, 69840, 70097, 70098, 70100, 70101, 70102, 70103, 70104,
     #  70105, 70106, 70107, 70108, 70109, 69854, 69855, 69856, 69981, 69982, 69983, 70116, 70117, 70118, 70110, 70111,
     #  70112, 70113, 70114, 70115, 69869, 69998, 69997, 69999, 69880, 70014],
     # [22784, 22785, 22786, 22917, 22921, 22922, 22923, 22680, 22681, 22682, 22940, 22941, 22944, 22945, 22946, 22947,
     #  22948, 22949, 22950, 22823, 22696, 22953, 22697, 22698, 22824, 22825, 22951, 22952, 22954, 22955, 22957, 22956,
     #  22958, 22959, 22962, 22711, 22839, 22840, 22841, 22963, 22960, 22722, 22961, 22856, 22733, 22867, 22744, 22878,
     #  22627, 22755, 22889, 22766, 22641, 22642, 22643, 22772, 22900, 22911],
        [51337, 51214, 51348, 51421, 51097, 51359, 51111, 51112, 51113, 51242, 51370, 51381, 51254,
         51255, 51256, 51387, 51391, 51392, 51393, 51150, 51151, 51152, 51409, 51410, 51412, 51413,
         51414, 51415, 51416, 51417, 51418, 51419, 51420, 51293, 51294, 51295, 51168, 51166, 51167,
         51422, 51423, 51424, 51425, 51426, 51427, 51430, 51431, 51429, 51428, 51309, 51310, 51311,
         51326
         ],
        [74116, 74378, 74130, 74131, 74132, 74261, 74389, 74400, 74273, 74274, 74275, 74406, 74410,
         74411, 74412, 74169, 74170, 74171, 74426, 74427, 74429, 74431, 74432, 74430, 74433, 74434,
         74435, 74436, 74437, 74439, 74312, 74185, 74186, 74187, 74313, 74314, 74440, 74441, 74442,
         74443, 74444, 74447, 74448, 74445, 74446, 74328, 74329, 74330, 74345, 74356, 74438, 74367
         ],
        [9344, 9345, 9102, 9103, 9104, 9361, 9362, 9364, 9365, 9366, 9367, 9368, 9369, 9370, 9371, 9372,
         9245, 9246, 9247, 9120, 9118, 9119, 9373, 9374, 9375, 9376, 9377, 9378, 9379, 9380, 9381,
         9382, 9261, 9262, 9263, 9383, 9278, 9289, 9300, 9049, 9311, 9063, 9064, 9065, 9194, 9322,
         9333, 9206, 9207, 9208, 9339, 9343
         ],
        [71812, 71687, 71688, 71689, 71818, 71830, 71831, 71832, 71963, 71967, 71968, 71801, 71969,
         72005, 71726, 71727, 71728, 71986, 71987, 71990, 71991, 71992, 71993, 71994, 71995, 71996,
         71997, 71742, 71743, 71744, 71998, 71999, 71869, 71870, 71871, 72006, 72007, 72000, 72001,
         72002, 72003, 72004, 71757, 71886, 71885, 71887, 72008, 72009, 71902, 71779, 71790, 71673
         ],
        [25472, 25729, 25728, 25730, 25731, 25732, 25734, 25735, 25733, 25485, 25613, 25614, 25615,
         25470, 25725, 25471, 25726, 25630, 25727, 25652, 25401, 25663, 25415, 25416, 25417, 25546,
         25685, 25558, 25559, 25560, 25691, 25695, 25696, 25697, 25454, 25455, 25456, 25713, 25714,
         25716, 25717, 25718, 25719, 25721, 25722, 25723, 25724, 25597, 25598, 25599
         ]]
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
        modified = load_json_graph(prefix + name + '/' + 'modified' + '.json')
        raw_modified = json.loads(json.dumps(json_graph.node_link_data(modified)))
        raw_modified['id'] = 0
        case['modified'] = raw_modified
        cases.append(case)

        return {
            "index": index,
            "cases": cases,
            "mode_sequence": [0, 2, 1]
        }

    # exemplar_index = index % GRAPH_COUNT_IN_EACH_CASE
    exemplar_index = 0
    mode_sequence_choices = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
    # mode_sequence_choices = [[0, 1], [1, 0]]
    # mode_sequence_choices = [[1]]
    mode_sequence = mode_sequence_choices[index % len(mode_sequence_choices)]
    dataset_names = ['brain', 'highschool_circle', 'highschool_complex', 'road']
    # dataset_names = ['highschool_complex']
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
        modified = load_json_graph(prefix + name + '/' + 'modified' + '.json')
        raw_modified = json.loads(json.dumps(json_graph.node_link_data(modified)))
        raw_modified['id'] = exemplar_index
        case['modified'] = raw_modified
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
    name='finan512'
    # name = 'finan512_small'
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
    # s = search_similar_structures(settings)
    # connected_components = s.get_knn_connected_components()
    connected_components = target_nodes

    # exemplar_compound_graph = s.get_exemplar_compound_graph()
    # knn_exemplar_maps = s.get_knn_nodes_to_exemplar_maps()

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
            # 'node_maps': knn_exemplar_maps[i]
        })

    # print(exemplar_compound_graph)
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
    # correspondence, id_pairs = get_regal_correspondence(settings['sourceGraph'], settings['targetGraph'])
    # print('correspondence:')
    # print(correspondence)
    # print('correspondence id pair:')
    # print(id_pairs)

    # Method3: using markers and others
    # correspondence = [] # (generate in deformation trasfer)
    # whole_graph_data = json_graph.node_link_graph(settings['wholeGraphData'])
    source_graph = json_graph.node_link_graph(settings['sourceGraph'])
    deformed_source_graph = json_graph.node_link_graph(settings['_sourceGraph'])
    target_graph = json_graph.node_link_graph(settings['targetGraph'])
    # deformed_target_graph_network = generate(source_graph, deformed_source_graph, target_graph, markers)
    deformed_target_graph_network = generate(source_graph, deformed_source_graph, target_graph)
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