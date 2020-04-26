import os
import shutil
from models.utils import load_json_graph, save_json_graph, save_json_graph_with_attr
from models.gk_weisfeiler_lehman import GK_WL
import networkx as nx

def relable_from_zero(g):
    i = 0
    label = {}
    for node in g.nodes:
        label[node] = i
        i += 1
    return nx.relabel_nodes(g, label)

if __name__ == '__main__':
    dataset = 'email'
    G = load_json_graph('./data/' + dataset + '/graph-with-pos.json')
    graphs = [G.subgraph(c) for c in nx.connected_components(G)]
    node = 29720 # 768(vis)
    threshold = 0.7
    exemplar = graphs[0]
    for graph in graphs:
        if node in graph.nodes:
            exemplar = graph
            break
    wl = GK_WL()
    similarity_tuple = []
    _exemplar = relable_from_zero(exemplar)

    if True:
    # for exemplar in graphs:
    #     if len(exemplar.nodes) < 5 or len(exemplar.nodes) > 25:
    #         continue
    #     similarity_tuple = []
    #     _exemplar = relable_from_zero(exemplar)

        for graph in graphs:
            if graph != exemplar:
                if len(graph.nodes) < 5 or len(graph.nodes) > 25:
                    continue
                _graph = relable_from_zero(graph)
                similarity = wl.compare(_exemplar, _graph, h=1, node_label=False)
                if similarity > threshold:
                    similarity_tuple.append((graph, similarity))


        similarity_tuple = sorted(similarity_tuple, key=lambda s: s[1], reverse=True)
        print(list(exemplar.nodes)[0], len(exemplar.nodes), len(similarity_tuple))
        shutil.rmtree('./data/user_study/' + dataset)
        os.mkdir('./data/user_study/' + dataset)
        save_json_graph(exemplar, './data/user_study/' + dataset + '/exemplar.json')
        i = 0
        for tuple in similarity_tuple:
            save_json_graph_with_attr(tuple[0], './data/user_study/' + dataset + '/target_' + str(i) + '.json', attr_name='similarity', attr_value=str(tuple[1]))
            i += 1

    # G = load_json_graph('./data/bn-mouse-kasthuri/graph-with-pos.json')
    # exemplar_nodes = [720,939,940,941,942,943,944] # road [320, 394, 395, 423, 425, 428, 430, 431, 434, 730, 739, 869]
    # exemplar = G.subgraph(exemplar_nodes)
    # save_json_graph(exemplar, './data/user_study/brain/exemplar.json')
    # targets = [
    #     [93, 587, 588, 590, 591, 592, 593, 595, 596, 597, 598, 599, 600, 601],
    #     [58, 208, 209, 211, 214, 216, 217, 218, 219, 220, 221, 224, 225, 226, 231, 232, 234, 235, 236, 238],
    #     [647, 748, 749, 750, 751, 752, 753, 754, 755],
    #     [728, 955, 956, 957, 958, 959, 960, 961],
    #
    #     # road
    #     # [235, 277, 278, 279, 280, 281, 282, 614, 850, 851],
    #     # [418, 419, 420, 422, 423, 426, 683, 686, 687, 688, 690, 692],
    #     # [616, 618, 619, 620, 621, 622, 645, 647, 648, 650, 995],
    #     # [486, 487, 488, 490, 491, 492, 493, 496, 506, 507, 509, 710]
    # ]
    # wl = GK_WL()
    # _exemplar = relable_from_zero(exemplar)
    # i = 0
    # for nodes in targets:
    #     graph = G.subgraph(nodes)
    #     _graph = relable_from_zero(graph)
    #     similarity = wl.compare(_exemplar, _graph, h=1, node_label=False)
    #     print(similarity)
    #     save_json_graph_with_attr(graph, './data/user_study/brain/target_' + str(i) + '.json', attr_name='similarity', attr_value=str(similarity))
    #     i += 1
