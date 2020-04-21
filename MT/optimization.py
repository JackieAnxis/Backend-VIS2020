import numpy as np

from MT.Graph import Graph
from MT.correspondence import compute_distance_matrix
from MT.deform import deform_v2, deform_v7
from models.utils import load_json_graph, save_json_graph


def magnify(source_G, rate):
    V = source_G.nodes
    n = V.shape[0]
    center = np.mean(V, axis=0)
    deformed_source_G = source_G.copy()
    for i in range(n):
        deformed_source_G.nodes[i] += (deformed_source_G.nodes[i] - center) * rate

    return deformed_source_G

def cal_radius(G):
    r = np.max(np.sqrt(np.sum((G.nodes[G.edges[:, 0]] - G.nodes[G.edges[:, 1]]) ** 2, axis=1)), axis=0)
    return r

def merge(G, subGs, iter=1000, alpha=1, beta=10, gamma=2000):
    target_pos = {}
    target_r = {}

    i = 0
    for subG in subGs:
        center = np.mean(subG.nodes, axis=0)
        R = np.sqrt(np.sum((subG.nodes - center) ** 2, axis=1))
        for id in subG.id2index:
            index = G.id2index[id]
            # if index not in target_pos:
            #     target_r[index] = []
            #     target_pos[index] = []
            # target_r[index].append(R[subG.id2index[id]])
            # target_pos[index].append(subG.nodes[subG.id2index[id]])
            target_r[index] = R[subG.id2index[id]]
            target_pos[index] = [subG.nodes[subG.id2index[id]], i]
        i += 1
        # r = cal_radius(subG)
        # for id in subG.id2index:
        #     index = G.id2index[id]
        #     if index not in target_pos:
        #         target_r[index] = []
        #         target_pos[index] = []
        #     target_r[index].append(r)
        #     target_pos[index].append(subG.nodes[subG.id2index[id]])

    # V = deform_v2(G, target_pos, iter, alpha, beta, gamma)
    # V = deform_v3(G, target_pos, iter, alpha, beta, gamma)
    # G0.nodes = V

    G1 = G.copy()
    for index in target_pos:
        # target_r[index] = np.max(target_r[index])
        # target_pos[index] = [np.mean(target_pos[index], axis=0), 1]
        G1.nodes[index] = target_pos[index][0]

    # return G1, G1

    G0 = G.copy()
    print('begin to compute distance matrix...')
    D = compute_distance_matrix(G1.nodes, G1.nodes)
    surroundings_index = np.nonzero(

        np.sum(D[list(target_pos.keys())] < np.array(list(target_r.values()))[:, np.newaxis], axis=0))[0]
    non_fixed_surroundings_index = np.nonzero(np.sum(D[list(target_pos.keys())] < np.array(list(target_r.values()))[:, np.newaxis] / 2, axis=0))[0]
    surroundings_id = [G.index2id[index] for index in surroundings_index]
    fixed_surroundings_id = [G.index2id[index] for index in non_fixed_surroundings_index]
    # fixed_surroundings_id_set = set(fixed_surroundings_id)
    print(len(target_pos), len(surroundings_id), len(fixed_surroundings_id))
    surroundings_G = Graph(G0.to_networkx().subgraph(surroundings_id))

    # d = np.min(D[list(target_pos.keys())][:, list(filter(lambda i: i not in target_pos, surroundings_index))], axis=0)
    # min_d = np.min(d)
    # max_d = np.max(d)
    new_target_pos = {}
    for index in non_fixed_surroundings_index:
        id = G.index2id[index]
        new_index = surroundings_G.id2index[id]
        if index not in target_pos:
            new_target_pos[new_index] = [G0.nodes[index], -1] # target_pos[index]
        else:
            new_target_pos[new_index] = target_pos[index]

    # V = surroundings_G.nodes
    V = deform_v7(surroundings_G, new_target_pos, iter, alpha, beta, gamma)

    # V = deform_v4(surroundings_G, new_target_pos, iter, alpha, beta, gamma)
    # V = deform_v3(surroundings_G, new_target_pos, iter, alpha, beta, gamma)
    # V = deform_v2(surroundings_G, target_pos, iter, alpha, beta, gamma)
    # surroundings_G.nodes = V
    # G0 = surroundings_G

    for index in range(V.shape[0]):
        id = surroundings_G.index2id[index]
        G0_index = G0.id2index[id]
        G0.nodes[G0_index] = V[index]
    return G0, G1

if __name__ == '__main__':
    def modify(source_G, source_nodes):
        V = source_G.nodes
        n = V.shape[0]
        center = np.mean(V, axis=0)
        radius = np.mean(np.sqrt(np.sum((V - center) ** 2, axis=1))) * 1.5
        interval = 2.0 * np.pi / n
        deformed_source_G = source_G.copy()
        for i in range(n):
            x = center[0] + radius * np.sin(interval * i)
            y = center[1] - radius * np.cos(interval * i)
            id = source_nodes[i]
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])

        return deformed_source_G

    # prefix = './data/mammalia-voles-plj-trapping-25/'
    # nodes_sets = [[122, 143, 50, 47, 51, 63, 48, 144],
    #               [129, 43, 42, 44, 40],
    #               [139, 135, 137, 140, 138, 141, 145, 131, 132, 136]]

    prefix = './data/mammalia-voles-bhp-trapping-60/'
    nodes_sets = [[46, 50, 3, 0, 4, 32, 49, 48]]
    graph = load_json_graph(prefix + 'graph-with-pos.bak.json')
    # path = nx.shortest_path(graph, source=10, target=23)
    # nodes_sets = [path]

    G = Graph(graph.subgraph([0, 1, 2, 3, 4, 7, 9, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]))
    save_json_graph(G.to_networkx(), prefix + 'graph-with-pos.json')
    subGs = []
    for nodes in nodes_sets:
        sub = graph.subgraph(nodes).copy()
        sub_G = Graph(sub)

        # subm = layout(sub)
        # sub_Gm = Graph(subm)
        sub_Gm = magnify(sub_G, 1.5)
        sub_Gm = modify(sub_G, nodes)

        # R, t = aligning(sub_G, sub_Gm, np.array([[index, index] for index in sub_G.index2id]))
        # sub_Gm.nodes = sub_Gm.nodes.dot(R.T) + t

        subGs.append(sub_Gm)
    G0, G1 = merge(G, subGs)
    save_json_graph(G0.to_networkx(), prefix + 'new.json')
    save_json_graph(G1.to_networkx(), prefix + '_new.json')