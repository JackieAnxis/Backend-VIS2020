# -*- coding: UTF-8
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from deformation.Graph import Graph
from deformation.utils import load_json_graph, save_json_graph

def fitting(source_G, target_G, markers):
    source_marker_nodes = source_G.nodes[markers[:, 0], :].T  # 2 * n
    target_marker_nodes = target_G.nodes[markers[:, 1], :].T  # 2 * n
    source_center = np.mean(source_marker_nodes, axis=1)
    target_center = np.mean(target_marker_nodes, axis=1)

    h = 0
    s = 1
    t = (source_center - target_center)

    # R.dot(X)+t = X'
    x0 = np.zeros((4,))  # R.size / 2 + t.size / 2
    x0[0] = h
    x0[1] = s
    x0[2:4] = t

    def resSimXform(x, A, B):
        h = x[0]
        s = x[1]
        t = x[2:4]
        R = np.array([[s, h], [-h, s]])
        rot_A = R.dot(A) + t[:, np.newaxis]
        result = np.sqrt(np.sum((B - rot_A) ** 2, axis=0))
        return result

    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='lm',
                      args=(target_marker_nodes, source_marker_nodes),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    h = b.x[0]
    s = b.x[1]
    t = b.x[2:4]
    R = np.array([[s, h], [-h, s]])

    return R, t


def length_minimize(source_G, target_G, marker):
    x0 = target_G.nodes.flatten()
    x0[marker[:, 1] * 2] = source_G.nodes[marker[:, 0]][:, 0]
    x0[marker[:, 1] * 2 + 1] = source_G.nodes[marker[:, 0]][:, 1]

    pair_index = []
    pair_expected_length = []
    for i in range(0, target_G.nodes.shape[0]):
        for j in range(i + 1, target_G.nodes.shape[0]):
            if target_G.rawgraph.has_edge(target_G.index2id[i], target_G.index2id[j]):
            # if i in marker[:, 1] and j in marker[:, 1]:
            #     source_marker_id_i = marker[:, 0][list(marker[:, 1]).index(i)]
            #     source_marker_id_j = marker[:, 0][list(marker[:, 1]).index(j)]
            #     pair_expected_length.append(source_G.nodes[source_marker_id_i] - source_G.nodes[source_marker_id_j])
            #     pair_index.append([i, j, 1])
            # else:
                pair_expected_length.append(target_G.nodes[i] - target_G.nodes[j])
                pair_index.append([i, j, 1])

    pair_index = np.array(pair_index, dtype=np.int32)
    pair_expected_length = np.sqrt(np.sum(np.array(pair_expected_length) ** 2, axis=1)) * pair_index[:, 2]

    def resSimXform(x, A, B):
        x = x.reshape(((int(x.shape[0] / 2), 2)))
        edge_lengths = np.sqrt(np.sum((x[A[:, 0]] - x[A[:, 1]]) ** 2, axis=1))
        E1 = np.sum((edge_lengths / edge_lengths[0] - B / B[0]) ** 2)
        E2 = np.sum(np.sqrt(np.sum((x[marker[:, 1]] - source_G.nodes[marker[:, 0]])**2, axis=1)))
        return E1 + E2

    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='trf', args=(pair_index, pair_expected_length),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    return b.x

if __name__ == '__main__':
    prefix = './data/power-662-bus/'

    source = load_json_graph(prefix + 'source.json')
    target = load_json_graph(prefix + 'target0.json')

    markers = np.array([[575, 257], [574, 222], [476, 245], [588, 181]])

    source_G = Graph(source)
    target_G = Graph(target)
    markers[:, 0] = np.array([source_G.id2index[str(marker)] for marker in markers[:, 0]])
    markers[:, 1] = np.array([target_G.id2index[str(marker)] for marker in markers[:, 1]])
    print('fitting...')
    R, t = fitting(source_G, target_G, markers)
    target_G.nodes = target_G.nodes.dot(R.T) + t
    print('Minimize...')
    X = length_minimize(source_G, target_G, markers)
    target_G.nodes = X.reshape((target_G.nodes.shape[0], 2))
    print('!')
    for node in target.nodes:
        index = target_G.id2index[str(node)]
        target.nodes[node]['x'] = target_G.nodes[index][0]
        target.nodes[node]['y'] = target_G.nodes[index][1]

    save_json_graph(target, prefix + '_target0.json')