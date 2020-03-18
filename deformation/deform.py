import numpy as np
import networkx as nx
from scipy import sparse
from deformation.utils import load_json_graph, save_json_graph
from scipy.optimize import least_squares
from deformation.Graph import Graph
from deformation.correspondence import aligning, similarity_fitting

def deform(pos, target_sub_pos):
    '''
    :param pos: a dict
    :param target_sub_pos: a dict
    :return:
    '''
    dis_rate = np.zeros(shape=((len(pos) - len(target_sub_pos)) * len(target_sub_pos), 2))
    X = np.zeros(shape=(len(pos) - len(target_sub_pos), 2))
    ids = []
    row = 0
    for i in pos:
        if i not in target_sub_pos:
            X[row] = pos[i]
            ids.append(i)
            col = 0
            for j in target_sub_pos:
                dis = pos[i] - pos[j]
                dis_rate[row * len(target_sub_pos) + col] = dis
                col += 1
            row += 1

    M = np.zeros(shape=(len(target_sub_pos), 2))
    i = 0
    for id in target_sub_pos:
        M[i] = target_sub_pos[id]
        i += 1

    x0 = X.flatten()
    A = np.tile(M, (X.shape[0], 1))
    B = dis_rate

    print(ids)
    def resSimXform(x, A, B):
        x_count = int(x.shape[0] / 2)
        m_count = int(A.shape[0] / x_count)
        x = x.reshape((x_count, 2))
        X = np.tile(x, (1, m_count)).reshape((x_count * m_count, 2))
        vec = X - A
        result = np.sum((vec - B) ** 2, axis=1) / np.exp(np.sum(B**2, axis=1))
        return result

    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='lm',
                      args=(A, B),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    print(np.sum(resSimXform(x0, A, B)))
    print(np.sum(resSimXform(b.x, A, B)))
    X = b.x.reshape((int(b.x.shape[0] / 2), 2))

    for i in range(0, X.shape[0]):
        id = ids[i]
        target_sub_pos[id] = X[i]
    return target_sub_pos

def deform_v0(pos, target_sub_pos):
    '''
    :param pos: a dict
    :param target_sub_pos: a dict
    :return:
    '''
    dis_rate = np.zeros(shape=(len(pos) - len(target_sub_pos), len(target_sub_pos)))
    X = np.zeros(shape=(len(pos) - len(target_sub_pos), 2))
    ids = []
    row = 0
    for i in pos:
        if i not in target_sub_pos:
            X[row] = pos[i]
            ids.append(i)
            base_dis = 1
            col = 0
            for j in target_sub_pos:
                dis = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                if col == 0:
                    base_dis = dis
                dis_rate[row, col] = dis # / base_dis
                col += 1
            row += 1

    M = np.zeros(shape=(len(target_sub_pos), 2))
    i = 0
    for id in target_sub_pos:
        M[i] = target_sub_pos[id]
        i += 1

    # print(X - M)
    x0 = X.flatten()
    A = np.tile(M, (X.shape[0], 1))
    B = dis_rate.flatten()

    print(ids)
    def resSimXform(x, A, B):
        x_count = int(x.shape[0] / 2)
        m_count = int(A.shape[0] / x_count)
        x = x.reshape((x_count, 2))
        X = np.tile(x, (1, m_count)).reshape((x_count * m_count, 2))
        vec = X - A
        base_vec = vec[np.arange(0, x.shape[0]) * m_count]
        dis = np.sqrt(np.sum(vec**2, axis=1))
        base_dis = np.tile(np.sqrt(np.sum(base_vec ** 2, axis=1)),(m_count, 1)).flatten('F')
        # result = dis / base_dis - B
        result = (dis - B) / B
        return result

    b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='lm',
                      args=(A, B),
                      ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    print(np.sum(resSimXform(x0, A, B)))
    print(np.sum(resSimXform(b.x, A, B)))
    X = b.x.reshape((int(b.x.shape[0] / 2), 2))

    for i in range(0, X.shape[0]):
        id = ids[i]
        target_sub_pos[id] = X[i]
    return target_sub_pos


if __name__ == '__main__':
    prefix = 'data/power-662-bus/'
    graph0 = load_json_graph(prefix + 'source.json')
    graph1 = load_json_graph(prefix + '_target0.json')
    H = load_json_graph(prefix + '_source.json')


    G0 = Graph(graph0)
    G0.normalize()

    G1 = Graph(graph1)
    G1.normalize()

    markers = [[462,246],
     [466,182],
    [476,181],
    [477,194],
    [574,183],
    [575,222],
    [588,221],
    [589,220]]

    markers = [[str(marker[0]), str(marker[1])] for marker in markers]
    R, t = aligning(G0, G1, np.array([[G0.id2index[marker[0]], G1.id2index[marker[1]]] for marker in markers]))
    G1.nodes = G1.nodes.dot(R.T) + t

    graph0 = G0.to_networkx()
    graph1 = G1.to_networkx()

    h = Graph(H)
    h.normalize()
    H = h.to_networkx()

    G = nx.union(graph0, graph1)

    save_json_graph(G, prefix + '_target1.json')

    G = graph1
    pos = {}
    for node in G.nodes:
        pos[node] = np.array([G.nodes[node]['x'], G.nodes[node]['y']])

    target_sub_pos = {}
    for marker in markers:
        target_sub_pos[marker[1]] = np.array([H.nodes[marker[0]]['x'], H.nodes[marker[0]]['y']])
        # target_sub_pos[node] = np.array([H.nodes[node]['x'], H.nodes[node]['y']])

    # res = deform_v0(pos, target_sub_pos)
    res = deform(pos, target_sub_pos)

    for id in res:
        G.nodes[id]['x'], G.nodes[id]['y'] = res[id]
        if id in graph1.nodes:
            graph1.nodes[id]['x'], graph1.nodes[id]['y'] = res[id]



    save_json_graph(G, prefix + '_target2.json')
    save_json_graph(graph1, prefix + '_target3.json')