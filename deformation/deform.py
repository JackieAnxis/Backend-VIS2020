import numpy as np
import networkx as nx
from deformation.utils import load_json_graph, save_json_graph
from scipy import sparse
from scipy.optimize import least_squares
from deformation.Graph import Graph


def aligning(source_G, target_G, markers):
    # R = [[ s, h, x]
    #      [-h, s, y]
    #      [ 0, 0, 1]]
    # X = [s, h, tx, ty]
    # [x  y 0 1]
    # [y -x 1 0] dot [s h x y].T
    #     ...
    k = 4 # [s, h, tx, ty]
    M = np.zeros((markers.shape[0] * 2, k))
    C = np.zeros((markers.shape[0] * 2, 1))

    source_marker_nodes = source_G.nodes[markers[:, 0], :]
    target_marker_nodes = target_G.nodes[markers[:, 1], :]
    for i in range(0, markers.shape[0]):
        x = target_marker_nodes[i][0]
        y = target_marker_nodes[i][1]
        M[i * 2] = np.array([x, y, 1, 0])
        M[i * 2 + 1] = np.array([y, -x, 0, 1])
        C[i * 2:i * 2 + 2] = source_marker_nodes[i][:, np.newaxis]

    # X = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)[0]
    X = np.linalg.pinv(M).dot(C).flatten()

    s = X[0]
    h = X[1]
    t = X[2:]
    R = np.array([[s, h], [-h, s]])
    return R, t

def compute_laplacian_matrix(adj, nodes):
    n = nodes.shape[0]
    adj = adj.copy()
    w = 10
    for i in range(n):
        for j in range(i + 1, n):
            distance = (np.sum((nodes[i] - nodes[j]) ** 2))
            weight = 1
            if adj[i, j]:
                weight = w
            adj[i, j] = adj[j, i] = weight / distance
    L = np.diag(np.sum(adj, axis=1)) - adj
    return L

def deform_v1(G, target_pos):
    '''
    :param G:
    :param target_pos:
    :return:
    '''
    adj = G.compute_adjacent_matrix()
    V = G.nodes
    L = compute_laplacian_matrix(adj, V)

    res = np.inf
    for i in range(0, 100):
        n = V.shape[0]
        A = np.zeros([n * 2, 4])
        for j in range(n):
            A[j * 2] = [V[j, 0], V[j, 1], 1, 0]
            A[j * 2 + 1] = [V[j, 1], -V[j, 0], 0, 1]

        # Moore-Penrose Inversion
        A_pinv = np.linalg.pinv(A)

        _L = compute_laplacian_matrix(adj, V)
        M = np.zeros((2 * _L.shape[0], 2 * _L.shape[0]))
        index = np.arange(_L.shape[0], dtype=np.int32)
        for k in index:
            M[k * 2, index * 2] = _L[k]
            M[k * 2 + 1, index * 2 + 1] = _L[k]

        # T = np.diag()
        Delta = (L.dot(V))
        for k in index:
            M[k] -= np.dot([Delta[k, 0], Delta[k, 1], 0, 0], A_pinv)
            M[k * 2 + 1] -= np.dot([Delta[k, 1], -Delta[k, 0], 0, 0], A_pinv)

        # C = Delta.flatten('F').reshape(V.shape[0] * 2, 1)
        C = np.zeros((2 * _L.shape[0], 1))
        coe = np.zeros((len(target_pos) * 2, 2 * _L.shape[0]))
        pos = np.zeros((len(target_pos) * 2, 1))
        w = 1000
        j = 0
        for index in target_pos:
            coe[j, index*2] = 1 * w
            coe[j+1, index*2+1] = 1 * w
            pos[j] = target_pos[index][0] * w
            pos[j+1] = target_pos[index][1] * w
            j += 2
        M = np.vstack((M, coe))
        C = np.vstack((C, pos))
        X = sparse.linalg.lsqr(M, C, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)[0]
        _res = np.sum(np.abs(M.dot(X)[:, np.newaxis] - C))
        if _res >= res:
            break
        res = _res
        V = X.reshape((int(X.shape[0] / 2), 2))
        L = _L
    return V

    # def resSimXform(x, A, B):
    #     x_count = int(x.shape[0] / 2)
    #     V = x.reshape((x_count, 2))
    #     for index in target_pos:
    #         V[index] = target_pos[index]
    #     A = A[0:A.shape[1], :]
    #     B = B.reshape((x_count, 2))
    #     L = compute_laplacian_matrix(A, V)
    #     result = (L.dot(V) - B).flatten()
    #     return result
    #
    # b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='lm',
    #                   args=(A, B),
    #                   ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)

    print(np.sum(resSimXform(x0, A, B)))
    print(np.sum(resSimXform(b.x, A, B)))
    X = b.x.reshape((int(b.x.shape[0] / 2), 2))

    return X

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
    prefix = 'data/bn-mouse-kasthuri/'
    source = load_json_graph(prefix + 'source.json')
    target = load_json_graph(prefix + 'target0.json')
    _source = load_json_graph(prefix + '_source.json')
    source = Graph(source)
    target = Graph(target)
    markers = [[720, 676], [941, 853], [940, 855]]
    markers = np.array([[source.id2index[str(mk[0])], target.id2index[str(mk[1])]] for mk in markers])
    R, t = aligning(source, target, markers)
    target.nodes = target.nodes.dot(R.T) + t
    save_json_graph(nx.union(target.to_networkx(), source.to_networkx()), prefix + '_target0.json')
    target_pos = {}
    for mk in markers:
        s_id = mk[0]
        t_id = mk[1]
        target_pos[t_id] = source.nodes[s_id]

    X = deform_v1(target, target_pos)

    target.nodes = X
    R, t = aligning(source, target, markers)
    target.nodes = target.nodes.dot(R.T) + t

    save_json_graph(nx.union(target.to_networkx(), source.to_networkx()), prefix + '_target1.json')
    save_json_graph(target.to_networkx(), prefix + '_target2.json')

    exit()

    #####################
    prefix = 'data/bn-mouse-kasthuri/'
    # prefix = 'data/power-662-bus/'
    graph0 = load_json_graph(prefix + 'source.json')
    graph1 = load_json_graph(prefix + '_target0.json')
    H = load_json_graph(prefix + '_source.json')


    G0 = Graph(graph0)
    G0.normalize()

    G1 = Graph(graph1)
    G1.normalize()

    # markers = [[462, 246], [589, 220], [466, 181], [477, 195]]
    markers = [[ 720 , 676 ],
[ 940 , 855 ],
[ 941 , 853 ],
[ 942 , 848 ],
[ 943 , 851 ],
[ 944 , 849 ],]

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
    # H = graph0
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
    save_json_graph(nx.union(graph1, graph0), prefix + '_target3.json')