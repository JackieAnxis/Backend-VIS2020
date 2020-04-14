import numpy as np
import networkx as nx
from models.utils import save_json_graph
from MT.Graph import Graph
from scipy import sparse
from MT.correspondence import compute_distance_matrix

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

def compute_laplacian_matrix(adj, nodes, w=10):
    n = nodes.shape[0]
    adj = adj.copy()
    edge_weight = w
    for i in range(n):
        for j in range(i + 1, n):
            distance = (np.sum((nodes[i] - nodes[j]) ** 2))
            # distance = np.sqrt(np.sum((nodes[i] - nodes[j]) ** 2))
            weight = 1
            if adj[i, j]:
                weight = edge_weight
            adj[i, j] = adj[j, i] = weight / (distance**3+0.001)
    L = np.diag(np.sum(adj, axis=1)) - adj
    return L

def deform_v2(G, target_pos, iter=1000, alpha=100, beta=5, gamma=200):
    '''
    combine minimize the distance and direction difference,
    separate direction and distance from the direction protection.
    distance: L_w
    :param G: Graph object
    :param target_pos: dict, index 2 ndarray
    :return:
    '''
    def laplacian(A, D, w=1):
        L = (A * w + 1 - np.eye((A.shape[0]))) / (D + 0.00001)
        # L = A * w / (D + 0.00001)
        L = np.diag(np.sum(L, axis=0)) - L
        return L

    V = G.nodes
    n = V.shape[0]
    adj = G.compute_adjacent_matrix()
    D = compute_distance_matrix(V, V)
    D2 = D**2
    L = laplacian(adj, D2)
    Lwd = laplacian(adj, D2*D)

    C1 = Lwd.dot(V).flatten()[:, np.newaxis]
    V0 = V.copy()
    V = V.copy()
    coe = np.zeros((len(target_pos) * 2, 2 * n))
    pos = np.zeros((len(target_pos) * 2, 1))
    j = 0
    for index in target_pos:
        weight = target_pos[index][1] # weight
        coe[j, index * 2] = weight
        coe[j + 1, index * 2 + 1] = weight
        pos[j] = target_pos[index][0][0] * weight
        pos[j + 1] = target_pos[index][0][1] * weight
        # V[index] = target_pos[index]
        j += 2

    N = np.arange(0, n)

    def stress(L, V, V0, alpha, beta, gamma):
        LEFT = np.tile(V, (V.shape[0], 1))
        LEF0 = np.tile(V0, (V0.shape[0], 1))
        RIGT = np.tile(V, (1, V.shape[0])).flatten().reshape(V.shape[0]**2, 2)
        RIG0 = np.tile(V0, (1, V0.shape[0])).flatten().reshape(V0.shape[0] ** 2, 2)
        VECT = LEFT - RIGT
        VEC0 = LEF0 - RIG0
        DIST = np.sqrt(np.sum(VECT**2, axis=1))
        DIS0 = np.sqrt(np.sum(VEC0**2, axis=1))
        DIST[DIST == 0] += 0.0001
        DIS0[DIS0 == 0] += 0.0001
        NORM = VECT / DIST[:, np.newaxis]
        NOR0 = VEC0 / DIS0[:, np.newaxis]
        WGHT = -1 * ((1 - np.eye((L.shape[0]))) * L).flatten()
        STRS = alpha * WGHT * np.sum((NORM - NOR0)**2, axis=1) + beta * WGHT * (DIST - DIS0)**2

        return np.sum(STRS)

    # str = stress(L, np.array(
    #     [[0, 0], [np.sqrt(3/20)+1, 1/2-np.sqrt(3 / 5)], [2, 1], [3 - np.sqrt(3/20), 1/2-np.sqrt(3 / 5)], [4, 0]]), V0,
    #        alpha, beta, gamma)
    strs = strs_ = 0
    eps = 10**(-8)
    k = 0
    for k in range(iter):
        D_t = compute_distance_matrix(V, V)
        Lwd_t = laplacian(adj, D2 * D_t)
        Lwdv_t = laplacian(adj, D * D_t)

        M1 = np.zeros((2 * n, 2 * n))
        M1[np.ix_(N * 2, N * 2)] = Lwd_t
        M1[np.ix_(N * 2 + 1, N * 2 + 1)] = Lwd_t

        M2 = np.zeros((2 * n, 2 * n))
        M2[np.ix_(N * 2, N * 2)] = L
        M2[np.ix_(N * 2 + 1, N * 2 + 1)] = L

        C2 = Lwdv_t.dot(V).flatten()[:, np.newaxis]

        M = np.vstack((M1 * alpha, M2 * beta, coe * gamma))
        C = np.vstack((C1 * alpha, C2 * beta, pos * gamma))

        X = sparse.linalg.lsqr(M, C, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)[0]
        V_ = X.reshape((int(X.shape[0] / 2), 2))
        strs_ = stress(L, V_, V0, alpha, beta, gamma)
        if k > 0:
            if (strs - strs_) / strs < eps:
                break
        strs = strs_
        V = V_

    print("residual of iteration", k, strs_)
    return V

def deform_v3(G, target_pos, iter=1000, alpha=100, beta=5, gamma=200):
    '''
    Stree majorization, Intelligent Graph Layout Using Many Users’ Input
    distance: L_w
    :param G: Graph object
    :param target_pos: dict, index 2 ndarray
    :return:
    '''
    def laplacian(A, D, w=1):
        L = (A * w + 1 - np.eye((A.shape[0]))) / D
        L = np.diag(np.sum(L, axis=0)) - L
        return L

    V = G.nodes
    n = V.shape[0]
    N = np.arange(0, n)
    eps = 10 ** (-4)

    D = compute_distance_matrix(V, V) + eps
    adj = np.ones((G.nodes.shape[0], G.nodes.shape[0])) # G.compute_adjacent_matrix()

    D2 = D**2
    target_idxs = list(target_pos.keys())
    W = D2[target_idxs] / np.max(D2)
    adj[target_idxs] = 5 / W
    adj[:, target_idxs] = 5 / W.T

    V = V.copy()
    for index in target_pos:
        weight = target_pos[index][1] # weight
        # V[index] = target_pos[index][0]
        for jndex in target_pos:
            if jndex != index:
                adj[index, jndex] = (gamma * weight * target_pos[jndex][1])
                D[index, jndex] = np.sqrt(np.sum((target_pos[index][0] - target_pos[jndex][0]) ** 2))
            else:
                D[index, jndex] = eps


    L = laplacian(adj, 1)

    def stress(L, V, D0):
        D = compute_distance_matrix(V, V)
        STRS = np.sum((D - D0)**2 * (-L))
        return STRS

    strs = strs_ = 0
    k = 0
    for k in range(iter):
        D_t = compute_distance_matrix(V, V) + eps
        Lwdv_t = laplacian(adj, (D_t / D))

        M = np.zeros((2 * n, 2 * n))
        M[np.ix_(N * 2, N * 2)] = L
        M[np.ix_(N * 2 + 1, N * 2 + 1)] = L

        C = Lwdv_t.dot(V).flatten()[:, np.newaxis]

        X = sparse.linalg.lsqr(M, C, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)[0]
        V_ = X.reshape((int(X.shape[0] / 2), 2))
        strs_ = stress(L, V_, D)
        if k > 0:
            if (strs - strs_) / strs < eps:
                break
        strs = strs_
        V = V_
    # T = np.array([[0, 0], [np.sqrt(3/20)+1, 1/2-np.sqrt(3 / 5)], [2, 1], [3 - np.sqrt(3/20), 1/2-np.sqrt(3 / 5)], [4, 0]])
    # s0 = stress(L, T, D, alpha, beta)
    # s1 = stress(L, V, D, alpha, beta)
    print("residual of iteration", k, strs_)
    return V

def deform_v4(G, target_pos, iter=1000, alpha=100, beta=5, gamma=200):
    '''

    :param G: Graph object
    :param target_pos: dict, index 2 ndarray
    :return:
    '''

    V = G.nodes
    n = V.shape[0]
    N = np.arange(0, n)
    eps = 10e-6

    # D = compute_distance_matrix(V, V) + eps
    adj = G.compute_adjacent_matrix()

    w = 10
    C1 = np.zeros((G.nodes.shape[0] * 2, 1))
    for i in range(G.nodes.shape[0]):
        sum = np.zeros((2))
        for j in range(G.nodes.shape[0]):
            if j != i:
                if i in target_pos and j in target_pos:
                    v = target_pos[i][0] - target_pos[j][0]
                else:
                    v = G.nodes[i] - G.nodes[j]
                if i in target_pos:
                    v *= w
                if j in target_pos:
                    v *= w

                if adj[i, j]:
                    sum += v * 2
                else:
                    sum += v

        C1[i * 2:(i + 1) * 2] = sum[:, np.newaxis]

    adj += 1
    adj -= np.eye(n)
    adj[list(target_pos.keys()), :] *= w
    adj[:, list(target_pos.keys())] *= w

    L = np.diag(np.sum(adj, axis=0)) - adj

    M1 = np.zeros((G.nodes.shape[0] * 2, G.nodes.shape[0] * 2))
    M1[np.ix_(np.arange(G.nodes.shape[0]) * 2, np.arange(G.nodes.shape[0]) * 2)] = L
    M1[np.ix_(np.arange(G.nodes.shape[0]) * 2 + 1, np.arange(G.nodes.shape[0]) * 2 + 1)] = L

    C2 = np.zeros((len(target_pos) * 2, 1))
    M2 = np.zeros((len(target_pos) * 2, G.nodes.shape[0] * 2))
    target_pos = [(k, v) for k, v in target_pos.items()]
    for i in range(len(target_pos)):
        index = target_pos[i][0]
        pos = target_pos[i][1][0]
        weight = target_pos[i][1][1]
        M2[i * 2, index * 2] = 1
        M2[i * 2 + 1, index * 2 + 1] = 1
        C2[i * 2:(i + 1) * 2] = pos[:, np.newaxis]

    M = np.vstack((M1, M2 * gamma))
    C = np.vstack((C1, C2 * gamma))

    X = sparse.linalg.lsqr(M, C, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)[0]
    V_ = X.reshape((int(X.shape[0] / 2), 2))
    return V_


def deform(G, target_pos, iter=100, alpha=20, beta=0, gamma=200):
    '''
    combine minimize the distance and direction difference.
    :param G: Graph object
    :param target_pos: dict, index 2 ndarray
    :return:
    '''
    V = G.nodes
    n = V.shape[0]
    adj = G.compute_adjacent_matrix()
    L = compute_laplacian_matrix(adj, V)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = np.sqrt(np.sum((V[i] - V[j]) ** 2))

    LD = L * D

    # C = np.zeros((n * 2, 1))
    coe = np.zeros((len(target_pos) * 2, 2 * n))
    pos = np.zeros((len(target_pos) * 2, 1))
    j = 0
    for index in target_pos:
        coe[j, index * 2] = 1
        coe[j + 1, index * 2 + 1] = 1
        pos[j] = target_pos[index][0]
        pos[j + 1] = target_pos[index][1]
        j += 2

    _res = res = np.inf
    N = np.arange(0, n)
    k = 0
    for k in range(iter):
        C1 = L.dot(V).flatten()[:, np.newaxis]
        M1 = np.zeros((2 * n, 2 * n))
        M1[np.ix_(N * 2, N * 2)] = L
        M1[np.ix_(N * 2 + 1, N * 2 + 1)] = L

        D = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                D[i, j] = D[j, i] = (np.sqrt(np.sum((V[i] - V[j]) ** 2)) + 0.001)
        LDS = LD / D
        LDS += -np.diag(np.sum(LDS, axis=1))
        C2 = LDS.dot(V).flatten()[:, np.newaxis]
        M2 = np.zeros((2 * n, 2 * n))
        M2[np.ix_(N * 2, N * 2)] = L
        M2[np.ix_(N * 2 + 1, N * 2 + 1)] = L

        M = np.vstack((M1 * alpha, M2 * beta, coe * gamma))
        C = np.vstack((C1 * alpha, C2 * beta, pos * gamma))

        X = sparse.linalg.lsqr(M, C, iter_lim=5000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)[0]
        _res = np.sum(np.abs(M.dot(X)[:, np.newaxis] - C))
        if _res >= res:
            break
        res = _res
        V = X.reshape((int(X.shape[0] / 2), 2))

    print("residual of iteration", k, _res)
    return V

def non_rigid_registration(source_G, target_G, markers, alpha=10, beta=10, gamma=1000, iter=1000):
    _target_G = target_G.copy()

    R, t = aligning(source_G, _target_G, markers)
    _target_G.nodes = _target_G.nodes.dot(R.T) + t

    target_pos = {}
    for mk in markers:
        s_index = mk[0]
        t_index = mk[1]
        target_pos[t_index] = []
        target_pos[t_index].append(source_G.nodes[s_index])
        target_pos[t_index].append(1) # weight


    X = deform_v2(_target_G, target_pos, iter, alpha, beta, gamma)
    _target_G.nodes = X

    # _R, _t = aligning(target_G, _target_G, np.array([[index, index] for index in target_G.index2id]))
    # _target_G.nodes = _target_G.nodes.dot(_R.T) + _t
    # _target_G.nodes = (_target_G.nodes - t).dot(np.linalg.inv(R).T)  ############

    return _target_G

if __name__ == '__main__':
    prefix = './data/test/'
    source = nx.Graph()
    source.add_edges_from([[0,1], [1,2], [2,3], [3,4]])
    source.nodes[0]['x'] = 0.0
    source.nodes[0]['y'] = 0.0
    source.nodes[1]['x'] = 1.0
    source.nodes[1]['y'] = 1.0
    source.nodes[2]['x'] = 2.0
    source.nodes[2]['y'] = 2.0
    source.nodes[3]['x'] = 3.0
    source.nodes[3]['y'] = 1.0
    source.nodes[4]['x'] = 4.0
    source.nodes[4]['y'] = 0.0

    source_G = Graph(source)
    target_pos = {
        0: [np.array([0.0, 0.0]), 1],
        2: [np.array([2.0, 1.0]), 1],
        4: [np.array([4.0, 0.0]), 1],
    }


    G0 = source_G.copy()
    V = deform_v2(source_G, target_pos, alpha=0, beta=10)
    print(V)
    G0.nodes = V

    G1 = source_G.copy()
    V = deform_v3(source_G, target_pos, alpha=10, beta=1)
    print(V)
    G1.nodes = V
    save_json_graph(source_G.to_networkx(), prefix + 'result/target0.json')
    save_json_graph(G0.to_networkx(), prefix + 'result/target1.json')
    save_json_graph(G1.to_networkx(), prefix + 'result/target2.json')