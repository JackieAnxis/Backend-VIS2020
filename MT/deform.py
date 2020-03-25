import numpy as np
import networkx as nx
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

def deform_v2(G, target_pos, iter=1000, alpha=10, beta=10, gamma=200):
    '''
    combine minimize the distance and direction difference,
    separate direction and distance from the direction protection.
    distance: L_w
    :param G: Graph object
    :param target_pos: dict, index 2 ndarray
    :return:
    '''
    def laplacian(A, D, w=10):
        L = (A * w + 1 - np.eye((A.shape[0]))) / (D + 0.00001)
        # L = A * w / (D + 0.00001)
        L = np.diag(np.sum(L, axis=0)) - L
        return L

    V = G.nodes
    n = V.shape[0]
    adj = G.compute_adjacent_matrix()
    D = compute_distance_matrix(V, V)
    L_1 = laplacian(adj, D)
    L_2 = laplacian(adj, D**2)
    L_3 = laplacian(adj, D**3)

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
    C1 = L_3.dot(V).flatten()[:, np.newaxis]
    k = 0
    for k in range(iter):
        new_D = compute_distance_matrix(V, V)
        new_L_3 = laplacian(adj, new_D * (D**2))
        M1 = np.zeros((2 * n, 2 * n))
        M1[np.ix_(N * 2, N * 2)] = new_L_3
        M1[np.ix_(N * 2 + 1, N * 2 + 1)] = new_L_3
        M2 = np.zeros((2 * n, 2 * n))
        M2[np.ix_(N * 2, N * 2)] = L_3
        M2[np.ix_(N * 2 + 1, N * 2 + 1)] = L_3

        C2 = new_L_3.dot(V).flatten()[:, np.newaxis]

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

def non_rigid_registration(source_G, target_G, markers):
    target_G = target_G.copy()

    R, t = aligning(source_G, target_G, markers)
    target_G.nodes = target_G.nodes.dot(R.T) + t

    target_pos = {}
    for mk in markers:
        s_id = mk[0]
        t_id = mk[1]
        target_pos[t_id] = source_G.nodes[s_id]

    X = deform_v2(target_G, target_pos)
    target_G.nodes = X
    return target_G

if __name__ == '__main__':
    prefix = './data/bn-mouse-kasthuri/'
    source = nx.Graph()
    source.add_edges_from([[0,1], [1,2], [2,3], [3,4]])
    source.nodes[0]['x'] = 0
    source.nodes[0]['y'] = 0
    source.nodes[1]['x'] = 1
    source.nodes[1]['y'] = 1
    source.nodes[2]['x'] = 2
    source.nodes[2]['y'] = 2
    source.nodes[3]['x'] = 3
    source.nodes[3]['y'] = 1
    source.nodes[4]['x'] = 4
    source.nodes[4]['y'] = 0

    source_G = Graph(source)
    target_pos = {
        0: np.array([0, 0]),
        2: np.array([2,1]),
        4: np.array([4, 0]),
    }
    V = deform_v2(source_G, target_pos)
    print(V)