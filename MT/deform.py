import numpy as np
from scipy import sparse

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
            weight = 1
            if adj[i, j]:
                weight = edge_weight
            adj[i, j] = adj[j, i] = weight / (distance+0.001)
    L = np.diag(np.sum(adj, axis=1)) - adj
    return L

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

def non_rigid_registration(source_G, target_G, markers, iter=10):
    target_G = target_G.copy()

    R, t = aligning(source_G, target_G, markers)
    target_G.nodes = target_G.nodes.dot(R.T) + t

    target_pos = {}
    for mk in markers:
        s_id = mk[0]
        t_id = mk[1]
        target_pos[t_id] = source_G.nodes[s_id]

    X = deform(target_G, target_pos, iter)
    target_G.nodes = X
    return target_G