import numpy as np
import networkx as nx
from models.utils import save_json_graph, load_json_graph
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

    source_marker_nodes = source_G.nodes[markers[:, 0], :].copy()
    target_marker_nodes = target_G.nodes[markers[:, 1], :].copy()
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
    def laplacian(A, D, w=10):
        L = (A * w + 1 - np.eye((A.shape[0]))) / (D + 0.00001)
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
        weight = 1 # target_pos[index][1] # weight
        coe[j, index * 2] = weight
        coe[j + 1, index * 2 + 1] = weight
        pos[j] = target_pos[index][0][0] * weight
        pos[j + 1] = target_pos[index][0][1] * weight
        V[index] = target_pos[index][0] ######
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

def deform_v7(G, target_pos, iter=1000, alpha=100, beta=5, gamma=200):
    '''
    repetition Intelligent Graph Layout Using Many Users’ Input
    distance: L_w
    :param G: Graph object
    :param target_pos: dict, index 2 ndarray
    :return:
    '''
    V = G.nodes
    V0 = V.copy()
    n = V.shape[0]
    N = np.arange(n)
    eps = 1e-5

    VD0 = np.tile(V, (1, n)) - np.tile(V.flatten(), (n, 1)).reshape((n, n*2)) # initial vector direction
    # VD0[i, j*2:(j+1)*2] = V[i] - V[j]
    VD0X = VD0[:, N * 2]
    VD0Y = VD0[:, N * 2 + 1]
    D0 = np.sqrt(VD0X**2 + VD0Y**2) # initial distance
    # D0 = compute_distance_matrix(V, V)
    VD0X /= (D0 + eps)
    VD0X[N, N] = 0
    VD0Y /= (D0 + eps)
    VD0Y[N, N] = 0
    # D0 *= 2
    m = 0  # node count in substrucutres
    target_pos_list = []
    substructure_nodes = []
    for k in target_pos:
        v = target_pos[k]
        target_pos_list.append((k,v))
        if v[1] >= 0: # it is a substructure, not fixed surrounding nodes
            m += 1
            substructure_nodes.append(k)
    M = np.arange(m)
    substructure_nodes = np.array(substructure_nodes)

    TVDX = VD0X.copy()  # target vectors direction of x
    TVDY = VD0Y.copy()  # target vectors direction of y
    TD = D0.copy()
    # TD *= 1.5
    wS = np.zeros((m, m))  # np.ones((m, m)) - np.eye(m)  # without bias on links or distance
    for i in range(len(substructure_nodes)):
        idxi = substructure_nodes[i]
        posi = target_pos[idxi][0]
        subi = target_pos[idxi][1]
        for j in range(i + 1, len(substructure_nodes)):
            idxj = substructure_nodes[j]
            posj = target_pos[idxj][0]
            subj = target_pos[idxj][1]
            v = posi - posj
            d = 0
            if subi == subj: # and G.index2id[idxj] in G.rawgraph[G.index2id[idxi]]:  # in the same substrucutre
                wS[i, j] = 1
                wS[j, i] = 1
                d = np.linalg.norm(v)
            else:
                for pos_i in [posi, G.nodes[idxi]]:
                    for pos_j in [posj, G.nodes[idxj]]:
                        v_tmp = pos_i - pos_j
                        d_tmp = np.linalg.norm(v_tmp) # * 1.5
                        if d_tmp > d:
                            v = v_tmp
                            d = d_tmp
            vd = v / d
            TVDX[idxi, idxj] = vd[0]
            TVDY[idxi, idxj] = vd[1]
            TD[idxi, idxj] = TD[idxj, idxi] = d


    TDS = TD[substructure_nodes, :][:, substructure_nodes]
    # wS /= (TDS**2 + eps) # give a bias on distance
    LwS = (-wS + np.diag(np.sum(wS, axis=0)))
    # LwSD = np.zeros((2 * m, 2 * n)) # double LwS, for x and y
    # LwSD[np.ix_(M * 2, substructure_nodes * 2)] = LwS
    # LwSD[np.ix_(M * 2 + 1, substructure_nodes * 2 + 1)] = LwS
    LwSD = np.zeros((2 * n, 2 * n))  # double LwS, for x and y
    LwSD[np.ix_(substructure_nodes * 2, substructure_nodes * 2)] = LwS
    LwSD[np.ix_(substructure_nodes * 2 + 1, substructure_nodes * 2 + 1)] = LwS

    for index in target_pos:
        V[index] = target_pos[index][0]

    VD = np.tile(V, (1, n)) - np.tile(V.flatten(), (n, 1)).reshape((n, n * 2))  # initial vector direction
    VDX = VD[:, N * 2]
    VDY = VD[:, N * 2 + 1]
    D = np.sqrt(VDX ** 2 + VDY ** 2)  # initial distance
    VDX /= (D + eps)
    VDX[N, N] = 0
    VDY /= (D + eps)
    VDY[N, N] = 0
    D = np.asarray(D, dtype=np.float32)

    # gdM = G.compute_graph_distance_matrix()
    # w = ((3 - gdM) > 0) * (3 - gdM)
    # w = G.compute_adjacent_matrix() + np.ones((n, n)) - np.eye(n)
    w = np.ones((n, n)) - np.eye(n)  # without bias on links or distance
    w /= (D**2 + eps) # give a bias on distance
    Lw = (-w + np.diag(np.sum(w, axis=0)))
    LwD = np.zeros((2 * n, 2 * n))  # double Lw, for x and y
    LwD[np.ix_(N * 2, N * 2)] = Lw
    LwD[np.ix_(N * 2 + 1, N * 2 + 1)] = Lw
    
    ####
    wVD = w # .copy()
    wVD[np.ix_(substructure_nodes, substructure_nodes)] = 0 # -= wS
    
    C2 = np.vstack((np.sum(TVDX * wVD, axis=1), np.sum(TVDY * wVD, axis=1))).flatten('F')[:, np.newaxis] #[x,y,...].T
    ####

    coe = np.zeros((len(substructure_nodes) * 2, 2 * n))
    pos = np.zeros((len(substructure_nodes) * 2, 1))
    j = 0
    for index in substructure_nodes:
        coe[j, index * 2] = 1
        coe[j + 1, index * 2 + 1] = 1
        pos[j] = target_pos[index][0][0]
        pos[j + 1] = target_pos[index][0][1]
        j += 2

    M1 = (LwD + LwSD * gamma)

    V = V0

    for k in range(iter):
        Lwd = -w * (TD + eps) / (D + eps)
        Lwd[N, N] = 0
        Lwd[N, N] = -np.sum(Lwd, axis=0)
        LwdD = np.zeros((n * 2, n * 2))
        LwdD[np.ix_(N * 2, N * 2)] = Lwd
        LwdD[np.ix_(N * 2 + 1, N * 2 + 1)] = Lwd

        DS = D[substructure_nodes, :][:, substructure_nodes]
        LwdS = -wS * (TDS + eps) / (DS + eps)
        LwdS[M, M] = 0
        LwdS[M, M] = -np.sum(LwdS, axis=0)
        # LwdSD = np.zeros((m * 2, n * 2))
        # LwdSD[np.ix_(M * 2, substructure_nodes * 2)] = LwdS
        # LwdSD[np.ix_(M * 2 + 1, substructure_nodes * 2 + 1)] = LwdS
        # C1 = np.vstack((LwdD, LwdSD * gamma)).dot(V.flatten())[:, np.newaxis]
        LwdSD = np.zeros((n * 2, n * 2))
        LwdSD[np.ix_(substructure_nodes * 2, substructure_nodes * 2)] = LwdS
        LwdSD[np.ix_(substructure_nodes * 2 + 1, substructure_nodes * 2 + 1)] = LwdS
        C1 = (LwdD + LwdSD * gamma).dot(V.flatten())[:, np.newaxis]

        wVD = w / (D + eps)
        wVD[N, N] = 0
        LwVD = (-wVD + np.diag(np.sum(wVD, axis=0)))
        LwVDD = np.zeros((n * 2, n * 2))
        LwVDD[np.ix_(N * 2, N * 2)] = LwVD
        LwVDD[np.ix_(N * 2 + 1, N * 2 + 1)] = LwVD
        M2 = LwVDD

        M3 = np.vstack((M1 * beta, M2 * alpha)) #, coe * gamma))
        C3 = np.vstack((C1 * beta, C2 * alpha)) #, pos * gamma))

        X = sparse.linalg.lsqr(M3, C3, iter_lim=50000, atol=1e-8, btol=1e-8, conlim=1e7, show=False, x0=V.flatten())[0]
        V_ = X.reshape((int(X.shape[0] / 2), 2))
        VD = None
        D = None
        VD = np.tile(V_, (1, n)) - np.tile(V_.flatten(), (n, 1)).reshape((n, n * 2))  # initial vector direction
        VDX = VD[:, N * 2]
        VDY = VD[:, N * 2 + 1]
        D = np.sqrt(VDX ** 2 + VDY ** 2)  # initial distance
        VDX /= (D + eps)
        VDX[N, N] = 0
        VDY /= (D + eps)
        VDY[N, N] = 0

        wSTRSalpha = w.copy()
        wSTRSalpha[np.ix_(substructure_nodes, substructure_nodes)] = 0 # -= wS

        wSTRSbeta = w.copy()
        wSTRSbeta[np.ix_(substructure_nodes, substructure_nodes)] += wS * gamma

        COS = VDX * TVDX + VDY * TVDY
        strs_ = np.sum((D - TD) ** 2 * wSTRSbeta) * beta + (np.sum((1 - COS) * wSTRSalpha)) * alpha # + np.sum(np.linalg.norm(V[substructure_nodes] - pos.reshape((int(pos.shape[0] / 2), 2)))) * gamma
        # print('stress: ', strs_)
        if strs_ == 0:
            break
        if k > 0:
            if (strs - strs_) / strs < eps * 100:
                break
        strs = strs_
        V = V_
    return V

def non_rigid_registration(source_G, target_G, markers, alpha=10, beta=10, gamma=1000, iter=1000):
    _target_G = target_G.copy()

    target_pos = {}
    for mk in markers:
        s_index = mk[0]
        t_index = mk[1]
        target_pos[t_index] = []
        target_pos[t_index].append(source_G.nodes[s_index])
        target_pos[t_index].append(1)


    # X = deform_v2(_target_G, target_pos, iter, alpha, beta, gamma)
    X = deform_v7(_target_G, target_pos, iter, alpha, beta, gamma)
    _target_G.nodes = X

    # _R, _t = aligning(target_G, _target_G, np.array([[index, index] for index in target_G.index2id]))
    # _target_G.nodes = _target_G.nodes.dot(_R.T) + _t
    # _target_G.nodes = (_target_G.nodes - t).dot(np.linalg.inv(R).T)  ############

    return _target_G

if __name__ == '__main__':
    # prefix = './data/test/'
    # source = nx.Graph()
    # source.add_edges_from([[0, 2], [0, 3], [2, 6], [3, 6], [1, 4], [1, 5], [4, 7], [5, 7], [6, 7]])
    # source.nodes[0]['x'] = 0.0
    # source.nodes[0]['y'] = 1.0
    # source.nodes[1]['x'] = 0.0
    # source.nodes[1]['y'] = 4.0
    # source.nodes[2]['x'] = 3.0
    # source.nodes[2]['y'] = 0.0
    # source.nodes[3]['x'] = 3.0
    # source.nodes[3]['y'] = 2.0
    # source.nodes[4]['x'] = 3.0
    # source.nodes[4]['y'] = 3.0
    # source.nodes[5]['x'] = 3.0
    # source.nodes[5]['y'] = 5.0
    # source.nodes[6]['x'] = 6.0
    # source.nodes[6]['y'] = 1.0
    # source.nodes[7]['x'] = 6.0
    # source.nodes[7]['y'] = 4.0
    # target_pos = {
    #     0: [np.array([0.0, 1.0]), 1],
    #     1: [np.array([0.0, 4.0]), 2],
    #     2: [np.array([3.0, 4.0]), 1],
    #     3: [np.array([3.0, -1.0]), 1],
    #     4: [np.array([3.0, 7.0]), 2],
    #     5: [np.array([3.0, 1.0]), 2],
    #     6: [np.array([5.0, 2.0]), -1],
    #     7: [np.array([5.0, 4.0]), -1],
    # }

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

    target_pos = {
        0: [np.array([0.0, 0.0]), 0],
        2: [np.array([2.0, 1.0]), 1],
        4: [np.array([4.0, 0.0]), 0],
    }

    source_G = Graph(source)
    G0 = source_G.copy()
    V = deform_v7(G0, target_pos, alpha=1, beta=1, gamma=100)
    print(V)
    G0.nodes = V
    print('xxxxxxxxxxxxx')
    G1 = source_G.copy()
    V = deform_v7(G1, target_pos, alpha=0, beta=1, gamma=100)
    print(V)
    G1.nodes = V
    save_json_graph(source_G.to_networkx(), prefix + 'result/target0.json')
    save_json_graph(G0.to_networkx(), prefix + 'result/target1.json')
    save_json_graph(G1.to_networkx(), prefix + 'result/target2.json')