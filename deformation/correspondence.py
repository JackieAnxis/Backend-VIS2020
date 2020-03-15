# -*- coding: UTF-8
import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.spatial import KDTree


def similarity_fitting(source_G, target_G, marker):
    source_marker_nodes = source_G.nodes[marker[:, 0], :].T # 2 * n
    target_marker_nodes = target_G.nodes[marker[:, 1], :].T # 2 * n
    source_center = np.mean(source_marker_nodes, axis=1)
    target_center = np.mean(target_marker_nodes, axis=1)
    X = source_marker_nodes - source_center[:, np.newaxis]
    Y = target_marker_nodes - target_center[:, np.newaxis]
    S = Y.dot(X.T)
    U, D, V = np.linalg.svd(S) # https://zhuanlan.zhihu.com/p/35893884
    R = U.dot(V.T)
    t = (source_center - target_center.dot(R.T))
    # n = marker.shape[0]
    # s = D.sum() * n / np.multiply(target_center, target_center).sum()
    
    # R.dot(X)+t = X'
    for k in range(0, 2):
        x0 = np.zeros((3, )) # R.size / 2 + t.size / 2
        x0[0:2] = R[k, :]
        x0[2] = t[k]

        def resSimXform(x, A, B):
            R = x[0:2]
            t = x[2]
            rot_A = B.copy()
            rot_A[k,:] = R.dot(A) + t
            result = np.sqrt(np.sum((B-rot_A)**2, axis=0))
            return result
        b = least_squares(fun=resSimXform, x0=x0, jac='2-point', method='lm', args=(target_marker_nodes, source_marker_nodes),
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=100000)
        R[k, :] = b.x[0:2]
        t[k] = b.x[2]
    # V_hat = np.vstack((source_marker_nodes[:, 1] - source_marker_nodes[:, 0], source_marker_nodes[:, 2] - source_marker_nodes[:, 0])).T
    # V = np.vstack((target_marker_nodes[:, 1] - target_marker_nodes[:, 0], target_marker_nodes[:, 2] - target_marker_nodes[:, 0])).T
    # R = V_hat.dot(np.linalg.inv(V))
    # t = source_marker_nodes[:, 1] - R.dot(target_marker_nodes[:, 1])

    rotated_target_marker_nodes = R.dot(target_marker_nodes) + t[:, np.newaxis]
    res = np.sum(np.sqrt(np.sum((source_marker_nodes - rotated_target_marker_nodes)**2, axis=1)))/source_marker_nodes.shape[1]

    print("Alignment error is {}".format(res))
    # print("Cost is {}".format(b.cost))
    print(R)
    print(t)
    return R, t

def El_linear_system(source_G, target_G, marker, wl):
    target_G.compute_adjacent_matrix()
    L = target_G.rw_laplacian_matrix(target_G.adj_matrix)
    V = target_G.nodes
    Delta = L.dot(V)
    # n = L.shape[0]
    n = target_G.new_nodes.shape[0]
    
    # LS = np.zeros([3*n, 3*n])
    # LS[0*n:1*n, 0*n:1*n] = (-1) * L
    # LS[1*n:2*n, 1*n:2*n] = (-1) * L
    # LS[2*n:3*n, 2*n:3*n] = (-1) * L
    C = np.zeros(((n+marker.shape[0])*2, 1))
    index = np.arange(L.shape[0], dtype=np.int32)
    LS = np.zeros([2*n, 2*n])
    for i in index:
        LS[i * 2, index*2] = (-1) * L[i]
        LS[i*2+1, index*2+1] = (-1) * L[i]

    # for i in range(n):
    for i in range(L.shape[0]):
        nb_idx = target_G.get_local_neighbor(i, target_G.adj_matrix) # node i and its neibors
        ring = np.array([i] + nb_idx)  # node i and its neibors in subgraph
        V_ring = V[ring]
        n_ring = V_ring.shape[0]
    
        A = np.zeros([n_ring * 2, 4])
        for j in range(n_ring):
            A[j] =          [V_ring[j, 0], -V_ring[j, 1], 1, 0]
            A[j+n_ring] =   [V_ring[j, 1],  V_ring[j, 0], 0, 1]
    
        # Moore-Penrose Inversion
        A_pinv_s = np.linalg.pinv(A)
        A_pinv = np.dot(np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A))
        s = A_pinv[0]
        h = A_pinv[1]
        # t = A_pinv[2:4]
    
        T_delta = np.vstack([
            Delta[i, 0]*s - Delta[i, 1]*h,
            Delta[i, 0]*h + Delta[i, 1]*s
        ])
    
        LS[i*2, np.hstack([ring*2, ring*2+1])] += T_delta[0]
        LS[i*2+1, np.hstack([ring*2, ring*2+1])] += T_delta[1]
        # LS[i+2*n, np.hstack([ring, ring+n, ring+2*n])] += T_delta[2]
    
    constraint_coef = []

    # Handle constraints
    for i in range(0, marker.shape[0]):
        source_marker_id = marker[i, 0]
        target_marker_id = marker[i, 1]
        constraint_coef.append(np.arange(2*n) == target_marker_id * 2)
        constraint_coef.append(np.arange(2*n) == target_marker_id *2 + 1)
        C[n*2+i*2:n*2+i*2+2, :] = source_G.nodes[source_marker_id][:,np.newaxis]

    constraint_coef = np.matrix(constraint_coef)

    A = np.vstack([wl * LS, 2 * wl * constraint_coef])
    M = sparse.coo_matrix(A)
    C = 2 * wl * C
    return M, C


def Es_linear_system(source_G, target_G, target_edge_adj, marker, ws):
    n = target_G.new_edges.shape[0]
    source_marker_nodes = marker[:, 0]
    target_marker_nodes = marker[:, 1]
    A = target_G.A
    self_I = np.zeros((n*n*3*2*2, 3)) # A matrix is a 3*2 matrix, and each edge needs two A matrix
    adjc_I = np.zeros((n*n*3*2*2, 3))
    C = np.zeros((n*n*2*2, 1)) # R matrix is a 2*2 matrix, each edge needs one R matrix
    for i in range(0, n):
        self_edge = target_G.new_edges[i]
        self_is_marker = np.array([False, False, False])
        for k in range(0, 2): # two nodes in one edge
            if np.sum(target_marker_nodes == self_edge[k], axis=0): # whether point k in edge i is a marker;
                self_is_marker[k] = True
        for j in range(0, n):
            if i != j and target_edge_adj[i, j]:
                adjc_edge = target_G.new_edges[j]
                adjc_is_marker = np.array([False, False, False])
                for k in range(0, 2): # two nodes in one edge
                    if np.sum(target_marker_nodes == adjc_edge[k], axis=0): # whether point k in edge i is a marker;
                        adjc_is_marker[k] = True
                for k in range(0, 2):
                    # one edge has n adj edges, with one R matrix per edge, R matrix is a 2*2 matrix
                    row_index = np.array([0,1]) + i*n*2*2 + j*2*2 + k*2
                    row = np.tile(row_index, [3, 1]).T
                    self_col = np.tile(self_edge * 2 + k, [2, 1])
                    adjc_col = np.tile(adjc_edge * 2 + k, [2, 1])
                    # [-(u00+u10) u00 u10]
                    # [-(u01+u11) u01 u11]
                    self_value = ws * A[i]
                    adjc_value = -ws * A[j]
                    
                    # force target marker into source marker's place, in case of C is all zero
                    if np.sum(self_is_marker):
                        target_marker_id = self_edge[self_is_marker]
                        source_marker_id = -np.ones((target_marker_id.shape), dtype=np.int32)
                        for l in range(0, target_marker_id.shape[0]):
                            source_marker_id[l] = source_marker_nodes[target_marker_nodes == target_marker_id[l]]
                        C[row_index] -= self_value[:, self_is_marker].dot(source_G.nodes[source_marker_id, k])[:, np.newaxis]
                        self_value[:, self_is_marker] = 0
                    if np.sum(adjc_is_marker):
                        target_marker_id = adjc_edge[adjc_is_marker]
                        source_marker_id = -np.ones((target_marker_id.shape), dtype=np.int32)
                        for l in range(0, target_marker_id.shape[0]):
                            source_marker_id[l] = source_marker_nodes[target_marker_nodes==target_marker_id[l]]
                        C[row_index] -= adjc_value[:, adjc_is_marker].dot(source_G.nodes[source_marker_id, k])[:, np.newaxis]
                        adjc_value[:, adjc_is_marker] = 0

                    # one face relates to one A matrix, it needs 3*2 units
                    index = np.linspace(0, 5, 6, dtype=np.int32) + i*n*2*3*2 + j*2*3*2 + k*3*2
                    self_I[index, :] = \
                         np.hstack((row.flatten()[:, np.newaxis],
                                    self_col.flatten()[:, np.newaxis], 
                                    self_value.flatten()[:, np.newaxis]))
                    adjc_I[index, :] = \
                        np.hstack((row.flatten()[:, np.newaxis],
                                    adjc_col.flatten()[:, np.newaxis], 
                                    adjc_value.flatten()[:, np.newaxis]))

    self_I = self_I[self_I[:, 0] >= 0, :]
    adjc_I = adjc_I[adjc_I[:, 0] >= 0, :]
    self_M = sparse.coo_matrix((self_I[:, 2], (self_I[:, 0], self_I[:, 1])), shape=(2*2*n*n, 2*target_G.new_nodes.shape[0]))
    adjc_M = sparse.coo_matrix((adjc_I[:, 2], (adjc_I[:, 0], adjc_I[:, 1])), shape=(2*2*n*n, 2*target_G.new_nodes.shape[0]))
    M = self_M + adjc_M
    return M, C

def Ei_linear_system(source_G, target_G, wi):
    n = target_G.new_edges.shape[0]
    A = target_G.A
    I = np.zeros((n*2*2*3, 3))
    C = wi*np.tile(np.reshape(np.eye(2), [2*2, 1]), (n, 1))
    for i in range(0, n):
        edge = target_G.new_edges[i]
        value = wi * A[i]
        for k in range(0, 2):
            row_index = np.array([0, 1]) + i*2*2 + k*2
            row = np.tile(row_index, [3,1]).T
            col = np.tile(edge*2 + k, [2, 1])
            index = np.linspace(0, 5, 6, dtype=np.int32) + i*2*2*3 + k*2*3
            I[index, :] = np.hstack((row.flatten()[:, np.newaxis], col.flatten()[:, np.newaxis], value.flatten()[:, np.newaxis]))

    M = sparse.coo_matrix((I[:, 2], (I[:, 0], I[:, 1])), shape=(2*2*n, 2*target_G.new_nodes.shape[0]))
    return M, C

# def Ec_linear_system(source_G, target_G, marker, wc):
def Ec_linear_system(source_G, target_G, marker, wc, K, max_dis):
    target_nodes_count = target_G.nodes.shape[0]
    source_marker_nodes = marker[:, 0]
    target_marker_nodes = marker[:, 1]
    C = np.zeros((2 * target_nodes_count, 1))
    source_tree = KDTree(source_G.nodes)
    for i in range(target_nodes_count):
        if np.sum(target_marker_nodes == i):
            valid_nodes = source_marker_nodes[target_marker_nodes == i]
        else:
            node = target_G.new_nodes[i]
            # valid_node = find_closest_validpt(node, source_G.nodes) # TODO
            valid_nodes = find_closest_validpt(node, source_tree, K, max_dis) # TODO
        # C[np.linspace(0, 1, 2, dtype=np.int32)+i*2] = wc * source_G.nodes[valid_node, :].T
        C[np.linspace(0, 1, 2, dtype=np.int32)+i*2] = wc * np.mean(source_G.nodes[valid_nodes, :], axis=0)[:, np.newaxis]
    # value = np.tile(wc, [3*target_nodes_count, 1])
    # row = np.arange(0, 3*target_nodes_count)
    # col = np.arange(0, 3*target_nodes_count)
    M = np.hstack((wc * np.eye(2 * target_nodes_count), np.zeros((2 * target_nodes_count, 2 * (target_G.new_nodes.shape[0]-target_nodes_count)))))
    M = sparse.csr_matrix(M)
    return M, C

# def find_closest_validpt(node, search_nodes):
#     '''
#     find the closest node, not valid, 
#     delete the normals' intersection angle judgement
#     '''
#     d = np.sum((np.tile(node, [search_nodes.shape[0], 1]) - search_nodes)**2, axis=1) # distances
#     ind = np.argsort(d) # sort accord to the distance between spt and vpts
#     return np.array([ind[0]])
def find_closest_validpt(node, tree, K, max_dis):
    _, corresind = tree.query(node, k=K, distance_upper_bound=max_dis)
    corresind = corresind[corresind >= 0]
    corresind = corresind[corresind < tree.data.shape[0]]
    # intersection angle: x0*x1+y0*y1
    # inter_angles = np.sum(np.tile(G.perpendicular[i], [corresind.shape[0], 1])*search_G.perpendicular[corresind], axis=1)

    d = np.sum((np.tile(node, [tree.data.shape[0], 1]) - tree.data)**2, axis=1) # distances
    ind = np.argsort(d) # sort accord to the distance between spt and vpts
    corresind = ind[0:K]
    return corresind

# def non_rigid_registration(source_G, target_G, ws, wi, wc, marker):
def non_rigid_registration(source_G, target_G, ws, wi, wc, marker, K, max_dis):
    # change target into source
    source_G = source_G.copy()
    source_G.normalize()
    target_G = target_G.copy()
    target_G.normalize()
    marker[:, 0] = np.array([source_G.id2index[str(id)] for id in marker[:, 0]])
    marker[:, 1] = np.array([target_G.id2index[str(id)] for id in marker[:, 1]])
    R = np.array([[1,0],[0,1]])
    t = np.array([0,0])
    # # TODO NEXT TWO LINES NEED TO BE REVOVERY!
    R, t = similarity_fitting(source_G, target_G, marker)
    target_G.nodes = target_G.nodes.dot(R.T) + t
    target_G.compute_third_node()
    source_G.compute_third_node()
    target_edge_adj = target_G.find_adj_edges()
    target_G.build_elementary_cell()
    # smooth
    ElM, ElC = El_linear_system(source_G, target_G, marker, ws)
    # EsM, EsC = Es_linear_system(source_G, target_G, target_edge_adj, marker, ws)
    # identity
    EiM, EiC = Ei_linear_system(source_G, target_G, wi)
    M = sparse.vstack([ElM, EiM])
    C = np.vstack((ElC, EiC))
    X = sparse.linalg.lsqr(M, C, iter_lim=30000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
    target_G.nodes = X[0].reshape((target_G.new_nodes.shape[0], 2))[0:target_G.nodes.shape[0],:]
    print(target_G.nodes)
    for i in range(0, len(wc)):
        ws += i * wc[i] / 1000
        target_G.compute_third_node()
        target_G.build_elementary_cell()
        # smooth
        ElM, ElC = El_linear_system(source_G, target_G, marker, ws)
        # EsM, EsC = Es_linear_system(source_G, target_G, target_edge_adj, marker, ws)
        # identity
        EiM, EiC = Ei_linear_system(source_G, target_G, wi)
        # closest
        EcM, EcC = Ec_linear_system(source_G, target_G, marker, wc[i], K, max_dis)
        # M = sparse.vstack([EsM, EiM, EcM])
        # C = np.vstack((EsC, EiC, EcC))
        M = sparse.vstack([ElM, EiM, EcM])
        C = np.vstack((ElC, EiC, EcC))
        X = sparse.linalg.lsqr(M, C, iter_lim=10000, atol=1e-8, btol=1e-8, conlim=1e7, show=False)
        target_G.nodes = X[0].reshape((target_G.new_nodes.shape[0], 2))[0:target_G.nodes.shape[0],:]
    return source_G, target_G, R, t

def find_closest_nodes(G, search_G, K, max_dis):
    tree = KDTree(search_G.nodes)
    correspondence = -np.ones((G.nodes.shape[0], K), dtype=np.int32)
    for i in range(0, correspondence.shape[0]):
        # edge = [G.index2id[index] for index in G.edges[i]]
        _, corresind = tree.query(G.nodes[i, :], k=K, distance_upper_bound=max_dis)
        corresind[corresind < 0] = -1
        corresind[corresind >= tree.data.shape[0]] = -1
        # intersection angle: x0*x1+y0*y1
        # inter_angles = np.sum(np.tile(G.perpendicular[i], [corresind.shape[0], 1])*search_G.perpendicular[corresind], axis=1)
        correspondence[i, :] = corresind
        # corr_edges = [search_G.edges[index] for index in corresind[corresind>=0]]
        # for t in range(0, len(corr_edges)):
        #     corr_edges[t] = [search_G.index2id[index] for index in corr_edges[t]]
        # print(edge, corr_edges)
    return correspondence

def print_correspondence(target_G, source_G, correspondence):
    for i in range(0, len(correspondence)):
        target_node = target_G.index2id[i]
        source_nodes = [source_G.index2id[index] for index in correspondence[i][correspondence[i] >= 0]]
        print(target_node, source_nodes)

def build_correspondence(source_G, target_G, K, max_dis):
    source_G.compute_third_node()
    target_G.compute_third_node()
    # source_G.compute_edges_center()
    # target_G.compute_edges_center()
    correspondence_1 = find_closest_nodes(target_G, source_G, K, max_dis)
    # print_correspondence(target_G, source_G, correspondence_1)
    s2t_correspondence = find_closest_nodes(source_G, target_G, K, max_dis)
    correspondence_2 = -np.ones((correspondence_1.shape[0], correspondence_1.shape[0]), dtype=np.int32)
    index = np.zeros((correspondence_1.shape[0], 1), dtype=np.int32)
    for i in range(0, s2t_correspondence.shape[0]):
        row = s2t_correspondence[i, s2t_correspondence[i]>0] #
        col = index[row].flatten()
        correspondence_2[row, col] = i
        index[row] += 1
    correspondence_2 = np.array([np.array(row) for row in correspondence_2])
    # print('##################')
    # print_correspondence(target_G, source_G, correspondence_2)
    tmp_correspondence = np.hstack((correspondence_1, correspondence_2))
    # print('##################')
    # print_correspondence(target_G, source_G, tmp_correspondence)
    correspondence = []
    for i in range(0, tmp_correspondence.shape[0]):
        # ###########
        # target_edge = np.zeros(target_G.edges[i].shape, dtype=np.int32)
        # for j in range(0, 2):
        #     target_edge[j] = int(target_G.index2id[target_G.edges[i][j]])
        # source_edges = np.zeros(source_G.edges[tmp_correspondence[i][tmp_correspondence[i] > 0]].shape, dtype=np.int32)
        # for j in range(0, source_edges.shape[0]):
        #     for k in range(0, 2):
        #         source_edges[j][k] = int(source_G.index2id[source_G.edges[tmp_correspondence[i][j], k]])
        # ###########
        tmp = np.unique(tmp_correspondence[i,:])
        tmp = tmp[tmp >= 0]
        correspondence.append(tmp)
    print('##################')
    print_correspondence(target_G, source_G, correspondence)
    return correspondence

# def build_correspondence_v2(source_G, target_G):
#     if source_G.nodes.shape[0] > target_G.nodes.shape[0]:
#         G1 = source_G
#         G0 = target_G
#     else:
#         G0 = source_G
#         G1 = target_G
#     # 优化目标：尝试两两匹配节点，然后看边和边的重叠度，最大化重叠度，最小化位移
#     ### 第一步，找到最大化重叠程度 前α分之一的 方案 ###
#     n = G0.nodes.shape[0]
#     m = G1.nodes.shape[0]
#     # m * (m-1) * (m-2) * ... * n
#     def x(n, m):
#         for i in range(0, n):
#             for j in range(i + 1, m):
#                 x(n)
#                 [i, j]
#     count = 1
#     for i in range(n, m + 1):
#         count *= i
    
        