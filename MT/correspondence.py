# -*- coding: UTF-8
import numpy as np
import networkx as nx
from hopcroftkarp import HopcroftKarp
from MT.Hugarian import Hungarian

def maximum_matching(matrix):
    graph = {}
    for i in range(matrix.shape[0]):
        graph[str(i) + 's'] = np.nonzero(matrix[i])[0]
    match = HopcroftKarp(graph).maximum_matching()

    res = [] # [source. trarget]
    for i in range(matrix.shape[0]):
        key = str(i) + 's'
        if key in match:
            j = match[key]
            res.append([i, j])
    return res

def compute_distance_matrix(V0, V1):
    n = V0.shape[0]
    m = V1.shape[0]
    distance_matrix = np.zeros((n, m))
    for i in range(n):
        distance_matrix[i] = np.sqrt(np.sum((V1 - V0[i]) ** 2, axis=1))
    return distance_matrix


def build_correspondence_v1(source_G, target_G, correspondence, rate=2):
    distance_matrix = compute_distance_matrix(source_G.nodes, target_G.nodes)

    for corr in correspondence: # the nodes has constructed the correspondence
        distance_matrix[corr[0]] = np.ones((1, distance_matrix.shape[1])) * -1
        distance_matrix[:, corr[1]] = np.ones((distance_matrix.shape[0])) * -1

    min_for_each_row = np.zeros((distance_matrix.shape[0]))
    for i in range(distance_matrix.shape[0]):
        positive_cell = distance_matrix[i][distance_matrix[i]>0]
        if positive_cell.shape[0]:
            min_for_each_row[i] = np.min(positive_cell)
        else:
            min_for_each_row[i] = -2

    min_for_each_col = np.zeros((distance_matrix.shape[1]))
    for j in range(distance_matrix.shape[1]):
        positive_cell = distance_matrix[:, j][distance_matrix[:, j]>0]
        if positive_cell.shape[0]:
            min_for_each_col[j] = np.min(positive_cell)
        else:
            min_for_each_col[j] = -2

    source_corr = (distance_matrix == min_for_each_row[:, np.newaxis])
    target_corr = (distance_matrix == min_for_each_col)

    corr = source_corr * target_corr

    _res = maximum_matching(corr)

    # delete un-pleasing correspondece #
    A = target_G.compute_adjacent_matrix()
    res = []
    for cor in _res:
        length = np.sqrt(np.sum((source_G.nodes[cor[0]] - target_G.nodes[cor[1]])**2))
        mean_adj_edge_length = np.mean(np.sqrt(np.sum((np.diag(A[cor[1]]).dot(target_G.nodes - target_G.nodes[cor[1]]))**2, axis=1)))
        if length < mean_adj_edge_length * rate:
            res.append(cor)
    # delete un-pleasing correspondece #

    #####
    correspondence = correspondence.tolist()
    res = correspondence + res
    #####

    return np.array(res)

def build_correspondence_v2(source_G, target_G, correspondence, rate=2):
    distance_matrix = compute_distance_matrix(source_G.nodes, target_G.nodes)

    # # ignore the constructed correspondence
    # distance_matrix[np.ix_(correspondence[:, 0])] = np.ones((correspondence.shape[0],  target_G.nodes.shape[0])) * -1
    # distance_matrix[:, (correspondence[:, 1])] = np.ones((source_G.nodes.shape[0], correspondence.shape[0])) * -1

    A = target_G.compute_adjacent_matrix()
    R = np.zeros(shape=(1, target_G.nodes.shape[0]))
    for index in target_G.index2id:
        R[:, index] = np.mean(np.sqrt(np.sum((np.diag(A[index]).dot(target_G.nodes - target_G.nodes[index]))**2, axis=1))) * rate

    # Q = distance_matrix > np.zeros(shape=distance_matrix.shape)
    P = distance_matrix < np.tile(R, (distance_matrix.shape[0], 1))

    # ignore the constructed correspondence
    P[np.ix_(correspondence[:, 0])] = np.zeros((correspondence.shape[0], target_G.nodes.shape[0]))
    P[:, (correspondence[:, 1])] = np.zeros((source_G.nodes.shape[0], correspondence.shape[0]))

    eps = 10e-5
    profit_matrix = np.zeros(shape=P.shape)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j]:
                sum_adj_link_length_before = np.sum(np.sqrt(np.sum((np.diag(A[j]).dot(target_G.nodes) - target_G.nodes[j]) ** 2, axis=1)))
                sum_adj_link_length_after = np.sum(np.sqrt(np.sum((np.diag(A[j]).dot(target_G.nodes) - source_G.nodes[i]) ** 2, axis=1)))
                cost = np.abs(sum_adj_link_length_after - sum_adj_link_length_before) + eps
                # sum_adj_length_before = np.sum(
                #     np.sqrt(np.sum((target_G.nodes - target_G.nodes[j]) ** 2, axis=1)))
                # sum_adj_length_after = np.sum(
                #     np.sqrt(np.sum((target_G.nodes - source_G.nodes[i]) ** 2, axis=1)))
                # cost = np.abs(sum_adj_length_after - sum_adj_length_before) + eps
                profit = 1 / cost
                profit_matrix[i, j] = profit

    cost_matrix = distance_matrix / (P + eps)
    hungarian = Hungarian(cost_matrix)
    hungarian.calculate()
    # hungarian = Hungarian()
    # hungarian.calculate(profit_matrix, is_profit_matrix=True)
    res = hungarian.get_results()
    # res = np.array(res)

    Q = np.zeros(shape=distance_matrix.shape)
    for tuple in res:
        Q[tuple[0], tuple[1]] = 1

    res = []
    Q = Q * P
    for row in range(Q.shape[0]):
        if len(np.nonzero(Q[row])[0]):
            res.append([row, np.nonzero(Q[row])[0][0]])

    #####
    correspondence = correspondence.tolist()
    res = correspondence + res
    #####

    return np.array(res)

def build_correspondence_v3(source_G, target_G, correspondence, rate=2):
    distance_matrix = compute_distance_matrix(source_G.nodes, target_G.nodes)

    for corr in correspondence:  # the nodes has constructed the correspondence
        distance_matrix[corr[0]] = np.ones((1, distance_matrix.shape[1])) * -1
        distance_matrix[:, corr[1]] = np.ones((distance_matrix.shape[0])) * -1

    min_for_each_row = np.zeros((distance_matrix.shape[0]))
    for i in range(distance_matrix.shape[0]):
        positive_cell = distance_matrix[i][distance_matrix[i] > 0]
        if positive_cell.shape[0]:
            min_for_each_row[i] = np.min(positive_cell)
        else:
            min_for_each_row[i] = -2

    min_for_each_col = np.zeros((distance_matrix.shape[1]))
    for j in range(distance_matrix.shape[1]):
        positive_cell = distance_matrix[:, j][distance_matrix[:, j] > 0]
        if positive_cell.shape[0]:
            min_for_each_col[j] = np.min(positive_cell)
        else:
            min_for_each_col[j] = -2

    source_corr = (distance_matrix == min_for_each_row[:, np.newaxis])
    target_corr = (distance_matrix == min_for_each_col)

    P = source_corr * target_corr

    eps = 10e-5
    profit_matrix = np.zeros(shape=P.shape)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j]:
                # sum_adj_link_length_before = np.sum(np.sqrt(np.sum((np.diag(A[j]).dot(target_G.nodes) - target_G.nodes[j]) ** 2, axis=1)))
                # sum_adj_link_length_after = np.sum(np.sqrt(np.sum((np.diag(A[j]).dot(target_G.nodes) - source_G.nodes[i]) ** 2, axis=1)))
                sum_adj_link_length_before = np.sum(
                    np.sqrt(np.sum((target_G.nodes - target_G.nodes[j]) ** 2, axis=1)))
                sum_adj_link_length_after = np.sum(
                    np.sqrt(np.sum((target_G.nodes - source_G.nodes[i]) ** 2, axis=1)))
                cost = np.abs(sum_adj_link_length_after - sum_adj_link_length_before) + eps
                profit = 1 / cost
                profit_matrix[i, j] = profit

    # cost_matrix = distance_matrix / (P + eps)
    # hungarian = Hungarian(cost_matrix)
    # hungarian.calculate()
    hungarian = Hungarian()
    hungarian.calculate(profit_matrix, is_profit_matrix=True)
    res = hungarian.get_results()

    Q = np.zeros(shape=distance_matrix.shape)
    for tuple in res:
        Q[tuple[0], tuple[1]] = 1

    _res = []
    Q = Q * P
    for row in range(Q.shape[0]):
        if len(np.nonzero(Q[row])[0]):
            _res.append([row, np.nonzero(Q[row])[0][0]])

    # delete un-pleasing correspondece #
    A = target_G.compute_adjacent_matrix()
    res = []
    for cor in _res:
        length = np.sqrt(np.sum((source_G.nodes[cor[0]] - target_G.nodes[cor[1]]) ** 2))
        mean_adj_edge_length = np.mean(
            np.sqrt(np.sum((np.diag(A[cor[1]]).dot(target_G.nodes - target_G.nodes[cor[1]])) ** 2, axis=1)))
        if length < mean_adj_edge_length * rate:
            res.append(cor)
    # delete un-pleasing correspondece #

    #####
    correspondence = correspondence.tolist()
    res = correspondence + res
    #####

    return np.array(res)

# def build_correspondence_v4(source_G, target_G, correspondence, rate=2, w_1=10, w_2=1):
#     target = target_G.copy().to_networkx()
#     source = source_G.copy().to_networkx()
#     distance_matrix = compute_distance_matrix(source_G.nodes, target_G.nodes)
#     max_dis = np.max(distance_matrix)
#     # source_dis_mat = compute_distance_matrix(source_G.nodes, source_G.nodes)
#     # target_dis_mat = compute_distance_matrix(target_G.nodes, target_G.nodes)
#
#     # graphs = [source, target]
#     # Gs = [source_G, target_G]
#     # dis_mats = [source_dis_mat, target_dis_mat]
#     # for k in range(2):
#     #     dis_mat = dis_mats[k]
#     #     graph = graphs[k]
#     #     G = Gs[k]
#     #     for i in range(dis_mat.shape[0]):
#     #         sid = G.index2id[i]
#     #         for j in range(i + 1, dis_mat.shape[0]):
#     #             tid = G.index2id[j]
#     #             graph[sid][tid]['weight'] = dis_mat[i, j]
#
#     source_built_idx = {}
#     target_built_idx = {}
#     for k in range(correspondence.shape[0]):
#         sidx = correspondence[k][0]
#         tidx = correspondence[k][1]
#         source_built_idx[sidx] = True
#         target_built_idx[tidx] = True
#
#     SAM = source_G.compute_adjacent_matrix() # source adjacent matrix
#     TAM = target_G.compute_adjacent_matrix()  # source adjacent matrix
#
#     sm_idxs = correspondence[:, 0].flatten() # source markers' index
#     tm_idxs = correspondence[:, 1].flatten() # target markers' index
#     sm_idxs_bool = np.zeros((1, SAM.shape[0])) # whether it is one of source markers
#     tm_idxs_bool = np.zeros((1, TAM.shape[0])) # whether it is one of source markers
#     sm_idxs_bool[:, sm_idxs] = 1
#     tm_idxs_bool[:, tm_idxs] = 1
#     snb_idxs_bool = np.sum(SAM[sm_idxs, :], axis=0) * (1 - sm_idxs_bool) # whether it is one of source markers' neighbors
#     tnb_idxs_bool = np.sum(TAM[tm_idxs, :], axis=0) * (1 - tm_idxs_bool) # whether it is one of target markers' neighbors
#     source_mapping = np.nonzero(snb_idxs_bool)[1]
#     target_mapping = np.nonzero(tnb_idxs_bool)[1]
#     profit_matrix = np.zeros((SAM.shape[0], TAM.shape[0]))
#     for k in range(correspondence.shape[0]):
#         sm_idx = correspondence[k][0] # source marker index
#         tm_idx = correspondence[k][1] # target marker index
#         snb_idx = np.nonzero(SAM[sm_idx])[0] # nx.neighbors(source, source_G.index2id[sm_idx]) # id of source neighbors of source marker[k]
#         tnb_idx = np.nonzero(TAM[tm_idx])[0] # nx.neighbors(source, source_G.index2id[sm_idx]) # id of source neighbors of source marker[k]
#         for sidx in snb_idx:
#             sid = source_G.index2id[sidx]
#             sdgr = source.degree[sid]
#             for tidx in tnb_idx:
#                 tid = target_G.index2id[tidx]
#                 tdgr = target.degree[tid]
#                 degree_cost = np.abs(sdgr - tdgr) / np.max([sdgr, tdgr])
#                 distance_cost = distance_matrix[sidx, tidx] / max_dis
#                 profit_matrix[sidx, tidx] = (1 - degree_cost) * w_1 + (1 - distance_cost) * w_2
#
#     _profit_matrix = profit_matrix[source_mapping][:, target_mapping]
#
#     hungarian = Hungarian()
#     hungarian.calculate(_profit_matrix, is_profit_matrix=True)
#     _res = hungarian.get_results()
#
#     res = []
#     for tuple in _res:
#         if _profit_matrix[tuple[0], tuple[1]] > 0:
#             res.append([source_mapping[tuple[0]], target_mapping[tuple[1]]])\
#
#     #####
#     correspondence = correspondence.tolist()
#     res = correspondence + res
#     #####
#
#     return np.array(res)

def build_correspondence_v4(source_G, target_G, correspondence, step=1, w_1=1, w_2=100, w_3=10):
    if correspondence.shape[0] >= np.min([source_G.nodes.shape[0], target_G.nodes.shape[0]]):
        return correspondence
    target = target_G.copy().to_networkx()
    source = source_G.copy().to_networkx()
    distance_matrix = compute_distance_matrix(source_G.nodes, target_G.nodes)
    source_gd_matrix = source_G.compute_graph_distance_matrix() # source graph distance matrix
    target_gd_matrix = target_G.compute_graph_distance_matrix() # target graph distance matrix
    # source_dis_mat = compute_distance_matrix(source_G.nodes, source_G.nodes)
    # target_dis_mat = compute_distance_matrix(target_G.nodes, target_G.nodes)

    # source_built_idx = {}
    # target_built_idx = {}
    # for k in range(correspondence.shape[0]):
    #     sidx = correspondence[k][0]
    #     tidx = correspondence[k][1]
    #     source_built_idx[sidx] = True
    #     target_built_idx[tidx] = True

    gd_cost_mat = -1 * np.ones(shape=(source_G.nodes.shape[0], target_G.nodes.shape[0]))
    dg_cost_mat = -1 * np.ones(shape=(source_G.nodes.shape[0], target_G.nodes.shape[0]))
    ed_cost_mat = -1 * np.ones(shape=(source_G.nodes.shape[0], target_G.nodes.shape[0]))
    for k in range(correspondence.shape[0]):
        sm_idx = correspondence[k, 0] # source marker' index
        tm_idx = correspondence[k, 1]  # target marker' index
        sm_ngbr_idxs = np.nonzero((source_gd_matrix[sm_idx] <= step) * (source_gd_matrix[sm_idx] > 0))[0]
        tm_ngbr_idxs = np.nonzero((target_gd_matrix[tm_idx] <= step) * (target_gd_matrix[tm_idx] > 0))[0]
        for i in sm_ngbr_idxs:
            sgd = source_gd_matrix[i][sm_idx] # node i to marker's distance
            sdg = source.degree[source_G.index2id[i]] # node i's degree
            for j in tm_ngbr_idxs:
                tgd = target_gd_matrix[j][tm_idx]  # node j to marker's distance
                tdg = target.degree[target_G.index2id[j]]  # node j's degree
                gd_cost = np.abs(sgd - tgd)
                dg_cost = np.abs(sdg - tdg) / np.max([sdg, tdg])
                ed_cost = distance_matrix[i, j]  # euclidean degree cost
                dg_cost_mat[i, j] = dg_cost
                ed_cost_mat[i, j] = ed_cost
                if gd_cost_mat[i, j] < 0 or gd_cost_mat[i, j] > gd_cost:
                    gd_cost_mat[i, j] = gd_cost


    eps = 10e-6
    gd_profit_mat = 1 - (gd_cost_mat + eps) / (np.max(gd_cost_mat) + eps)
    gd_profit_mat[gd_cost_mat == -1] = -1
    dg_profit_mat = 1 - dg_cost_mat # 1 - (dg_cost_mat + eps) / (np.max(dg_cost_mat) + eps)
    dg_profit_mat[dg_cost_mat == -1] = -1
    ed_profit_mat = 1 - (ed_cost_mat + eps) / (np.max(ed_cost_mat) + eps)
    ed_profit_mat[ed_cost_mat == -1] = -1
    profit_matrix = gd_profit_mat * w_1 + dg_profit_mat * w_2 + ed_profit_mat * w_3

    sm_idxs = correspondence[:, 0].flatten() # source markers' index
    tm_idxs = correspondence[:, 1].flatten() # target markers' index
    is_sm = np.zeros((source_G.nodes.shape[0])) # whether it is one of source markers
    is_tm = np.zeros((target_G.nodes.shape[0])) # whether it is one of source markers
    is_sm[sm_idxs] = 1
    is_tm[tm_idxs] = 1
    non_sm = 1 - is_sm
    non_tm = 1 - is_tm
    non_sm_idxs = np.nonzero(non_sm)[0]
    non_tm_idxs = np.nonzero(non_tm)[0]

    _gd_profit_mat = gd_profit_mat[non_sm_idxs, :][:, non_tm_idxs]
    source_in_step = np.zeros(shape=(source_G.nodes.shape[0]))
    source_in_step[non_sm_idxs] = np.max(_gd_profit_mat, axis=1) > -1
    target_in_step = np.zeros(shape=(target_G.nodes.shape[0]))
    target_in_step[non_tm_idxs] = np.max(_gd_profit_mat, axis=0) > -1

    source_mapping = np.nonzero(source_in_step * non_sm)[0]
    target_mapping = np.nonzero(target_in_step * non_tm)[0]

    _profit_matrix = profit_matrix[source_mapping, :][:, target_mapping]

    # SAM = source_G.compute_adjacent_matrix() # source adjacent matrix
    # TAM = target_G.compute_adjacent_matrix() # source adjacent matrix

    # sm_idxs = correspondence[:, 0].flatten() # source markers' index
    # tm_idxs = correspondence[:, 1].flatten() # target markers' index
    # sm_idxs_bool = np.zeros((1, SAM.shape[0])) # whether it is one of source markers
    # tm_idxs_bool = np.zeros((1, TAM.shape[0])) # whether it is one of source markers
    # sm_idxs_bool[:, sm_idxs] = 1
    # tm_idxs_bool[:, tm_idxs] = 1
    # source_non_marker_idxs = np.nonzero(1 - sm_idxs_bool)[1]
    # target_non_marker_idxs = np.nonzero(1 - tm_idxs_bool)[1]
    # gd_cost = -1 * np.ones((source_non_marker_idxs.shape[0], target_non_marker_idxs.shape[0]))
    # dg_cost = -1 * np.ones((source_non_marker_idxs.shape[0], target_non_marker_idxs.shape[0]))
    # ed_cost = -1 * np.ones((source_non_marker_idxs.shape[0], target_non_marker_idxs.shape[0]))
    # for i in range(source_non_marker_idxs.shape[0]):
    #     sidx = source_non_marker_idxs[i]
    #     sid = source_G.index2id[sidx]
    #     sdgr = source.degree[sid] # source node's degree
    #     sd2m = np.zeros(shape=(correspondence.shape[0])) # source node i's graph distance to markers
    #     for k in range(correspondence.shape[0]):
    #         smid = source_G.index2id[correspondence[k][0]] # source marker id
    #         sd2m[k] = nx.shortest_path_length(source, smid, sid)
    #     for j in range(target_non_marker_idxs.shape[0]):
    #         tidx = target_non_marker_idxs[j]
    #         tid = target_G.index2id[tidx]
    #         tdgr = target.degree[tid] # target node's degree
    #         td2m = np.zeros(shape=(correspondence.shape[0])) # target node i's graph distance to markers
    #         for k in range(correspondence.shape[0]):
    #             tmid = target_G.index2id[correspondence[k][1]]  # source marker id
    #             td2m[k] = nx.shortest_path_length(target, tmid, tid)
    #
    #         gd_cost[i, j] = np.min(np.abs(np.hstack((
    #             sd2m[np.where(td2m == np.min(td2m))] - td2m[np.where(td2m == np.min(td2m))],
    #             sd2m[np.where(sd2m == np.min(sd2m))] - td2m[np.where(sd2m == np.min(sd2m))]))))
    #         # gd_cost[i, j] = np.min(np.abs(td2m - sd2m)) # graph distance cost
    #         dg_cost[i, j] = np.abs(tdgr - sdgr) # graph degree cost
    #         ed_cost[i, j] = distance_matrix[i, j] # euclidean degree cost
    #
    # eps = 10e-5
    # profit_matrix = (1 - ((gd_cost + eps) / (np.max(gd_cost) + eps))) * w_1 + (1 - ((dg_cost + eps) / (np.max(dg_cost) + eps))) * w_2 + (1 - ((ed_cost + eps) / (np.max(ed_cost) + eps))) * w_3
    # source_in_rate_idxs = np.nonzero(np.min(gd_cost, axis=1) <= rate)[0]
    # target_in_rate_idxs = np.nonzero(np.min(gd_cost, axis=0) <= rate)[0]
    # _profit_matrix = profit_matrix[source_in_rate_idxs, :][:, target_in_rate_idxs]
    # source_mapping = [source_non_marker_idxs[i] for i in source_in_rate_idxs]
    # target_mapping = [target_non_marker_idxs[i] for i in target_in_rate_idxs]

    _res = []
    if len(source_mapping) > 0 and len(target_mapping) > 0:
        hungarian = Hungarian()
        hungarian.calculate(_profit_matrix, is_profit_matrix=True)
        _res = hungarian.get_results()

    res = []
    for tuple in _res:
        sid = source_mapping[tuple[0]]
        tid = target_mapping[tuple[1]]
        if profit_matrix[sid, tid] >= eps * -1.0:
            res.append([sid, tid])

    #####
    correspondence = correspondence.tolist()
    res = correspondence + res
    #####

    return np.array(res)