# -*- coding: UTF-8
import numpy as np
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


def build_correspondence_v1(source_G, target_G, correspondence):
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
    # L = np.diag(np.sum(A, axis=0)) - A
    # mean_adj_edge = L.dot(target_G.nodes) / L.diagonal()[:, np.newaxis]
    # mean_adj_edge_length = np.sqrt(np.sum(mean_adj_edge ** 2, axis=1))
    res = []
    for cor in _res:
        length = np.sqrt(np.sum((source_G.nodes[cor[0]] - target_G.nodes[cor[1]])**2))
        mean_adj_edge_length = np.mean(np.sqrt(np.sum((np.diag(A[cor[1]]).dot(target_G.nodes - target_G.nodes[cor[1]]))**2, axis=1)))
        if length < mean_adj_edge_length * 4:
            res.append(cor)
    # delete un-pleasing correspondece #

    #####
    correspondence = correspondence.tolist()
    res = correspondence + res
    #####

    return np.array(res)

def build_correspondence(source_G, target_G, correspondence):
    distance_matrix = compute_distance_matrix(source_G.nodes, target_G.nodes)

    # # ignore the constructed correspondence
    # distance_matrix[np.ix_(correspondence[:, 0])] = np.ones((correspondence.shape[0],  target_G.nodes.shape[0])) * -1
    # distance_matrix[:, (correspondence[:, 1])] = np.ones((source_G.nodes.shape[0], correspondence.shape[0])) * -1

    rate = 2
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
    hungarian = Hungarian()
    hungarian.calculate(profit_matrix, is_profit_matrix=True)
    hungarian.calculate()
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
