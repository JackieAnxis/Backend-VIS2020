# -*- coding: UTF-8
import numpy as np
from hopcroftkarp import HopcroftKarp

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


def build_correspondence(source_G, target_G, correspondence):
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
    L = np.diag(np.sum(A, axis=0)) - A
    mean_adj_edge = L.dot(target_G.nodes)
    mean_adj_edge_length = np.sqrt(np.sum(mean_adj_edge**2, axis=1))
    res = []
    for cor in _res:
        length = np.sqrt(np.sum((source_G.nodes[cor[0]] - target_G.nodes[cor[1]])**2))
        if length < mean_adj_edge_length[cor[1]] * 3:
            res.append(cor)
    # delete un-pleasing correspondece #

    #####
    correspondence = correspondence.tolist()
    res = correspondence + res
    #####

    return np.array(res)