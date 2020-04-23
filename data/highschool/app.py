#!/usr/bin/python
# -*- coding: UTF-8 -*-
import csv
import json
import sys
import time
import math
import networkx as nx
from models.utils import save_json_graph
from models.layout import layout

# 图的度数统计


def degree_statistic(graph):
    statistic = {}
    for node, degree in graph.degree():
        if degree not in statistic:
            statistic[degree] = 0
        statistic[degree] += 1
    return statistic

# 添加一条边（有权重）


def add_weighted_edge(graph, node0, node1):
    weight = 0
    if graph.has_edge(node0, node1):
        weight = graph[node0][node1]['weight']  # node0和node1的边的权重
    graph.add_edge(node0, node1, weight=weight+1)

# 将边（也就是人与人的联系）放入其相关的帧内


def map_link_to_snapshot(link, min_time, interval, snapshots):
    # 判断这条边应该属于哪一帧
    time = link['time']
    index = int(math.floor((time - min_time) / interval))

    # 它也有可能属于后面几帧
    i = index
    while (i < len(snapshots) and snapshots[i]['start_time'] <= time and snapshots[i]['end_time'] > time):
        add_weighted_edge(snapshots[i]['graph'], link['id0'], link['id1'])
        i += 1

    # 它也有可能属于前面几帧
    i = index - 1
    while (i >= 0 and snapshots[i]['start_time'] <= time and snapshots[i]['end_time'] > time):
        add_weighted_edge(snapshots[i]['graph'], link['id0'], link['id1'])
        i -= 1


with open('./thiers_2012.csv') as f:
    thiers_2012 = csv.reader(f, delimiter='\t')  # csv读取器
    links = []
    id_class = {}  # 存储所有节点和对应的班级
    min_time = sys.maxsize  # 最大整数
    max_time = 0
    # 读文件
    for row in thiers_2012:
        # row就是csv的每一行
        # 每一行都是一次人与人的交流，第一列是时间，第二列是人物1，第三列是人物2，第四列是人物1的班级，第五列是人物2的班级
        time = int(row[0])
        # 读取最大最小时间
        if time < min_time:
            min_time = time
        if time > max_time:
            max_time = time
        id0 = row[1]
        id1 = row[2]
        class0 = row[3]
        class1 = row[4]
        links.append({
            'time': time,
            'id0': id0,
            'id1': id1,
            'class0': class0,
            'class1': class1
        })

        if id0 not in id_class:
            id_class[id0] = class0
        if id1 not in id_class:
            id_class[id1] = class1

    # 将所有的id放进一个数组里面
    nodes = []
    for node in id_class:
        nodes.append(node)

    # 接下去每一个小时为一帧，建立动态图
    # 前后两帧的起始时间，只相差6分钟，也就意味着前后两帧有54分钟的重叠；
    # 最后结果：0-60分钟为第一帧，6-66分钟为第二帧，以此类推
    snapshots = []
    interval = 6 * 60  # 6分钟 * 60秒
    duration = 60 * 60  # 1小时
    time = min_time
    while time <= max_time:
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        snapshots.append({
            'start_time': time,
            'end_time': time + duration,
            'graph': graph
        })
        time += interval

    # 将边（也就是人与人的联系）放入其相关的帧内
    for link in links:
        map_link_to_snapshot(link, min_time, interval, snapshots)

    # i = 0
    # for snapshot in snapshots:
    #     i += len(snapshot.nodes)
    G = nx.Graph()
    i = 0
    j = 0
    for snapshot in snapshots:
        if len(snapshot['graph'].edges) > 0:
            subgraph_nodes_count = []
            subgraph_nodes = []
            for c in filter(lambda n: len(n) > 3, nx.connected_components(snapshot['graph'])):
                subgraph_nodes += c
                subgraph_nodes_count.append(len(c))
            graph = snapshot['graph'].subgraph(subgraph_nodes)
            mapping = {}
            for node in graph.nodes:
                mapping[node] = j
                j += 1
            G = nx.union(G, nx.relabel_nodes(graph, mapping))
            print(subgraph_nodes_count, len(snapshot['graph'].nodes), i, len(snapshots))
        i += 1

    save_json_graph(G, './graph.json')
    nx.write_edgelist(G, './graph.edgelist')

    H = layout(G)

    save_json_graph(H, './graph-with-pos.json')

    # vectors = []
    # # 建立邻接矩阵
    # for snapshot in snapshots:
    #     A = nx.adjacency_matrix(snapshot['graph']).todense()
    #     vector = []
    #     for row in A:
    #         vector = vector + row.tolist()[0]
    #     vectors.append(vector)
    #
    # print(len(vectors), len(vectors[0]))
    # vectors = np.matrix(vectors, dtype='int16')
    #
    # # estimator = PCA(n_components=2)
    # estimator = TSNE(n_components=2)
    # X_pca = estimator.fit_transform(vectors)
    #
    # i = 0
    # data = []
    # for snapshot in snapshots:
    #     # 去掉那些孤立的节点
    #     snapshot['graph'].remove_nodes_from(
    #         list(nx.isolates(snapshot['graph'])))
    #
    #     # 统计这个snapshot的度数分布
    #     statistic = degree_statistic(snapshot['graph'])
    #
    #     data.append({
    #         "degree": statistic,
    #         "vector": X_pca[i].tolist(),
    #         "graph": json_graph.node_link_data(snapshot['graph'])
    #     })
    #     i += 1
    # with open('./test_data.json', 'w') as wf:
    #     json.dump(data, wf)
