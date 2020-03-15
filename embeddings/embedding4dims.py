import networkx as nx
import numpy as np
import csv

graph_path = "./data/bn-mouse-kasthuri/graph.edgelist"
graph = nx.read_edgelist(graph_path,  nodetype=int, data=(('weight',float),),)
for edge in graph.edges():
  graph[edge[0]][edge[1]]['weight'] = 1
# print(sorted(nx.degree(graph))) # 节点的度 list
nodes_degree = sorted(nx.degree(graph))
neighbor_ave_degree = nx.average_neighbor_degree(graph)
# print(nodes_degree)
# print(nx.average_neighbor_degree(graph)) # 邻接点的平均度 字典

neighbor_standerr_degree = {} # 度数标准差
for node in graph.nodes():
  neighbors = list(nx.all_neighbors(graph, node))
  degrees = []
  for neighbor_node in neighbors:
    degrees.append(graph.degree(neighbor_node))
  if len(degrees) < 2:
    std = 0
  else:
    std = np.std(degrees,ddof=1)
  neighbor_standerr_degree[node] = std

# print(degreeErr) #一个数的标准差暂存为0

with open('./data/test/test.csv', 'w',encoding='utf-8',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'dim_1', 'dim_2', 'dim3'])
    row = []
    for degree in nodes_degree:
      row.append(degree[0])
      row.append(degree[1])
      row.append(neighbor_ave_degree[degree[0]])
      row.append(neighbor_standerr_degree[degree[0]])
      writer.writerow(row)
      row = []


# print(nx.clustering(graph)) # 局部聚类系数
