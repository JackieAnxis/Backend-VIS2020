import json
import networkx as nx
from networkx.readwrite import json_graph
import grakel as gk
from grakel.utils import graph_from_networkx
from grakel.graph_kernels import GraphletSampling


def nx_to_gk(nx_graph):
    nodes = nx_graph.nodes()
    edges = []
    for e in nx_graph.edges():
        edges.append((e[0], e[1]))
        edges.append((e[1], e[0]))
    return gk.Graph(edges)


# filename = '../../data/bn-mouse-kasthuri/graph.json'
filename = 'backend/data/bn-mouse-kasthuri/graph.json'
with open(filename) as f:
    js_graph = json.load(f)
    nx_graph = json_graph.node_link_graph(js_graph)

#g = graph_from_networkx(nx_graph, as_Graph=True)

g = nx_to_gk(nx_graph)

glk = GraphletSampling()
mat = glk.fit_transform([g])

print(len(mat))
print(len(mat[0]))
