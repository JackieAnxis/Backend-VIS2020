from MT.deform import deform_v2
from MT.Graph import Graph
import numpy as np
from models.utils import load_json_graph, save_json_graph

def merge(G, target_pos, iter=1000, alpha=1, beta=0, gamma=200):
    V = deform_v2(G, target_pos, iter, alpha, beta, gamma)
    G.nodes = V
    return G

if __name__ == '__main__':
    prefix = './data/mammalia-voles-plj-trapping-25/'
    graph = load_json_graph(prefix + 'graph-with-pos.json')
    # nodes = [86, 18, 17, 21, 35, 36, 40]
    nodes = [3,1,2,6,5,0]
    sub = graph.subgraph(nodes)


    def modify(source_G, source_nodes):
        V = source_G.nodes
        n = V.shape[0]
        center = np.mean(V, axis=0)
        radius = np.mean(np.sqrt(np.sum((V - center) ** 2, axis=1)))
        interval = 2.0 * np.pi / n
        deformed_source_G = source_G.copy()
        for i in range(n):
            x = center[0] + radius * np.sin(interval * i)
            y = center[1] - radius * np.cos(interval * i)
            id = source_nodes[i]
            index = deformed_source_G.id2index[str(id)]
            deformed_source_G.nodes[index] = np.array([x, y])

        return deformed_source_G

    sub_G = Graph(sub)
    G = Graph(graph)
    sub_G = modify(sub_G, nodes)
    target_pos = {}
    for id in sub_G.id2index:
        target_pos[G.id2index[id]] = sub_G.nodes[sub_G.id2index[id]]
    G = merge(G, target_pos)
    save_json_graph(G.to_networkx(), prefix+'new.json')