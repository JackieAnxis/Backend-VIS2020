import networkx as nx
from models.layout import layout
from models.utils import save_json_graph

if __name__ == '__main__':
    # #node = 5, 10, 30, 50
    # #density = 1, 5, 10, 20
    G = nx.Graph()
    for node_count in [5, 10, 30, 50]:
        for density in [1, 5, 10, 20]:
            edge_count = node_count * density
            begin = len(G.nodes)
            g = nx.gnm_random_graph(node_count, edge_count)
            g = nx.relabel_nodes(g, lambda x: x + len(G.nodes))
            G = nx.union(G, g)

    G = layout(G)
    save_json_graph(G, './data/synthetic/graph-with-pos.json')