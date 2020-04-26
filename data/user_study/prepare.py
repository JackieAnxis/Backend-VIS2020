import networkx as nx
from models.utils import load_json_graph, save_json_graph
from models.gk_weisfeiler_lehman import GK_WL

def relable_from_zero(g):
    i = 0
    label = {}
    for node in g.nodes:
        label[node] = i
        i += 1
    return nx.relabel_nodes(g, label)

names = ['email_small', 'email_star', 'highschool_circle', 'highschool_complex', 'road', 'vis']
wl = GK_WL()
for name in names:
    for i in range(4):
        path = './data/user_study/' + name + '/' + str(i) + '.json'
        G1 = load_json_graph(path)
        G1 = relable_from_zero(G1)
        for j in range(i + 1, 4):
            path = './data/user_study/' + name + '/' + str(j) + '.json'
            G2 = load_json_graph(path)
            G2 = relable_from_zero(G2)
            similarity = wl.compare(G1, G2, h=1, node_label=False)
            print(name, i, j, similarity)
        # layouted = nx.spring_layout(G, iterations=10)
        # for node in G.nodes:
        #     [x, y] = layouted[node]
        #     G.nodes[node]['x'] = x
        #     G.nodes[node]['y'] = y
        # save_json_graph(G, path)