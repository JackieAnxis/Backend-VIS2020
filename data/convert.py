import csv
import json
import networkx as nx
from networkx.readwrite import json_graph

def connected_component_subgraphs(G):
    for c in nx.connected_components(G):
        yield G.subgraph(c)

# path = './bn-mouse-kasthuri/'
# path = './email-Eu-core/'
# path = './bio-DM-LC/'
path = './VIS/'

G = nx.read_edgelist(path + "graph.edgelist", nodetype=int,
                     data=(('weight', float),))
G.to_undirected()
max_connected_component = max(connected_component_subgraphs(G), key=len)

id_map = {}
i = 0
for node in max_connected_component:
    id_map[node] = i
    if not node == i:
        print('false')
    i += 1

H = nx.relabel_nodes(max_connected_component, id_map)

with open(path + 'id_map.csv', 'w') as f:
    spamwriter = csv.writer(f, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['raw_id', 'new_id'])
    for key in id_map:
        spamwriter.writerow([key, id_map[key]])

nx.write_edgelist(H, path + 'graph.edgelist', data=False)

data = json_graph.node_link_data(H)
with open(path + 'graph.json', 'w') as f:
    json.dump(data, f)
