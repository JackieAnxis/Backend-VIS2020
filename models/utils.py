import json
from networkx.readwrite import json_graph

def load_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

def save_json_graph(G, filename):
    js_graph = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(js_graph, f)