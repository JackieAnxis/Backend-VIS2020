import os
import json
import pickle
from networkx.readwrite import json_graph

def save_pickle_file(filename, file):
    with open(filename, 'wb') as f:
        pickle.dump(file, f)
        print("save {}".format(filename))


def load_pickle_file(filename):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            file = pickle.load(f)
        return file
    else:
        print("{} not exist".format(filename))


def load_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

def save_json_graph(G, filename):
    js_graph = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(js_graph, f)