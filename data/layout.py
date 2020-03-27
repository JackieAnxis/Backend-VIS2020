from tulip import tlp
import json
from networkx.readwrite import json_graph
import networkx as nx

def load_json_graph(filename):
    with open(filename) as f:
        js_graph = json.load(f)
    return json_graph.node_link_graph(js_graph)

def save_json_graph(G, filename):
    js_graph = json_graph.node_link_data(G)
    with open(filename, 'w') as f:
        json.dump(js_graph, f)

def nx2tlp(G):
    graph = tlp.newGraph()
    nodes_map = {}
    for node in G.nodes:
        nodes_map[node] = graph.addNode()
    for edge in G.edges:
        graph.addEdge(nodes_map[edge[0]], nodes_map[edge[1]])
    return graph, nodes_map

def overlap_removal(graph, layout):
    # get a dictionnary filled with the default plugin parameters values
    # graph is an instance of the tlp.Graph class
    params = tlp.getDefaultPluginParameters('Fast Overlap Removal', graph)

    # set any input parameter value if needed
    # params['overlap removal type'] = ...
    params['layout'] = layout
    # params['bounding box'] = ...
    # params['rotation'] = ...
    # params['number of passes'] = ...
    # params['x border'] = ...
    # params['y border'] = ...

    # either create or get a layout property from the graph to store the result of the algorithm
    resultLayout = graph.getLayoutProperty('resultLayout')
    success = graph.applyLayoutAlgorithm('Fast Overlap Removal', resultLayout, params)

    return resultLayout

def SM(tlpgraph):
    params = tlp.getDefaultPluginParameters('Stress Majorization (OGDF)', tlpgraph)
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    success = tlpgraph.applyLayoutAlgorithm('FM^3 (OGDF)', resultLayout, params)

    # or store the result of the algorithm in the default Tulip layout property named 'viewLayout'
    # success = graph.applyLayoutAlgorithm('FM^3 (OGDF)', params)
    return resultLayout

def Orthotree(tlpgraph):
    # get a dictionnary filled with the default plugin parameters values
    # graph is an instance of the tlp.Graph class
    params = tlp.getDefaultPluginParameters('MMM Example Nice Layout (OGDF)', tlpgraph)

    # set any input parameter value if needed
    # params['Layer spacing'] = ...
    # params['Node spacing'] = ...

    # either create or get a layout property from the graph to store the result of the algorithm
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    success = tlpgraph.applyLayoutAlgorithm('MMM Example Nice Layout (OGDF)', resultLayout, params)
    return resultLayout

def FM3(tlpgraph):
    # get a dictionnary filled with the default plugin parameters values
    # graph is an instance of the tlp.Graph class
    params = tlp.getDefaultPluginParameters('FM^3 (OGDF)', tlpgraph)

    # set any input parameter value if needed
    # params['Edge Length Property'] = ...
    # params['Node Size'] = ...
    # params['Unit edge length'] = ...
    # params['New initial placement'] = ...
    # params['Fixed iterations'] = ...
    # params['Threshold'] = ...
    # params['Page Format'] = ...
    params['Quality vs Speed'] = 'GorgeousAndEfficient'
    # params['Edge Length Measurement'] = ...
    # params['Allowed Positions'] = ...
    # params['Tip Over'] = ...
    # params['Pre Sort'] = ...
    # params['Galaxy Choice'] = ...
    # params['Max Iter Change'] = ...
    # params['Initial Placement Mult'] = ...
    # params['Force Model'] = ...
    # params['Repulsive Force Method'] = ...
    # params['Initial Placement Forces'] = ...
    # params['Reduced Tree Construction'] = ...
    # params['Smallest Cell Finding'] = ...

    # either create or get a layout property from the graph to store the result of the algorithm
    resultLayout = tlpgraph.getLayoutProperty('resultLayout')
    success = tlpgraph.applyLayoutAlgorithm('FM^3 (OGDF)', resultLayout, params)

    # or store the result of the algorithm in the default Tulip layout property named 'viewLayout'
    # success = graph.applyLayoutAlgorithm('FM^3 (OGDF)', params)
    return resultLayout

def layout(G):
    graph, nodes_map = nx2tlp(G)
    # resultLayout = FM3(graph)
    # resultLayout = SM(graph)
    resultLayout = Orthotree(graph)
    resultLayout = overlap_removal(graph, resultLayout)
    for n in G.nodes:
        pos = resultLayout[nodes_map[n]]
        G.nodes[n]['x'] = pos[0]
        G.nodes[n]['y'] = pos[1]
    return G

if __name__ == '__main__':
    # path = './bn-mouse-kasthuri/'
    path = './power-662-bus/'
    filename = path + "graph.edgelist"
    G = nx.read_edgelist(filename, nodetype=int, data=(('weight', float),))
    G.to_undirected()
    G = layout(G)
    save_json_graph(G, path+'graph-with-pos.json')