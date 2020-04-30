import json
from statistics import mean

with open('graph-with-pos.json') as f:
    data = json.load(f)

nodesX = list(map(lambda  n: n['x'], data['nodes']))
nodesY = list(map(lambda  n: n['y'], data['nodes']))


mx = mean(nodesX)
my = mean(nodesY)

for i in range(len(data['nodes'])):
    data['nodes'][i]['x'] = 2 * mx - data['nodes'][i]['x'] 
    data['nodes'][i]['y'] = 2 * my - data['nodes'][i]['y'] 

with open('graph-with-pos-invert.json', 'w') as f:
    json.dump(data, f)