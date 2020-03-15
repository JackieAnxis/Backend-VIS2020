import karateclub as kc
import networkx as nx
import pandas as pd
import numpy as np

dataset = 'bn-mouse-kasthuri'
prefix = '../data/'
outfile = f'{prefix}{dataset}/graphwave2.csv'


model = kc.GraphWave()
graph = nx.read_edgelist(f'{prefix}{dataset}/graph.edgelist')
model.fit(graph)
embed = model.get_embedding()

print(embed.shape)
print(embed[0][:5])

# np.savetxt(f'{prefix}{dataset}/graphwave2.csv', embed, delimiter=',')
pd.DataFrame(embed).to_csv(outfile)
