import os
import numpy as np
import networkx as nx

# dataset = 'bn-mouse-kasthuri'
# dataset = 'email-Eu-core'
# dataset = 'bio-DM-LC'
dataset = 'VIS'

prefix = '../data/'
python = 'python3'

# role2vec
# os.system(python + ' ./role2vec/main.py --graph-input ' + prefix + dataset + '/graph.edgelist --output ' + prefix + dataset + '/role2vec.csv')
os.system(python + ' ./graphwave/main.py --edgelist-input --input ' + prefix + dataset + '/graph.edgelist --output ' + prefix + dataset + '/graphwave.csv')
# os.system(python + ' ./walklets/main.py --edgelist-input --input ' + prefix + dataset + '/graph.edgelist --output ' + prefix + dataset + '/wavelets.csv')
