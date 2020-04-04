import csv
import json
import networkx as nx

data_path = 'IEEE VIS papers 1990-2018 - Main dataset.tsv'
f = open(data_path, 'r')
reader = csv.reader(f, delimiter='\t')
lines = list(reader)
f.close()

key2index = {}
edges = []
author2id = {}
for i in range(0, len(lines)):
    line = lines[i]
    if i == 0:
        for i in range(0, len(line)):
            key = line[i]
            key2index[key] = i
    else:
        if int(line[key2index['Year']]) >= 2010:
            authors = line[key2index['AuthorNames-Deduped']].split(';')
            for x in range(0, len(authors)):
                author = line[key2index['Year']] + '-' + authors[x]
                if author not in author2id:
                        author2id[author] = len(author2id)
            # for y in range(1, len(authors)):
            #     edges.append([author2id[authors[0]], author2id[authors[y]]])
            for x in range(0, len(authors)):
                author0 = line[key2index['Year']] + '-' + authors[x]
                for y in range(x + 1, len(authors)):
                    author1 = line[key2index['Year']] + '-' + authors[y]
                    edges.append([author2id[author0], author2id[author1]])
print(author2id)
f = open('author2id.csv', 'w')
writer = csv.writer(f, delimiter=',')
writer.writerow(['name', 'id'])
for author in author2id:
    writer.writerow([author, author2id[author]])
f.close()

f = open('graph.edgelist', 'w')
writer = csv.writer(f, delimiter=' ')
for edge in edges:
    writer.writerow(edge)
f.close()