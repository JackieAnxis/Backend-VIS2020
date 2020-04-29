import csv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

with open('interaction.csv') as f:
    spamreader = csv.reader(f, delimiter=',')
    i = 0
    j2m = {}
    data = []
    for row in spamreader:
        if i == 0:
            j = 0
            for m in row:
                j2m[j] = m
                j += 1
        else:
            j = 0
            for time in row:
                data.append([j2m[j], int(time)])
                j += 1
        i += 1

with open('interactiondict.csv', 'w') as f:
    spamwriter = csv.writer(f, delimiter=',')
    spamwriter.writerow(['method', 'time'])
    for row in data:
        spamwriter.writerow(row)

sns.set(style="darkgrid")
time = pd.read_csv('./interactiondict.csv')
tips = sns.load_dataset("tips")
ax = sns.pointplot(y="method", x="time", data=time, join=False, orient='h')
print('xxx')