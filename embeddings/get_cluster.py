import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
# import matplotlib.pyplot as plt
import pandas as pd


def get_cluster_label():
  mycsv=pd.read_csv('./data/test/graphwave/approximation50.csv', index_col=0)
  mylist=mycsv.values.tolist()
  X = np.array(mylist)
  db = skc.DBSCAN(eps=0.04, min_samples=2).fit(X)
  labels = db.labels_  #和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声
  # print('每个样本的簇标号:')
  # print(labels)

  raito = len(labels[labels[:] == -1]) / len(labels)  #计算噪声点个数占总数的比例
  print('noise rate:', format(raito, '.2%'))
  n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
  print('cluster num: %d' % n_clusters_)
  # print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels))
  return labels.tolist()
  # for i in range(n_clusters_):
      # print('簇 ', i, '的所有样本:')
      # one_cluster = X[labels == i]
      # print(one_cluster)
      # plt.plot(one_cluster[:,0],one_cluster[:,1],'o')

  # plt.show()