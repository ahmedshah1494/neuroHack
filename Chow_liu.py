import itertools

from sklearn.model_selection import cross_val_score
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import sparse

from sklearn.metrics import hamming_loss
from sklearn.datasets import fetch_mldata
from sklearn.metrics import mutual_info_score
from scipy.sparse.csgraph import minimum_spanning_tree

# from pystruct.learners import OneSlackSSVM
# from pystruct.models import MultiLabelClf
# from pystruct.datasets import load_scene


files = list(filter(lambda x:'clean' in x and "py" not in x,  os.listdir("neuroHack")))
print files

dataset = np.empty((0,25))
for f in files:
	l = np.loadtxt("neuroHack/"+f)
	print "f",f,l.shape
	dataset = np.append(dataset, l, axis=0)

labels = dataset[:,0]
data = dataset[:,1:]

dataset_het = np.where(dataset[:,0]==1)
dataset_het = dataset[dataset_het[0]]
labelsHet = dataset_het[:,0]
dataHet = dataset_het[:,1:]

dataset_hom = np.where(dataset[:,0]==0)
dataset_hom = dataset[dataset_hom[0]]
labelsHom = dataset_hom[:,0]
dataHom = dataset_hom[:,1:]



print labels.shape
print data.shape

def chow_liu_tree(y_):
    # compute mutual information using sklearn
    n_labels = y_.shape[1]
    mi = np.zeros((n_labels, n_labels))
    for i in range(n_labels):
        for j in range(n_labels):
            mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    edges = np.vstack(mst.nonzero()).T
    edges.sort(axis=1)
    return edges

tree = chow_liu_tree(data)+1
treeHet = chow_liu_tree(dataHet)+1
treeHom = chow_liu_tree(dataHom)+1

print tree
print "\n\n\n Heterozygotes"
print treeHom
print "\n\n\n Homozygotes"
print treeHet

