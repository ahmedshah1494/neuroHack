from sklearn.svm import LinearSVC
from sklearn.linear_model import RandomizedLogisticRegression
import numpy as np
import os

files = list(filter(lambda x: "csv" == x.split('.')[-1] and 'clean' in x, os.listdir(".")))
allData = None
for fname in files:
	data = np.loadtxt(fname)
	if allData == None:
		allData = data
	else:
		allData = np.concatenate((allData,data),axis=0)

#+++++++++++++++LinearSVM for feature selection++++++++++++
# svc = LinearSVC(penalty='l1',dual=False)
# svc.fit(np.delete(allData,0,1),allData[:,[0]].ravel())
# print(svc.coef_)
# print(np.argmax(svc.coef_),np.max(svc.coef_))

# ++++++++++++++RandomizedLinearRegressiong+++++++++++++++
rlr = RandomizedLogisticRegression(normalize=True,n_jobs=1)
rlr.fit(np.delete(allData,0,1),allData[:,[0]].ravel())
print(rlr.scores_)
print(np.argmax(rlr.scores_),np.max(rlr.scores_))
