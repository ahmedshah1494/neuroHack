from sklearn.svm import LinearSVC
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.mixture import GaussianMixture
import numpy as np
import os

files = list(filter(lambda x: "csv" == x.split('.')[-1] and 'clean' in x, os.listdir(".")))
allData = None
WT_Data = None
HET_Data = None
for fname in files:
	data = np.loadtxt(fname)
	if allData == None:
		allData = data
	else:
		allData = np.concatenate((allData,data),axis=0)
	if "WT" in fname:
		if WT_Data == None:
			WT_Data = data
		else:
			WT_Data = np.concatenate((WT_Data,data),axis=0)
		allData = np.concatenate((allData,data),axis=0)
	if "HET" in fname:
		if HET_Data == None:
			HET_Data = data
		else:
			HET_Data = np.concatenate((HET_Data,data),axis=0)

#+++++++++++++++LinearSVM for feature selection++++++++++++
# svc = LinearSVC(penalty='l1',dual=False)
# svc.fit(np.delete(allData,0,1),allData[:,[0]].ravel())
# coefs = svc.coef_**2
# print(coefs)
# print(np.argmax(coefs),np.max(coefs))
# coefs_sorted = enumerate(coefs[0])
# print(sorted(coefs_sorted,key=lambda x: x[1]))

# ++++++++++++++RandomizedLinearRegressiong+++++++++++++++
# rlr = RandomizedLogisticRegression(normalize=True,n_jobs=1)
# rlr.fit(np.delete(allData,0,1),allData[:,[0]].ravel())
# print(rlr.scores_)
# print(np.argmax(rlr.scores_),np.max(rlr.scores_))

WT_gmm = GaussianMixture(n_components=1, covariance_type='full')
WT_gmm.fit(np.delete(WT_Data,0,1))
HET_gmm = GaussianMixture(n_components=1, covariance_type='full')
HET_gmm.fit(np.delete(HET_Data,0,1))

print(WT_gmm.covariances_)
print(HET_gmm.covariances_)