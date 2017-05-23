from sklearn.svm import LinearSVC
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_mutual_info_score
import numpy as np
import os

labels = ['default labels','Area','Distance from Origin','Distance to Image Border XY',
'Ellipticity (oblate)','Ellipticity (prolate)','Intensity Max Ch=1',
'Intensity Max Ch=2','Intensity Mean Ch=1','Intensity Mean Ch=2',
'Intensity Median Ch=1','Intensity Median Ch=2','Intensity Min Ch=1',
'Intensity Min Ch=2','Intensity StdDev Ch=1','Intensity StdDev Ch=2',
'Intensity Sum Ch=1','Intensity Sum Ch=2','Number of Vertices','PositionX',
'PositionY','PositionZ','Sphericity','Volume']
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

allData_X = allData - (sum(allData)/float(allData.shape[0]))
WT_Data_X = WT_Data - (sum(WT_Data)/float(WT_Data.shape[0]))
HET_Data_X = HET_Data - (sum(HET_Data)/float(HET_Data.shape[0]))

buckets_sizes = (np.amax(allData_X,axis=0) - np.amin(allData_X,axis=0))/15
print(buckets_sizes.shape)
# quant_func = np.vectorize(lambda x: x / buckets_sizes)
allData_quant_X = np.array(list(map(lambda x: (x / buckets_sizes).astype(int),allData_X)))
WT_Data_quant_X = np.array(list(map(lambda x: (x / buckets_sizes).astype(int),WT_Data_X)))
HET_Data_quant_X = np.array(list(map(lambda x: (x / buckets_sizes).astype(int),HET_Data_X)))
#+++++++++++++++LinearSVM for feature selection++++++++++++
# scores = []
# def svc_score(est,x,y):
# 	scores.append(est.coef_)
# 	return 0
# svc = LinearSVC(penalty='l1',dual=False)
# cross_val_score(svc,np.delete(allData_X,0,1),y=allData[:,[0]].ravel(),n_jobs=1,cv=5,scoring=svc_score)
# scores = scores**2
# coefs = (sum(scores)/float(len(scores)))
# coefs_sorted = enumerate(coefs[0])
# print(sorted(coefs_sorted,key=lambda x: x[1]))

# ++++++++++++++RandomizedLinearRegressiong+++++++++++++++
# scores = []
# def rlr_score(est,x,y):
# 	scores.append(est.scores_)
# 	print (len(scores))
# 	return 0
# rlr = RandomizedLogisticRegression(normalize=True,n_jobs=1)
# scr = cross_val_score(rlr,np.delete(allData_X,0,1),y=allData[:,[0]].ravel(),cv=5,n_jobs=1,scoring=rlr_score)
# coefs = (sum(scores)/float(len(scores)))
# coefs_sorted = enumerate(coefs[0])
# print(sorted(coefs_sorted,key=lambda x: x[1]))

# ++++++++++++++++++DecisionTree++++++++++++++++++++++++++++
# scores = []
# def dtc_score(est,x,y):
# 	scores.append(est.feature_importances_)
# 	return 0
# dtc = DecisionTreeClassifier(presort=True)
# cross_val_score(dtc,np.delete(allData,0,1),y=allData[:,[0]].ravel(),n_jobs=1,cv=5,scoring=dtc_score)
# coefs = (sum(scores)/float(len(scores)))
# coefs_sorted = enumerate(coefs)
# print(sorted(coefs_sorted,key=lambda x: x[1]))

# +++++++++++++++++++++MutualInfo++++++++++++++++++++++++++
MI = []
dataset = allData_quant_X
dataset = np.delete(dataset,0,1)
for i in range(dataset.shape[1]):
	for j in range(i,dataset.shape[1]):
		if i == j:
			continue		
		MI.append([labels[i],labels[j],adjusted_mutual_info_score(dataset[:,[i]].ravel(),dataset[:,[j]].ravel())])
MI = sorted(MI,key=lambda x:x[2])
np.savetxt("ALL_MI.txt",MI,fmt='%s',delimiter=',')		