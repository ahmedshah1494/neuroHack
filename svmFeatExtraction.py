# from sklearn.svm import LinearSVC
# from sklearn.linear_model import RandomizedLogisticRegression
# from sklearn.mixture import GaussianMixture
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import adjusted_mutual_info_score
import matplotlib.pyplot as plt
import numpy as np
import os

labels = ['default labels','Area','Distance from Origin','Distance to Image Border XY',
'Ellipticity (oblate)','Ellipticity (prolate)','Intensity Max Ch=1',
'Intensity Max Ch=2','Intensity Mean Ch=1','Intensity Mean Ch=2',
'Intensity Median Ch=1','Intensity Median Ch=2','Intensity Min Ch=1',
'Intensity Min Ch=2','Intensity StdDev Ch=1','Intensity StdDev Ch=2',
'Intensity Sum Ch=1','Intensity Sum Ch=2','Number of Vertices','PositionX',
'PositionY','PositionZ','Sphericity','Volume']
geometric_labels = ['Area','Distance from Origin','Distance to Image Border XY',
					'Number of Vertices','PositionX','PositionY','PositionZ',
					'Volume','default labels']
ch1_intensity_labels = ['Intensity Max Ch=1','Intensity Mean Ch=1','Intensity Median Ch=1',
						'Intensity Min Ch=1','Intensity StdDev Ch=1','Intensity Sum Ch=1']
ch2_intensity_labels = ['Intensity Max Ch=2','Intensity Mean Ch=2','Intensity Median Ch=2',
						'Intensity Min Ch=2','Intensity StdDev Ch=2','Intensity Sum Ch=2']											

def makeDataset():
	global allData
	global WT_Data 
	global HET_Data 
	global allData_quant_X 
	global WT_Data_quant_X 
	global HET_Data_quant_X 
	global allData_X 
	global WT_Data_X 
	global HET_Data_X 

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

	buckets_sizes = (np.amax(allData_X,axis=0) - np.amin(allData_X,axis=0))/32
	print(buckets_sizes.shape)
	# quant_func = np.vectorize(lambda x: x / buckets_sizes)
	allData_quant_X = np.array(list(map(lambda x: (x / buckets_sizes).astype(int),allData_X)))
	WT_Data_quant_X = np.array(list(map(lambda x: (x / buckets_sizes).astype(int),WT_Data_X)))
	HET_Data_quant_X = np.array(list(map(lambda x: (x / buckets_sizes).astype(int),HET_Data_X)))

# makeDataset()
#+++++++++++++++LinearSVM for feature selection++++++++++++
# dataset = HET_Data
# dataset_X = HET_Data_X
# scores0 = []
# def svc_score(est,x,y):
# 	scores0.append(est.coef_)
# 	return 0
# svc = LinearSVC(penalty='l1',dual=False)
# cross_val_score(svc,np.delete(dataset_X,0,1),y=dataset[:,[0]].ravel(),n_jobs=1,cv=5,scoring=svc_score)
# scores = np.array(scores0)**2
# scores0 = np.array(sum(scores)/float(len(scores)))


# # ++++++++++++++RandomizedLinearRegressiong+++++++++++++++
# scores1 = []
# def rlr_score(est,x,y):
# 	scores1.append(est.scores_)
# 	print (len(scores1))
# 	return 0
# rlr = RandomizedLogisticRegression(normalize=True,n_jobs=1)
# scr = cross_val_score(rlr,np.delete(dataset_X,0,1),y=dataset[:,[0]].ravel(),cv=5,n_jobs=1,scoring=rlr_score)
# scores1 = np.array(sum(scores1)/float(len(scores1)))

# # ++++++++++++++++++DecisionTree++++++++++++++++++++++++++++
# scores2 = []
# def dtc_score(est,x,y):
# 	scores2.append(est.feature_importances_)
# 	return 0
# dtc = DecisionTreeClassifier(presort=True)
# cross_val_score(dtc,np.delete(dataset_X,0,1),y=dataset[:,[0]].ravel(),n_jobs=1,cv=5,scoring=dtc_score)
# scores2 = np.array(sum(scores2)/float(len(scores2)))

# scores = (scores0 + scores1 + scores2)/3
# scores = enumerate(scores[0])
# scores = list(map(list,scores))
# print (scores)
# np.savetxt("HET_FEAT.txt",sorted(scores,key=lambda x: x[1]),fmt="%s")
# +++++++++++++++++++++MutualInfo++++++++++++++++++++++++++
def getMI():
	makeDataset()
	MI = []
	dataset = allData_quant_X
	dataset = np.delete(dataset,0,1)
	for i in range(dataset.shape[1]):
		for j in range(i,dataset.shape[1]):
			if i == j \
				or (labels[i] in geometric_labels and labels[j] in geometric_labels)\
				or (labels[i] in ch1_intensity_labels and labels[j] in ch1_intensity_labels)\
				or (labels[i] in ch2_intensity_labels and labels[j] in ch2_intensity_labels)\
				or ((labels[i] in ch1_intensity_labels and labels[j] in ch2_intensity_labels)\
					and ch1_intensity_labels.index(labels[i]) == ch2_intensity_labels.index(labels[j])):
				continue		
			MI.append([labels[i] + " & " + labels[j],adjusted_mutual_info_score(dataset[:,[i]].ravel(),dataset[:,[j]].ravel())])
	MI = sorted(MI,key=lambda x:x[-1])
	np.savetxt("ALL_MI.txt",MI,fmt='%s',delimiter=',')		

def loadMI(fname):
	data = np.loadtxt(fname,delimiter=',',dtype=str)
	for r in data:
		r[1] = r[1][2:-1]
	return data
def plotMI():
	ALL_data = loadMI('ALL_MI.txt')
	WT_data = loadMI('WT_MI.txt')
	HET_data = loadMI('HET_MI.txt')
	width = 0.35
	# labels = ALL_data[-6:][:,[0]].ravel()
	# data = ALL_data[-5:][:,[1]].astype(float).ravel()
	# plt.rcParams.update({'figure.autolayout': True})
	# fig, ax = plt.subplots()
	# rects1 = ax.bar(np.arange(5),data, 0.35, color='r')
	# ax.set_xticklabels(labels, rotation=90,size=8)
	# plt.savefig("ALL_plot.png")

	# xlabels = list(WT_data[-5:][:,[0]].ravel())
	# data1 = WT_data[-5:][:,[1]].astype(float).ravel()
	# data2 = []
	# for l in xlabels:
	# 	print(l)
	# 	for r in HET_data:
	# 		if l == r[0]:
	# 			data2.append(float(r[1]))
	# plt.rcParams.update({'figure.autolayout': True})
	# fig, ax = plt.subplots()
	# rects1 = ax.bar(np.arange(5), data1, width, color='b')
	# rects2 = ax.bar(np.arange(5) + width, data2, width, color='y')
	# ax.set_xticklabels([0] + xlabels, rotation=90,size=8)
	# ax.legend((rects1[0], rects2[0]), ('WT', 'HET'))
	# plt.savefig("WT_plot.png")

	xlabels = list(HET_data[-5:][:,[0]].ravel())
	data1 = HET_data[-5:][:,[1]].astype(float).ravel()
	data2 = []
	for l in xlabels:
		print(l)
		for r in WT_data:
			if l == r[0]:
				data2.append(float(r[1]))
	plt.rcParams.update({'figure.autolayout': True})
	fig, ax = plt.subplots()
	rects1 = ax.bar(np.arange(5), data2, width, color='b')
	rects2 = ax.bar(np.arange(5) + width, data1, width, color='y')
	ax.set_xticklabels([0] + xlabels, rotation=90,size=8)
	ax.legend((rects1[0], rects2[0]), ('WT', 'HET'))
	plt.savefig("HET_plot.png")

def plotFeat():
	feat = np.loadtxt('ALL_FEAT.txt')
	xlabels = list(map(lambda x: labels[int(x)], feat[-6:][:,[0]].ravel()))
	data = feat[-5:][:,[1]].astype(float).ravel()
	plt.rcParams.update({'figure.autolayout': True})
	fig, ax = plt.subplots()
	rects1 = ax.bar(np.arange(5),data, 0.35, color='r')
	ax.set_xticklabels(xlabels, rotation=90,size=8)
	ax.set_ylabel('Scores')
	plt.savefig("ALL_FEAT_plot.png")
# plotFeat()
plotMI()
# getMI()
