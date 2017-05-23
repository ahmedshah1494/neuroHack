import numpy as np
from sklearn.model_selection import cross_val_score
import os
from sklearn.ensemble import RandomForestClassifier

files = list(filter(lambda x:'clean' in x and "py" not in x,  os.listdir("neuroHack")))
print files

dataset = np.empty((0,25))
for f in files:
	l = np.loadtxt("neuroHack/"+f)
	print "f",f,l.shape
	dataset = np.append(dataset, l, axis=0)

scores = []
def rlr_score(est,x,y):
	scores.append(est.feature_importances_)
	print (len(scores))
	return 0

clf = RandomForestClassifier(n_estimators=16, max_features= None, verbose = 1)
# clf = clf.fit(X, Y)


scr = cross_val_score(clf,np.delete(dataset,0,1),y=dataset[:,[0]].ravel(),cv=5,n_jobs=1,scoring=rlr_score)
ahmed_is_funny = sum(scores)/float(len(scores))

coefs_sorted = enumerate(ahmed_is_funny)
print(sorted(coefs_sorted,key=lambda x: x[1]))



# [(18, 0.00048362857733763576), (17, 0.00075028238756564441), 
# (23, 0.00080911727523198655), (5, 0.00088288073429002684), 
# (16, 0.00093270591574053303), (15, 0.0010325469978474608), 
# (7, 0.0010874747703830039), (1, 0.0016278558981464024), 
# (6, 0.0016482657169113323), (4, 0.0017163707329705049), 
# (10, 0.001732137038797613), (12, 0.0020634363121831657), 
# (9, 0.0026389433022615556), (11, 0.0035805979305658388), 
# (22, 0.0048282197283189138), (8, 0.013846264428880742), 
# (14, 0.014590273632110937), (13, 0.023015825049393396), 
# (20, 0.032183285335646482), (21, 0.050613530593441272), 
# (19, 0.061741041211648626), (2, 0.13307312110800426), 
# (3, 0.16903752334295039), (0, 0.47608467197937221)]

