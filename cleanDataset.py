import numpy as np
import os

def list2int(l,val):
	return l.index(val)
files = list(filter(lambda x: "csv" == x.split('.')[-1] and 'clean' not in x, os.listdir(".")))
dic = {}
for fname in files:
	data = np.loadtxt(fname,skiprows=2,delimiter=',',dtype=str)
	for r in data:
		dic[r[1][2:-1]] = 0
print(dic)
diclabels = list(dic.keys())
for fname in files:
	data = np.loadtxt(fname,skiprows=2,delimiter=',',converters={1:(lambda x: diclabels.index(str(x)[2:-1]))})
	np.savetxt(fname[:-5]+"_clean.csv",np.delete(data,0,1))
