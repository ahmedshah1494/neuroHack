import os

files = list(filter(lambda x: "csv" == x.split('.')[-1] and 'clean' in x, os.listdir(".")))
WT_files = list(filter(lambda x: "WT" in x, files))
HET_files = list(filter(lambda x: "HET" in x, files))
for i in range(len(WT_files)):
	fld = [WT_files[i],HET_files[i]]
	