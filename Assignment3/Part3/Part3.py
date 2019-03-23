import numpy as np 
import pandas as pd 

df = pd.read_csv("../AAAI.csv")
HLD = [df.iloc[i,3] for i in range(len(df))]
N = len(HLD)

gold_standard = {}
a = set(HLD)
for i in range(len(a)):
	gold_standard[list(a)[i]] = []

for i in range(len(HLD)):
	gold_standard[HLD[i]].append(i)

counter = 0
for key in gold_standard:
	gold_standard[counter] = gold_standard.pop(key)
	counter += 1


c1 = np.load("../Part1/Complete.npy")
c2 = np.load("../Part1/Single.npy")



def calc_entropy(X, N):
	entropy = 0
	for i in range(len(X)):
		p = (float(len(X[i]))+np.finfo(float).eps)/N
		entropy += (-1)*p*np.log2(p)

	return entropy


def calc_NMI(X, C, N):
	Hy = calc_entropy(X, N)
	Hc = calc_entropy(C, N)

	a = {}
	for key in X:
		for value in X[key]:
			a[value] = key

	Hyc = 0
	for x in C:
		Ycx = [[] for i in range(len(X))]
		p = float(len(x))/N
		for value in x:
			Ycx[a[value]].append(value)

		Hcyx = calc_entropy(Ycx, len(x))
		Hyc += p*Hcyx

	Iyc = Hy - Hyc
	return (2*Iyc)/(Hy+Hc)


nmi = calc_NMI(gold_standard, c1, N)
print("Complete: "+str(nmi))

nmi = calc_NMI(gold_standard, c2, N)
print("Single: "+str(nmi))