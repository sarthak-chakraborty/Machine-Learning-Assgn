import numpy as np 
import pandas as pd 


# Calculates entropy of a list of list.
def calc_entropy(X, N):
	entropy = 0
	for i in range(len(X)):
		p = (float(len(X[i]))+np.finfo(float).eps)/N
		entropy += (-1)*p*np.log2(p)

	return entropy



# Calculates NMI value of a set of clusters
def calc_NMI(X, C, N):
	Hy = calc_entropy(X, N)	# Entropy of gold-standard data
	Hc = calc_entropy(C, N)	# Entropy of cluster

	# Inverse Mapping of the gold standard labels
	a = {}
	for key in X:
		for value in X[key]:
			a[value] = key

	# Calculate the entropy of data given the clusters
	Hyc = 0
	for x in C:
		Ycx = [[] for i in range(len(X))]
		p = float(len(x))/N
		for value in x:
			Ycx[a[value]].append(value)

		Hcyx = calc_entropy(Ycx, len(x))
		Hyc += p*Hcyx

	Iyc = Hy - Hyc 		# Mutul Information

	return (2*Iyc)/(Hy+Hc) 	# Return NMI




# Read data
df = pd.read_csv("../AAAI.csv")
HLD = [df.iloc[i,3] for i in range(len(df))]
N = len(HLD)



# Create gold standard clusters
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




# Load the clusters obtained from different algorithms
c1 = np.load("../Part1/Complete.npy")
c2 = np.load("../Part1/Single.npy")
c3 = np.load("../Part2/Graph.npy")



# Calculate NMI Values
nmi = calc_NMI(gold_standard, c1, N)
print("\nComplete Linkage Heirarchical Clustering: "+str(nmi))

nmi = calc_NMI(gold_standard, c2, N)
print("\nSingle Linkage Heirarchical Clustering: "+str(nmi))

nmi = calc_NMI(gold_standard, c3, N)
print("\nGirvan Newman Graph CLustering: "+str(nmi))

print("")