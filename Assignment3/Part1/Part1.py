import numpy as np 
import pandas as pd 


df = pd.read_csv("../AAAI.csv")
topics = [df.iloc[i,2].split('\n')for i in range(len(df))]



class Cluster:

	def __init__(self, linkage):
		self.linkage = linkage
		self.cluster = dict()


	def compute_jaccard_coef(self, X, Y):
		intersection_count = len(list(set(X) & set(Y)))
		union_count = len(set(X).union(set(Y)))

		return float(intersection_count)/union_count


	def find_similarity(self, X):
		max_coef = -1
		max_point_similarity = -1
		min_point_similarity = -1
		index1, index2 = -1, -1

		for i in range(len(X[0])):
			for j in range(i+1, len(X[0])):
				max_point_similarity = -1
				min_point_similarity = 2
				for k in X[0][i]:
					for l in X[0][j]:
						coef = self.compute_jaccard_coef(X[1][k], X[1][l])
						max_point_similarity = coef if(coef>max_point_similarity) else max_point_similarity
						min_point_similarity = coef if(coef<min_point_similarity) else min_point_similarity

				if(self.linkage == 'single'):
					if(max_point_similarity > max_coef):
						index1 = i
						index2 = j
						max_coef = max_point_similarity
				elif(self.linkage == 'complete'):
					if(min_point_similarity > max_coef):
						index1 = i
						index2 = j
						max_coef = min_point_similarity

		return index1, index2
			

	def fit(self, X):
		i=len(X)
		count = 150
		self.cluster[i] = ([[j] for j in range(i)], X)
		while(i > 1):
			index1, index2 = self.find_similarity(self.cluster[i])

			elem1 = list(self.cluster[i][0])
			elem1.append(list(set(elem1[index1]).union(set(elem1[index2]))))
			del elem1[index1]
			if(index1 < index2):
				del elem1[index2-1]
			else:
				del elem1[index2]
			elem2 = list(self.cluster[i][1])

			self.cluster.update({i-1:(elem1, elem2)})

			i -= 1
			








print("\n")
print('##############################')
print("\tCOMPLETE LINKAGE")
print('##############################')
print('No. of Clusters = 9\n')

clf = Cluster('complete')
clf.fit(topics)

c = clf.cluster[9][0]
for i in range(len(c)):
	print("Cluster " + str(i) + ": " + str(c[i]) + "\n")

np.save("Complete", c)


print("\n\n")


print('##############################')
print("\tSINGLE LINKAGE")
print('##############################')
print('No. of Clusters = 9\n')

clf = Cluster('single')
clf.fit(topics)

c = clf.cluster[9][0]
for i in range(len(c)):
	print("Cluster " + str(i) + ": " + str(c[i]) + "\n")

np.save("Single", c)