import numpy as np 
import pandas as pd 



# Class that would perform the heirarchical clustering algorithm
class Cluster:
	# Initialize the class variables
	def __init__(self, linkage):
		self.linkage = linkage
		self.cluster = dict()


	# Takes two lists of items as inouts and returns the Jaccard Coefficient
	def compute_jaccard_coef(self, X, Y):
		intersection_count = len(list(set(X) & set(Y)))
		union_count = len(set(X).union(set(Y)))

		return float(intersection_count)/union_count


	# Takes a list of clusters and a list of topics as inputs and returns the the cluster index that has the highest similarity based on Jaccard Coefficient
	def find_similarity(self, X, Y):
		# Intialize variables
		max_coef = -1
		index1, index2 = -1, -1

		# For each pair of clusters
		for i in range(len(X)):
			for j in range(i+1, len(X)):
				max_point_similarity = -1
				min_point_similarity = 2

				# For each pair of documents among the two clusters
				for k in X[i]:
					for l in X[j]:
						# Calculate Jaccard Coefficient
						coef = self.compute_jaccard_coef(Y[k], Y[l])
						# Get maximum similarity and minimum similarity
						max_point_similarity = coef if(coef>max_point_similarity) else max_point_similarity
						min_point_similarity = coef if(coef<min_point_similarity) else min_point_similarity

				# Maximize the maximum similarity or minimum similarity depending on the type of linkage
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

		# Return the indexes of the cluster that are similar
		return index1, index2
			

	# Performs the heirarchical clustering
	def fit(self, X):
		i=len(X)
		# Initialize the class variable of cluster with each point as a cluster
		self.cluster[i] = [[j] for j in range(i)]

		# While number of clusters is 1
		while(i > 1):
			# Find the index of the current cluster that are most similar
			index1, index2 = self.find_similarity(self.cluster[i], X)

			# Update the class variable
			elem = list(self.cluster[i])
			elem.append(elem[index1] + elem[index2])
			del elem[index1]
			if(index1 < index2):
				del elem[index2-1]
			else:
				del elem[index2]
			self.cluster.update({i-1:elem})

			i -= 1
			




# Read data and store the topics for each document
df = pd.read_csv("../AAAI.csv")
topics = [df.iloc[i,2].split('\n')for i in range(len(df))]



print("\n")
print('##############################')
print("\tCOMPLETE LINKAGE")
print('##############################')
print('No. of Clusters = 9\n')

clf = Cluster('complete')
clf.fit(topics)

# Print clusters
c = clf.cluster[9]
for i in range(len(c)):
	print("Cluster " + str(i) + ": " + str(c[i]) + "\n")

# Save the clusters in a "npy" file to be used later
np.save("Complete", c)


print("\n\n")


print('##############################')
print("\tSINGLE LINKAGE")
print('##############################')
print('No. of Clusters = 9\n')

clf = Cluster('single')
clf.fit(topics)

# Print clusters
c = clf.cluster[9]
for i in range(len(c)):
	print("Cluster " + str(i) + ": " + str(c[i]) + "\n")

np.save("Single", c)