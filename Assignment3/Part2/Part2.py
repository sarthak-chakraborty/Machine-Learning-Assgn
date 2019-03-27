import numpy as np 
import pandas as pd 
import networkx as nx 
import matplotlib.pyplot as plt 
import operator





# Class that would perform the Girvan Newman clustering algorithm
class Clustering:
	# Initialize the class variable
	def __init__(self):
		self.cluster = dict()


	# Performs the clustering algorithm. Takes the Graph G and total N points as input
	def fit(self, G, N):
		n = nx.number_connected_components(G)	# Find the number of connected components

		# Initialize the class variable of cluster with each point as a cluster
		components = nx.connected_components(G)
		self.cluster[n] = []
		for nodes in components:
			self.cluster[n].append(list(nodes))

		# while all points are identified as separate cluster
		while(n < N):
			edge_betweenness = nx.edge_betweenness_centrality(G)	# Calculate the betweenness centrality for all edges
			edge = max(edge_betweenness, key=edge_betweenness.get)	# Select the esge with maximum betweenness centrality
			G.remove_edge(edge[0], edge[1])							# Remove the edge from the graph

			new_n = nx.number_connected_components(G)	# Recompute number of connected components

			# If number of connected componensts increase, then update class variable
			if(new_n > n):
				components = nx.connected_components(G)
				self.cluster[new_n] = []
				for nodes in components:
					self.cluster[new_n].append(list(nodes))

			n = new_n





# Takes two lists of items as inouts and returns the Jaccard Coefficient
def compute_jaccard_coef(X, Y):
	intersection_count = len(list(set(X) & set(Y)))
	union_count = len(set(X).union(set(Y)))

	return float(intersection_count)/union_count





# Read data
df = pd.read_csv("../AAAI.csv")
topics = [df.iloc[i,2].split('\n')for i in range(len(df))]




G = nx.Graph()	# Create Graph

# Add nodes to the graph
for i in range(len(topics)):
	G.add_node(i)

# Add edges to the graph
nodes = G.nodes()
for i in range(len(nodes)):
	for j in range(i+1, len(nodes)):
		coef = compute_jaccard_coef(topics[i], topics[j])
		if(coef > 0.21):
			G.add_edge(i, j)

nx.draw(G)
plt.draw()
plt.title("Original Network")
plt.savefig("Input.png")




# Perform clustering
clf = Clustering()
clf.fit(G, len(topics))

print("\n")
c = clf.cluster[9]
for i in range(len(c)):
	print("Cluster " + str(i) + ": " + str(c[i]) + "\n")

np.save("Graph", c)




# Create output graph
G1 = nx.Graph()

# Add nodes to the graph
for i in range(len(topics)):
	G.add_node(i)

# Add edge to the new graph
for cluster in c:
	for i in range(len(cluster)):
		for j in range(i+1, len(cluster)):
			coef = compute_jaccard_coef(topics[cluster[i]], topics[cluster[j]])
			if(coef > 0.21):
				G1.add_edge(cluster[i], cluster[j])

plt.figure()
nx.draw(G1)
plt.draw()
plt.title("Network with 9 clusters")
plt.savefig("Output.png")
