import numpy as np 
import matplotlib.pyplot as plt
import math

# Load the already learned weights and the estimated error from the npy file
sample = np.load("../Part1/dataset.npy")
weights = np.load("../Part1/weights.npy")
err_train = np.load("../Part1/train error.npy")
err_test = np.load("../Part1/test error.npy")
deg = [i for i in range(1,10)]

print("\nPlots the graph of the dataset along with the curve fit")
x = [sample[i][0] for i in range(len(sample))]
y = [sample[i][1] for i in range(len(sample))]

# Generates the regression curve and plots it
for i in range(len(deg)):
	plt.figure()
	plt.scatter(x,y)
	x1 = np.linspace(0, 1, 100)
	y1 = []
	for k in x1:	# finds the value of the function using x values and the parameters of the function
		a = 0
		for j in range(deg[i]+1):
			a += weights[i][j]*math.pow(k,j)
		y1.append(a)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("REGRESSION LINE (DEGREE = "+str(deg[i])+")")
	plt.plot(x1, y1, "k")
	# plt.savefig("Degree "+str(deg[i]))

# Plots the train and test error vs Degree of polynomial
plt.figure()
plt.plot(deg, err_train, "b")
plt.plot(deg, err_test, "r")
plt.xlabel("Degree of Polynomial")
plt.ylabel("Error Value")
plt.title("ERROR vs DEGREE of Regression Line")
plt.legend(["Train","Test"])
plt.xticks(deg)
# plt.savefig("Error")
plt.show()
print("Over..")