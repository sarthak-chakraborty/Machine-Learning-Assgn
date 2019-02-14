import numpy as np 
import matplotlib.pyplot as plt
import math


sample = np.load("../Part1/dataset.npy")
weights = np.load("../Part1/weights.npy")
err_train = np.load("../Part1/train error.npy")
err_test = np.load("../Part1/test error.npy")
deg = [i for i in range(1,10)]

print("\nPlots the graph of the dataset along with the curve fit")
x = [sample[i][0] for i in range(len(sample))]
y = [sample[i][1] for i in range(len(sample))]

for i in range(len(deg)):
	plt.figure()
	plt.scatter(x,y)
	x1 = np.linspace(0, 1, 100)
	y1 = []
	for k in x1:
		a = 0
		for j in range(deg[i]+1):
			a += weights[i][j]*math.pow(k,j)
		y1.append(a)
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.title("REGRESSION LINE (DEGREE = "+str(deg[i])+")")
	plt.plot(x1, y1, "k")
	plt.savefig("Degree "+str(deg[i]))

plt.figure()
plt.plot(deg, err_train, "r")
plt.plot(deg, err_test, "k")
plt.xlabel("Degree of Regression Line")
plt.ylabel("Error Value")
plt.title("ERROR vs DEGREE of Regression Line")
plt.legend(["Train","Test"])
plt.xticks(deg)
plt.savefig("Error")
# plt.show()
print("Over..")