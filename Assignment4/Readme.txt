NAME: Sarthak Chakraborty
ROLL: 16CS30044


###################################
	     README
###################################

This readme file will tell you how to run the code.


Part 1
---------
1. 'part1.py' present in 'Part1' folder performs the implementation of the neural network described in the first part of the assignment.

2. The specifications of the neural networks are declared at the top of the program.

3. Batch Size can be changed by changing the value of the variable `batch_size`.

4. The dataset gets stored in '.npy' format in the same directory. You can load it by `np.load()`. Uncomment the commented portion to load the '.npy' file after commenting `preprocess(x_train, y_train, x_test, y_test)`.

5. Run the code using `python3 part1.py`.




Part 2
---------
1. 'Part2.py' present in 'Part2' folder performs the implementation of the neural network described in the first part of the assignment.

2. The specifications of the neural networks are declared at the top of the program.

3. Batch Size can be changed by changing the value of the variable `batch_size`.

4. 2 integer command line inputs are necessary to run the program. The input dentoes the number of neurons in each of the 2 hidden layers.

4. The dataset gets stored in '.npy' format in the same directory. You can load it by `np.load()`. Uncomment the commented portion to load the '.npy' file after commenting `preprocess(x_train, y_train, x_test, y_test)`.

5. Run the code using `python3 Part2.py <input1> <input2>`.


########################################################################################