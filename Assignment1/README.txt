NAME: Sarthak Chakraborty
ROLL: 16CS30044


###################################
			README
###################################

This readme file will tell you how to run the code and plot the graphs.




PART 1
------------------
Part 1 generates the sample and splits them. It then fits a regression curve for varying degrees using gradient descent and squarer error cost function and then estimates the error on training as well as test set.

Does not plot any graph.

The values of the weights and the error are stored in '.npy' files so that the plotting program can load the file easily.

'Dataset.csv' stores the data generated.

Number of samples can be changed by changing the value of 'n_sample'. Learning rate and degree of polynomial to fit can also be changed by changing the value of 'learning_rate' and 'deg'.

Supports only a single data size and a single learning rate. However, any number of degrees can be given and must be given in a list always.

Run using `python2 part1.py`.




PART 2
----------------
Loads the learned weights, dataset, and the error values from the '.npy' file saved in part 1. Must have the '.npy' file in the directory of Part1 for part 2 to run. If there is no such '.npy' file, then run part 1 before running part 2.

The directory of part 1 and part 2 must be kept intact.

Plots the dataset and the fitted curve for each degree of polynomial, and also the variation of train and test error with the degree of polynomial.

Run using `python2 part2.py`.




PART 3
----------------
Part 3 is same as part 1 but does the experiment with more than one sample size.

Supports more than 1 one dataset sizes. Number of samples can be changed by changing the value of 'n_sample'. However, the sample sizes must be given in a list.

Supports only a single value of learning rate (can be changed by changing the value of 'learning_rate'). However, any number of degrees can be given and must be given in a list always.

Plots the dataset along with the curve fitted, train and test error vs degree of polynomial for each sample size, and the learning curve.

Run unsing `python2 part3.py`.




PART 4
----------------
Part 4 performs gradient descent with different cost functions and varying learning rates.

'dataset.csv' contains the data generated.

Number of samples can be changed by changing the value of 'n_sample'. Learning rate and degree of polynomial to fit can also be changed by changing the value of 'learning_rate' and 'deg'.

Supports any number of values of learning rate, and any number of degrees can be given. Both, the learning rates and the degrees of polynomial must be given in a list always.

Weights are initialized randomly and their initialization values can be changed by modifying 'weights_square', 'weights_abs' and 'weights_power4'.

Read the report(marked as IMPORTANT) for explanation if changing the initial value of weights in 'weights_power4' results in overflow of value. Happens due to divergence of gradient descent.

Plots for all the degree and 3 cost functions are generated.

Plot for train and test error vs degree of polynomial for various cost functions is not generated. However it can be easily done by referring to its section in 'part3.py' ot 'part4.py'.

Plots of RMSE vs Learning rate is generated for all the degree of polynomials of the fitted curve.

Run using `python2 part4.py`.




GENERAL
---------------
Read the comments for a better walkthrough of the code. The function of calculating the gradient is completely vectorised and done using matrix multiplication instead of running a loop. The results generated are non-reproducible since no random seed is added in the code (not mentioned in PS). However on running the code, a general idea of the results can be found and compared with my results.



#####################################################################################################################################