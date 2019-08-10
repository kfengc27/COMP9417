import numpy as np

# Personal scripts
import plots as plt
import linearRegression as lr
from helpers import *

"""
Train on training data in data folder

Find best theta, lambda, and degree of polynomial based on training and validation data.

Results are saved to ./params.npz for prediction.
"""

# Key Variable
learningRate = 0.03
 # Regularisation values
lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
 # degrees of polynoomial to try
polynomials = [0,1,2,3,4,5,6]
# Amount of training
iterations = 10000

'''
Task 1 : Find optimum value for polynomials and lambda
'''

bestPolynomial = None
bestLambdas = None
bestTheta = None
bestCost = None

# Record cost for each lambdas
costHistory = {}

# Go thru each polynomial
for p in polynomials:

  costHistory[p] = []

  # Header
  print("----------")
  print("p = " + str(p))

  for l in lambdas:
    print("---")
    print("Lambda = " + str(l) + "")

    # Training Data
    [X, y, _] = data_from_file('./data/trainData.data', p)

    # Find Theta
    theta = np.matrix(np.zeros([X.shape[1], 1]))
    [theta, cost_history] = lr.gradient_descent(X, y, theta, learningRate, l, iterations)

    # Validation Data
    [X, y, _] = data_from_file('./data/validationData.data', p)

    # Make prediction (MSE Error)
    predictions = X*theta

    # Find total cost
    validationCost = lr.cost(X, y, theta, l)

    if bestCost == None or validationCost < bestCost:
      bestCost = validationCost
      bestPolynomial = p
      bestLambdas = l
      bestTheta = theta

    costHistory[p].append(validationCost)

    print("CV cost = " + str(validationCost))

#plt.poly_lambda_error(lambdas, costHistory)

'''
    Task 1 Completed: We got the best value for our parameter
'''
p = bestPolynomial
l = bestLambdas
theta = bestTheta

print("Best lambda is: " + str(l) + ". Best p is " + str(p) + ". C.V. cost " + str(bestCost) + ".")

# Save learned lambda, p, and theta to file
np.savez('params', l=l, p=p, theta=theta)
