import numpy as np

# Personal scripts
import plots as plt
import linearRegression as lr
from helpers import *

"""
Predict mpg for test set generated from autoMpg.
Run trainMpg.py before this to get best outcome.
"""

# Load in learned params
with np.load('params.npz') as data:
    l = data['l']
    p = data['p']
    theta = data['theta']

# Load in test set
[XTest, YTest, Data] = data_from_file('./data/testData.data', p)

# Make predictions
predictions = XTest * theta
testCost = lr.cost(XTest, YTest, theta, l)
diff = np.abs(YTest - predictions).round(2)

# Show results
print('Car: Predicted, Actual, difference')

# Round prediction for output
rounded = predictions.round(2)

# Output prediction
for i in range(0, predictions.shape[0]):
    print("Car " + str(i) + " : " + str(rounded[i, 0]) + ", " + str(YTest[i, 0]) + ", " + str(diff[i, 0]))

# Display accuracy of our model
plt.prediction_accuracy(predictions, YTest)
print("Test cost = " + str(testCost))
