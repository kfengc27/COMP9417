import numpy as np
import pandas as pd
from scipy.io import arff
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from scipy.stats import pearsonr

# Function to scale data
def scale(a):
    b = (a-a.min())/(a.max()-a.min())
    return b

# Load Data
loadData = arff.loadarff('autoMpg.arff')
data = pd.DataFrame(loadData[0])

# Column Name
data.columns = ['cylinders','displacement','horsepower','weight','acceleration','model','origin','mpg']

# Remove any null value in hp
data = data[~pd.DataFrame(data.horsepower.tolist()).isnull().all(1)]

# Change data type
data.cylinders = data.cylinders.astype('int64')
data.model = data.model.astype('int64')
data.origin = data.origin.astype('int64')
data.weight = data.weight.astype('int64')

area = np.pi*3

# Text output
print(pearsonr(data['origin'],data['mpg']))

# Graphical output
'''
plt.scatter(data['weight'], data['mpg'], s=area, alpha=0.5)
plt.title('Weight vs Mpg')
plt.xlabel('Weight')
plt.ylabel('Mpg')

plt.show()
'''
