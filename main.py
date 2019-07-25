from scipy.io import arff
import pandas as pd
import math, numpy

data,meta = arff.loadarff('autoMpg.arff')

print(meta)

# Train
for i in data: 

	# All variables
	cylinder = int(i[0])

	displacement = float(i[1])

	horsepower = 0
	if not math.isnan(i[2]) :
		horsepower = int(i[2])	

	weight = int(i[3])

	acceleration = float(i[4])

	model = int(i[5])

	carClass = int(i[6])

	# Weight

	print(data)