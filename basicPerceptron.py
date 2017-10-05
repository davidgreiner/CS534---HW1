import numpy as np
from binarizeData import BinarizeData

## Basic Perceptron algorithm for binary classification
## of individuals earning less than or more than 50K/year.

featureArray, dataArray = BinarizeData()

weightVector = np.zeros((len(featureArray)))
bias = 0
epochCount = 0
totalEpoch = 2
numberTrainingData = len(dataArray)

while epochCount < totalEpoch:

    for i in range(0, numberTrainingData):

        if dataArray[i, -1] == '>50K':
            y = 1

        else:
            y = -1

        idx = np.isin(featureArray, dataArray[i, 0:-1])

        if y*(weightVector[idx].sum()) <= 0:
            
            weightVector[idx] = weightVector[idx] + \
            y*np.ones((len(dataArray[i, 0:-1])))

            bias = bias + y

    epochCount += 1
