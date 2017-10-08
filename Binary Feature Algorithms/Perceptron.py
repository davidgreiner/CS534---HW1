import numpy as np
from binarizeData import BinarizeData

## Basic Perceptron algorithm for binary classification
## of individuals earning less than or more than 50K/year.

newdata, columns = BinarizeData()

weightVector = np.zeros((columns - 2))
bias = 0
epochCount = 1
totalEpoch = 5
numberTrainingData = len(newdata)
currentTrainingCount = 0

while epochCount <= totalEpoch:

    for i in range(0, numberTrainingData):

        if newdata[i][9] == 1:
            y = 1

        else:
            y = -1

        if y*(weightVector.dot(newdata[i][:-2]) + bias) <= 0:
            weightVector[:] = weightVector[:] + y*newdata[i][:-2]
            bias = bias + y

        currentTrainingCount += 1

    epochCount += 1
