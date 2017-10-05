import numpy as np
from binarizeData import BinarizeData
from Dev_Evaluator import DevEvaluator

## Basic Perceptron algorithm for binary classification
## of individuals earning less than or more than 50K/year.

featureArray, trainDataArray = BinarizeData("train")
devDataArray = BinarizeData("dev")

weightVector = np.zeros((len(featureArray)))
epochCount = 1
totalEpoch = 2
numberTrainingData = len(trainDataArray)
currentTrainingCount = 0

while epochCount <= totalEpoch:

    for i in range(0, numberTrainingData):

        if currentTrainingCount % 1000 == 0:

            devError = DevEvaluator(weightVector,
                                    featureArray, devDataArray)

            epochFraction = (i / numberTrainingData) * epochCount

            print("The error rate for epoch " + str(epochFraction) + \
                  " is " + str(devError) + "%")

        if trainDataArray[i, -1] == '>50K':
            y = 1

        else:
            y = -1

        idx = np.isin(featureArray, trainDataArray[i, 0:-1])

        if y*(weightVector[idx].sum() + weightVector[-1]) <= 0:
            
            weightVector[idx] = weightVector[idx] + \
            y*np.ones((len(trainDataArray[i, 0:-1])))

            weightVector[-1] = weightVector[-1] + y

        currentTrainingCount += 1

    epochCount += 1
