import numpy as np
import time
from binarizeData import BinarizeData
from Dev_Evaluator import DevEvaluator

## Averaged Smart Perceptron algorithm for binary classification
## of individuals earning less than or more than 50K/year.

featureArray, trainDataArray = BinarizeData("train")
devDataArray = BinarizeData("dev")

weightVector = np.zeros((len(featureArray)))
weightVectorAveraged = np.zeros((len(featureArray)))
epochCount = 1
totalEpoch = 5
numberTrainingData = len(trainDataArray)
currentTrainingCount = 1
bestErrorRate = 100.0
epochIteration = 0

startTime = time.time()

while epochCount <= totalEpoch:

    for i in range(0, numberTrainingData):

        if currentTrainingCount % 1000 == 0:

            devError = DevEvaluator(weightVector - (weightVectorAveraged / currentTrainingCount), \
                                    featureArray, devDataArray)

            epochFraction = (i / numberTrainingData) * epochCount

            if devError < bestErrorRate:
                bestErrorRate = devError
                epochIteration = epochFraction

            print("The error rate for epoch " + str(epochFraction) + \
                  " is " + str(devError) + "%")

        if trainDataArray[i, -1] == '>50K':
            y = 1

        else:
            y = -1

        idx = np.isin(featureArray, trainDataArray[i, 0:-1])

        if y*(weightVector[idx].sum() + weightVector[-1]) <= 0:
            
            weightVector[idx] = weightVector[idx] + \
            y*np.ones(len(trainDataArray[i, 0:-1]))

            weightVector[-1] = weightVector[-1] + y

            weightVectorAveraged[idx] = weightVectorAveraged[idx] + \
            currentTrainingCount * y * np.ones(len(trainDataArray[i, 0:-1]))

            weightVectorAveraged[-1] = weightVectorAveraged[-1] + y

        currentTrainingCount += 1

    epochCount += 1

print("The program ran for %s seconds" % (time.time() - startTime))
print("The best error rate was " + str(bestErrorRate) + " at epoch " + \
      str(epochIteration))
positiveFeatures = weightVectorAveraged.argsort()[-5:][::-1]
print("The most positive features are: " + str(featureArray[positiveFeatures]) + \
      " with weights of: " + str(weightVectorAveraged[positiveFeatures]))
negativeFeatures = weightVectorAveraged.argsort()[0:5][::-1]
print("The most negative features are: " + str(featureArray[negativeFeatures]) + \
      " with weights of: " + str(weightVectorAveraged[negativeFeatures]))
