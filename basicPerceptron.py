import numpy as np
import time
import matplotlib.pyplot as plt
from featuresBinarized import BinarizeData
from Dev_Evaluator import DevEvaluator

## Basic Perceptron algorithm for binary classification
## of individuals earning less than or more than 50K/year.

trainDataArray, devDataArray, testDataArray, featureArray = BinarizeData(sort =0, shuffle=0)
weightVector = np.zeros((len(trainDataArray[0, :-1])))
epochCount = 0
totalEpoch = 5
numberTrainingData = len(trainDataArray)
currentTrainingCount = 0
bestErrorRate = 100.0
epochIteration = 0

devErrorPlot = []
epochFractionPlot = []

startTime = time.time()

while epochCount < totalEpoch:

    for i in range(0, numberTrainingData):

        if currentTrainingCount % 1000 == 0:

            devError = DevEvaluator(weightVector, devDataArray)

            epochFraction = (i / numberTrainingData) + epochCount

            devErrorPlot.append(devError)
            epochFractionPlot.append(epochFraction)

            if devError < bestErrorRate:
                bestErrorRate = devError
                epochIteration = epochFraction

            print("The error rate for epoch " + str(epochFraction) + \
                  " is " + str(devError) + "%")

        if trainDataArray[i, -1] == 1:
            y = 1

        else:
            y = -1


        xi = trainDataArray[i, 0:-1]

        if y*np.dot(xi, weightVector) <= 0:
            
            weightVector = weightVector + \
            y*xi

        currentTrainingCount += 1

    epochCount += 1

print("The program ran for %s seconds" % (time.time() - startTime))
positiveFeatures = weightVector.argsort()[-5:][::-1]
print("The most positive features are: " + str(featureArray[positiveFeatures]) + \
      " with weights of: " + str(weightVector[positiveFeatures]))
negativeFeatures = weightVector.argsort()[0:5][::-1]
print("The most negative features are: " + str(featureArray[negativeFeatures]) + \
      " with weights of: " + str(weightVector[negativeFeatures]))
print("The best error rate was " + str(bestErrorRate) + " at epoch " + \
      str(epochIteration))

plt.plot(epochFractionPlot, devErrorPlot, 'ro')
plt.axis([0, totalEpoch, 0, 100])
plt.xlabel('Epoch Number')
plt.ylabel('Error Rate, %')
plt.show()
