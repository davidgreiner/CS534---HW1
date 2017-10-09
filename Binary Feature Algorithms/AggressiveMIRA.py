import numpy as np
import time
import matplotlib.pyplot as plt
from binarizeData import BinarizeData
from Dev_Evaluator import DevEvaluator

## Basic Perceptron algorithm for binary classification
## of individuals earning less than or more than 50K/year.

trainDataArray, devDataArray, featureArray = BinarizeData(shuffle=0)

p = 0.9

weightVector = np.zeros((len(featureArray)))
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

            devError = DevEvaluator(weightVector,
                                    featureArray, devDataArray)

            epochFraction = (i / numberTrainingData) + epochCount

            devErrorPlot.append(devError)
            epochFractionPlot.append(epochFraction)

            if devError < bestErrorRate:
                bestErrorRate = devError
                epochIteration = epochFraction

            print("The error rate for epoch " + str(epochFraction) + \
                  " is " + str(devError) + "%")

        xi = trainDataArray[i, :-1]

        if trainDataArray[i, -1] == 1:
            y = 1

        else:
            y = -1

        decision = y*(np.dot(weightVector, xi))

        if decision <= p:
            
            marginCorrection = ( (y - np.dot(weightVector, xi) ) / \
              np.dot(xi, xi) )

            weightVector = weightVector + \
              marginCorrection * xi

##            check = y * (np.dot(weightVector, xi))
##            print(check)
            
        currentTrainingCount += 1

    epochCount += 1

positiveFeatures = weightVector.argsort()[-5:][::-1]
print("The most positive features are: " + str(featureArray[positiveFeatures]) + \
      " with weights of: " + str(weightVector[positiveFeatures]))
negativeFeatures = weightVector.argsort()[0:5][::-1]
print("The most negative features are: " + str(featureArray[negativeFeatures]) + \
      " with weights of: " + str(weightVector[negativeFeatures]))
print("The program ran for %s seconds" % (time.time() - startTime))
print("The best error rate was " + str(bestErrorRate) + " at epoch " + \
      str(epochIteration))

plt.plot(epochFractionPlot, devErrorPlot, 'ro')
plt.axis([0, totalEpoch, 0, 100])
plt.xlabel('Epoch Number')
plt.ylabel('Error Rate, %')
plt.show()
