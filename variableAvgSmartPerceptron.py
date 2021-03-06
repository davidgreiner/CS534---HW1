import numpy as np
import time
import matplotlib.pyplot as plt
from featuresBinnedNumerical import BinarizeData
from Dev_Evaluator import DevEvaluator

## Averaged, smart perceptron with a variable learning rate
## for binary classification of individuals earning less than
## or more than 50K/year.

trainDataArray, devDataArray, testDataArray, featureArray = BinarizeData(sort =0, shuffle=0)
weightVector = np.zeros((len(trainDataArray[0, :-1])))
weightVectorAveraged = np.zeros((len(trainDataArray[0, :-1])))
epochCount = 0
totalEpoch = 1
numberTrainingData = len(trainDataArray)
currentTrainingCount = 1
bestErrorRate = 100.0
epochIteration = 0

devErrorPlot = []
epochFractionPlot = []
learningRate = 1
startTime = time.time()
numberofErrors = 0

while epochCount < totalEpoch:

    for i in range(0, numberTrainingData):

        if currentTrainingCount % 20 == 0:

            devError = DevEvaluator(weightVector - (weightVectorAveraged / currentTrainingCount), \
                                    devDataArray)

            epochFraction = (i / numberTrainingData) + epochCount
            
            if devError < bestErrorRate:
                bestErrorRate = devError
                epochIteration = epochFraction
                bestWeightVector = weightVector - (weightVectorAveraged / currentTrainingCount)


            devErrorPlot.append(devError)
            epochFractionPlot.append(epochFraction)

            if devError < bestErrorRate:
                bestErrorRate = devError
                epochIteration = epochFraction

##            print("The error rate for epoch " + str(epochFraction) + \
##                  " is " + str(devError) + "%")

        if trainDataArray[i, -1] == 1:
            y = 1

        else:
            y = -1

        xi = trainDataArray[i, 0:-1]

        if y*np.dot(xi, weightVector) <= 0:
            
            weightVector = weightVector + \
            y*xi*learningRate


            weightVectorAveraged = weightVectorAveraged + y * currentTrainingCount * xi * learningRate

            numberofErrors += 1
            learningRate = learningRate * (9925 / 10000) + 0.00101

        currentTrainingCount += 1

    epochCount += 1

finalWeightVector = weightVector - (weightVectorAveraged / currentTrainingCount)

positiveFeatures = finalWeightVector.argsort()[-5:][::-1]
print("The most positive features are: " + str(featureArray[positiveFeatures]) + \
      " with weights of: " + str(finalWeightVector[positiveFeatures]))
negativeFeatures = finalWeightVector.argsort()[0:5][::-1]
print("The most negative features are: " + str(featureArray[negativeFeatures]) + \
      " with weights of: " + str(finalWeightVector[negativeFeatures]))
positiveFeatures = weightVector.argsort()[-5:][::-1]

print("The best error rate was " + str(bestErrorRate) + " at epoch " + \
      str(epochIteration))

plt.plot(epochFractionPlot, devErrorPlot, 'ro')
plt.axis([0, totalEpoch, 0, 100])
plt.xlabel('Epoch Number')
plt.ylabel('Error Rate, %')
plt.show()
