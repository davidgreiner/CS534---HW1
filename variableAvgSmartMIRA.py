import numpy as np
import time
import matplotlib.pyplot as plt
from normalizedFeatures import BinarizeData
from Dev_Evaluator import DevEvaluator
from Predictor import Predictor

## Averaged Smart Perceptron algorithm for binary classification
## of individuals earning less than or more than 50K/year.

trainDataArray, devDataArray, testDataArray, featureArray = BinarizeData(shuffle=1)

weightVector = np.zeros((len(trainDataArray[0, :-1])))
weightVectorAveraged = np.zeros((len(trainDataArray[0, :-1])))
epochCount = 0
totalEpoch = 5
numberTrainingData = len(trainDataArray)
currentTrainingCount = 1
bestErrorRate = 100.0
epochIteration = 0

devErrorPlot = []
epochFractionPlot = []

startTime = time.time()
learningRate = 0.5
numberofErrors = 0

while epochCount < totalEpoch:

    for i in range(0, numberTrainingData):

        if currentTrainingCount % 1000 == 0:

            devError = DevEvaluator(weightVector - (weightVectorAveraged / currentTrainingCount), \
                                    devDataArray)

            epochFraction = (i / numberTrainingData) + epochCount

            devErrorPlot.append(devError)
            epochFractionPlot.append(epochFraction)

            if devError < bestErrorRate:
                bestErrorRate = devError
                epochIteration = epochFraction
                bestWeightVector = weightVector - (weightVectorAveraged / currentTrainingCount)
                
                

            print("The error rate for epoch " + str(epochFraction) + \
                  " is " + str(devError) + "%")

        if trainDataArray[i, -1] == 1:
        
            y = 1

        else:

            y = -1

        xi = trainDataArray[i, :-1]

        decision = y * np.dot(xi, weightVector)

        if decision <= 0:

            marginCorrection = ( (y - np.dot(xi, weightVector))) / \
              (np.dot(xi, xi))
            
            weightVector = weightVector + \
            marginCorrection*xi * learningRate

            weightVectorAveraged = weightVectorAveraged + \
            currentTrainingCount * marginCorrection * xi * learningRate

            numberofErrors += 1

            learningRate = learningRate * (1 / numberofErrors)

##            check = y * np.dot(xi, weightVector)
##            print(check)

        currentTrainingCount += 1

    epochCount += 1

print("The program ran for %s seconds" % (time.time() - startTime))
print("The best error rate was " + str(bestErrorRate) + " at epoch " + \
      str(epochIteration))

finalWeightVector = weightVector - (weightVectorAveraged / currentTrainingCount)

positiveFeatures = finalWeightVector.argsort()[-5:][::-1]
print("The most positive features are: " + str(featureArray[positiveFeatures]) + \
      " with weights of: " + str(finalWeightVector[positiveFeatures]))
negativeFeatures = finalWeightVector.argsort()[0:5][::-1]
print("The most negative features are: " + str(featureArray[negativeFeatures]) + \
      " with weights of: " + str(finalWeightVector[negativeFeatures]))

plt.plot(epochFractionPlot, devErrorPlot, 'ro')
plt.axis([0, totalEpoch, 0, 100])
plt.xlabel('Epoch Number')
plt.ylabel('Error Rate, %')
plt.show()
