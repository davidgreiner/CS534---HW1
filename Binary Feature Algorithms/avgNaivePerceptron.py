import numpy as np
import time
from binarizeData import BinarizeData
from Dev_Evaluator import DevEvaluator

trainDataArray, devDataArray, featureArray = BinarizeData()

weightVector = np.zeros(len(featureArray))
cachedweight = np.zeros(len(featureArray))
epochCount = 0
totalEpoch = 5
numberTrainingData = len(trainDataArray)
currentTrainingCount = 1
bestErrorRate = 100.0
epochIteration = 0

startTime = time.time()

while epochCount < totalEpoch:
    
    for i in range(len(trainDataArray)):

        if currentTrainingCount % 1000 == 0:

            devError = DevEvaluator(cachedweight / (currentTrainingCount),
                                    featureArray, devDataArray)

            epochFraction = (i / numberTrainingData) + epochCount

            if devError < bestErrorRate:
                bestErrorRate = devError
                epochIteration = epochFraction

            print("The error rate for epoch " + str(epochFraction) + \
                   " is " + str(devError) + "%")


        if trainDataArray[i, -1] == 1:
            y = 1

        else:
            y = -1

        xi = trainDataArray[i, :-1]

        if y * np.dot(weightVector, xi) <= 0:

            weightVector = weightVector + \
                    y * xi

        cachedweight = cachedweight + weightVector

        currentTrainingCount += 1

    epochCount += 1

finalWeight = cachedweight / currentTrainingCount

print("The program ran for %s seconds" % (time.time() - startTime))
print("The best error rate was " + str(bestErrorRate) + " at epoch " + \
      str(epochIteration))

positiveFeatures = cachedweight.argsort()[-5:][::-1]
print("The most positive features are: " + str(featureArray[positiveFeatures]) + \
      " with weights of: " + str(finalWeight[positiveFeatures]))
negativeFeatures = cachedweight.argsort()[0:5][::-1]
print("The most negative features are: " + str(featureArray[negativeFeatures]) + \
      " with weights of: " + str(finalWeight[negativeFeatures]))
