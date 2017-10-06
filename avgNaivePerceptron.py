import numpy as np
import time
from binarizeData import BinarizeData
from Dev_Evaluator import DevEvaluator

featureArray, trainDataArray = BinarizeData("train")
devDataArray = BinarizeData("dev")

weightVector = np.zeros(len(featureArray))
cachedweight = np.zeros(len(featureArray))
epochCount = 1
totalEpoch = 5
numberTrainingData = len(trainDataArray)
currentTrainingCount = 1
bestErrorRate = 100.0
epochIteration = 0

startTime = time.time()

while epochCount <= totalEpoch:
    
    for i in range(len(trainDataArray)):

        if currentTrainingCount % 1000 == 0:

            devError = DevEvaluator(cachedweight / (currentTrainingCount), featureArray, devDataArray)

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

        if y * weightVector[idx].sum() + weightVector[-1] <= 0:

            weightVector[idx] = weightVector[idx] + \
                    y * np.ones((len(trainDataArray[i, 0:-1])))

            weightVector[-1] = weightVector[-1] + y

            cachedweight[idx] = cachedweight[idx] + y * currentTrainingCount * np.ones((len(trainDataArray[i, 0:-1])))

            cachedweight[-1] = cachedweight[-1] + y * currentTrainingCount

        currentTrainingCount += 1

    epochCount += 1

print("The program ran for %s seconds" % (time.time() - startTime))
print("The best error rate was " + str(bestErrorRate) + " at epoch " + \
      str(epochIteration))
