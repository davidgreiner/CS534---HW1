import numpy as np
import time
from binarizeData import BinarizeData
from Dev_Evaluator import DevEvaluator

featureArray, trainDataArray = BinarizeData("train")
devDataArray = BinarizeData("dev")

weightVector = np.zeros(len(featureArray))
devDataArray = BinarizeData("dev")

epochCount = 1
totalEpoch = 5

currentCount = 0;
count = 0
cachedweight = np.zeros(len(featureArray))
currentTrainingCount = 1
numberTrainingData = len(trainDataArray)

startTime = time.time()

while epochCount <= totalEpoch:
    
    for i in range(len(trainDataArray)):

        if currentTrainingCount % 1000 == 0:

            devError = DevEvaluator(cachedweight / (currentTrainingCount), featureArray, devDataArray)

            epochFraction = (i / numberTrainingData) * epochCount

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

endTime = time.time() - startTime

print(endTime)
