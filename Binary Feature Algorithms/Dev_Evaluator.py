import numpy as np

## Test current Perceptron on developer data and report
## the error rate.

def DevEvaluator(weightVector, featureArray, devDataArray):

    numberDevData = len(devDataArray)
    devWrong = 0

    for i in range(0, numberDevData):

        if devDataArray[i, -1] == '>50K':
            y = 1

        else:
            y = -1

        idx = np.isin(featureArray, devDataArray[i, 0:-1])

        if y*(weightVector[idx].sum() + weightVector[-1]) <= 0:
            
            devWrong += 1

    devError = (devWrong / len(devDataArray)) * 100
    return devError
