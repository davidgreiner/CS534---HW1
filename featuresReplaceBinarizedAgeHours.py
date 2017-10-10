import numpy as np

## Load training data and convert into two-dimensional array.
## Find unique features and construct feature array to be used
## in Perceptron.

def BinarizeData(sort=0, shuffle=0):

    rawTrainData = np.genfromtxt("income-data/income.train.txt",
           dtype=[('f0', '<i4'), ('f1', 'U17'),
                  ('f2', 'U13'), ('f3', 'U22'),
                  ('f4', 'U18'), ('f5', 'U19'),
                  ('f6', 'U7'), ('f7', '<i4'),
                  ('f8', 'U27'), ('f9', 'U6')],
           delimiter=", ")

    data = np.array(rawTrainData.tolist())

    rawDevData = np.genfromtxt("income-data/income.dev.txt",
           dtype=[('f0', '<i4'), ('f1', 'U17'),
                  ('f2', 'U13'), ('f3', 'U22'),
                  ('f4', 'U18'), ('f5', 'U19'),
                  ('f6', 'U7'), ('f7', '<i4'),
                  ('f8', 'U27'), ('f9', 'U6')],
           delimiter=", ")

    devData = np.array(rawDevData.tolist())

    rawTestData = np.genfromtxt("income-data/income.test.txt",
       dtype=[('f0', '<i4'), ('f1', 'U17'),
                  ('f2', 'U13'), ('f3', 'U22'),
                  ('f4', 'U18'), ('f5', 'U19'),
                  ('f6', 'U7'), ('f7', '<i4'),
                  ('f8', 'U27'), ('f9', 'U6')],
       delimiter=",", autostrip=True)

    testData = np.array(rawTestData.tolist())

    if sort == 1:
        rawTrainData = np.sort(rawTrainData, order='f9', axis=0)
        rawTrainData = np.flip(rawTrainData, axis=0)

    data = np.array(rawTrainData.tolist())

    if shuffle == 1:
        np.random.shuffle(data)

    #for i in range(0, len(data)):
    #    data[i, 0] = 'Age ' + data[i, 0]
    #    data[i, 7] = data[i, 7] + ' Hours'

    age = np.unique(data[:,0])
    work = np.unique(data[:,1])
    education = np.unique(data[:,2])
    maritalstatus = np.unique(data[:,3])
    occupation = np.unique(data[:,4])
    race = np.unique(data[:,5])
    gender = np.unique(data[:,6])
    workhours = np.unique(data[:,7])
    country = np.unique(data[:,8])
    salary = np.unique(data[:,9])

    featureArray = np.hstack((work, education,
                maritalstatus, occupation, race,
                gender, country, ['Age'], ['WorkHours'], ['Bias']))


    binarizedData = []
    binarizedDevData = []
    binarizedTestData = []

    permutation = [7,0,1,2,3,4,5,8,6,9]

    isort = np.argsort(permutation)

    newdata = data[:, isort]
    newDevData = devData[:, isort]
    newTestData = testData[:, isort]

    for i in range(0, len(data)):
        row = np.isin(featureArray[:-3], newdata[i, :-3])
        row2 = np.append(row.astype(int), [data[i, 0], data[i, 7], 1])

        binarizedData.append(row2.astype(int))


    toInt = lambda i: int(i == '>50K')
    toIntFunc = np.vectorize(toInt)
    salary = toIntFunc(data[:,-1:])

    finalData = np.concatenate([binarizedData,salary], axis=1)

    for i in range(0, len(devData)):
        devRow = np.isin(featureArray[:-3], newDevData[i, :-3])
        devRow2 = np.append(devRow.astype(int), [devData[i, 0], devData[i, 7], 1])
              
        binarizedDevData.append(devRow2.astype(int))


    toInt = lambda i: int(i == '>50K')
    toIntFunc = np.vectorize(toInt)
    salaryDev = toIntFunc(devData[:,-1:])

    finalDevData = np.concatenate([binarizedDevData,salaryDev], axis=1)

    for i in range(0, len(testData)):
        testRow = np.isin(featureArray[:-3], newTestData[i, :-3])
        testRow2 = np.append(testRow.astype(int), [testData[i, 0], testData[i, 7], 1])
              
        binarizedTestData.append(testRow2.astype(int))

    finalTestData = np.concatenate([binarizedTestData,testData[:,-1:]], axis=1)
    
    
    return finalData, finalDevData, finalTestData, featureArray
