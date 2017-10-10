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

    featureArray = np.hstack((age, work, education,
                maritalstatus, occupation, race,
                gender, workhours, country, ['Age 0-17'],
                ['Age 18-30'], ['Age 31-50'],
                ['Age 51-70'], ['Age 71+'],
                ['WorkHours 0-10'], ['WorkHours 11-20'],
                ['WorkHours 21-30'], ['WorkHours 31-40'],
                ['WorkHours 41-50'], ['WorkHours 50+'], ['Bias']))


    binarizedData = []
    binarizedDevData = []
    binarizedTestData = []

    #print(newdata)

    for i in range(0, len(data)):
        row = np.isin(featureArray[:-12], data[i, :-1])
        ageBinned = np.zeros(5)
        workhoursBinned = np.zeros(6)

        age_int = int(data[i, 0])
        workhours_int = int(data[i, 7])
        
        if age_int <= 17:
            ageBinned[0] = 1;
        elif age_int <= 30:
            ageBinned[1] = 1;
        elif age_int <= 50:
            ageBinned[2] = 1;
        elif age_int <= 70:
            ageBinned[3] = 1;
        else:
            ageBinned[4] = 1;

        if workhours_int <= 10:
            workhoursBinned[0] = 1;
        elif workhours_int <= 20:
            workhoursBinned[1] = 1;
        elif workhours_int <= 30:
            workhoursBinned[2] = 1;
        elif workhours_int <= 40:
            workhoursBinned[3] = 1;
        elif workhours_int <= 50:
            workhoursBinned[4] = 1;
        else:
            workhoursBinned[5] = 1;
        
        row2 = np.append(row.astype(int), ageBinned)
        row3 = np.append(row2, workhoursBinned)
        row4 = np.append(row3, [1])

        binarizedData.append(row4.astype(int))

        #print(binarizedData[i])


    toInt = lambda i: int(i == '>50K')
    toIntFunc = np.vectorize(toInt)
    salary = toIntFunc(data[:,-1:])

    unStandData = np.concatenate([binarizedData,salary], axis=1)

    meanData = np.mean(unStandData[:, :-1], axis=0)
    stdDeviationData= np.std(unStandData[:, :-1], axis=0)

    normData = np.nan_to_num((unStandData[:, :-1] - meanData) / stdDeviationData)

    finalData = unStandData = np.concatenate([normData,salary], axis=1)


    #print(len(finalData[0]))

    for i in range(0, len(devData)):
        devRow = np.isin(featureArray[:-12], devData[i, :-1])

        ageBinned = np.zeros(5)
        workhoursBinned = np.zeros(6)

        age_int = int(devData[i, 0])
        workhours_int = int(devData[i, 7])
        
        if age_int <= 17:
            ageBinned[0] = 1;
        elif age_int <= 30:
            ageBinned[1] = 1;
        elif age_int <= 50:
            ageBinned[2] = 1;
        elif age_int <= 70:
            ageBinned[3] = 1;
        else:
            ageBinned[4] = 1;

        if workhours_int <= 10:
            workhoursBinned[0] = 1;
        elif workhours_int <= 20:
            workhoursBinned[1] = 1;
        elif workhours_int <= 30:
            workhoursBinned[2] = 1;
        elif workhours_int <= 40:
            workhoursBinned[3] = 1;
        elif workhours_int <= 50:
            workhoursBinned[4] = 1;
        else:
            workhoursBinned[5] = 1;
        
        devRow2 = np.append(devRow.astype(int), ageBinned)
        devRow3 = np.append(devRow2, workhoursBinned)
        devRow4 = np.append(devRow3, [1])
              
        binarizedDevData.append(devRow4.astype(int))
        #print(binarizedData[i])

    toInt = lambda i: int(i == '>50K')
    toIntFunc = np.vectorize(toInt)
    salaryDev = toIntFunc(devData[:,-1:])

    unStandDevData = np.concatenate([binarizedDevData,salaryDev], axis=1)

    meanData = np.mean(unStandDevData[:, :-1], axis=0)
    stdDeviationData= np.std(unStandDevData[:, :-1], axis=0)

    normDevData = np.nan_to_num((unStandDevData[:, :-1] - meanData) / stdDeviationData)

    finalDevData = unStandDevData = np.concatenate([normDevData,salaryDev], axis=1)

    #print(binarizedData[0])
    #print(len(finalData[0]))
    ##print(featureArray)
    ##print(finalData[0])

    for i in range(0, len(testData)):
        testRow = np.isin(featureArray[:-12], testData[i, :-1])

        ageBinned = np.zeros(5)
        workhoursBinned = np.zeros(6)

        age_int = int(testData[i, 0])
        workhours_int = int(testData[i, 7])
        
        if age_int <= 17:
            ageBinned[0] = 1;
        elif age_int <= 30:
            ageBinned[1] = 1;
        elif age_int <= 50:
            ageBinned[2] = 1;
        elif age_int <= 70:
            ageBinned[3] = 1;
        else:
            ageBinned[4] = 1;

        if workhours_int <= 10:
            workhoursBinned[0] = 1;
        elif workhours_int <= 20:
            workhoursBinned[1] = 1;
        elif workhours_int <= 30:
            workhoursBinned[2] = 1;
        elif workhours_int <= 40:
            workhoursBinned[3] = 1;
        elif workhours_int <= 50:
            workhoursBinned[4] = 1;
        else:
            workhoursBinned[5] = 1;
        
        testRow2 = np.append(testRow.astype(int), ageBinned)
        testRow3 = np.append(testRow2, workhoursBinned)
        testRow4 = np.append(testRow3, [1])
              
        binarizedTestData.append(testRow4.astype(int))
        #print(binarizedData[i])

    unStandTestData = np.concatenate([binarizedDevData,testData[:,-1:]], axis=1)

    meanData = np.mean(unStandTestData[:, :-1], axis=0)
    stdDeviationData= np.std(unStandTestData[:, :-1], axis=0)

    normTestData = np.nan_to_num((unStandTestData[:, :-1] - meanData) / stdDeviationData)

    finalTestData = unStandTestData = np.concatenate([normTestData,testData[:,-1:]], axis=1)

    
    
    return finalData, finalDevData, finalTestData, featureArray
