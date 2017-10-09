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

    featureArray = np.hstack((work,
                maritalstatus, occupation, race, gender, country,
                ['Age'], ['WorkHours'], ['Education'], ['Bias']))

    educationDict = {'Preschool': 0, '1st-4th': 1, '5th-6th': 2,
                     '7th-8th': 3, '9th': 4, '10th': 5,  '11th': 6,
                     '12th': 7, 'HS-grad': 8, 'Some-college': 9, 'Assoc-voc': 10,
                     'Assoc-acdm': 11, 'Bachelors': 12, 'Masters': 13,
                     'Doctorate': 14, 'Prof-school': 15}


    binarizedData = []
    binarizedDevData = []

    permutation = [6,0,8,1,2,3,4,7,5,9]

    isort = np.argsort(permutation)
    ##print(isort)

    newdata = data[:, isort]
    newDevData = devData[:, isort]
    ##print(newdata)

    for i in range(0, len(data)):
        educationVal = newdata[i, -2]
    ##    print(educationVal)
        row = np.isin(featureArray[:-4], newdata[i, :-4])
        row2 = np.append(row.astype(int), [data[i, 0], data[i, 7],
                                           educationDict[educationVal], 1])

        binarizedData.append(row2.astype(int))
        #print(binarizedData[i])


    toInt = lambda i: int(i == '>50K')
    toIntFunc = np.vectorize(toInt)
    salary = toIntFunc(data[:,-1:])

    finalData = np.concatenate([binarizedData,salary], axis=1)

    #print(binarizedData[0])
    #print(len(finalData[0]))

    for i in range(0, len(devData)):
        educationVal = newDevData[i, -2]
        devRow = np.isin(featureArray[:-4], newDevData[i, :-4])
        devRow2 = np.append(devRow.astype(int), [devData[i, 0], devData[i, 7],
                                            educationDict[educationVal], 1])
              
        binarizedDevData.append(devRow2.astype(int))
        #print(binarizedData[i])


    toInt = lambda i: int(i == '>50K')
    toIntFunc = np.vectorize(toInt)
    salaryDev = toIntFunc(devData[:,-1:])

    finalDevData = np.concatenate([binarizedDevData,salaryDev], axis=1)

    #print(binarizedData[0])
    #print(len(finalData[0]))
    
    return finalData, finalDevData, featureArray
