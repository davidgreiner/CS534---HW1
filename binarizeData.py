import numpy as np

## Load training data and convert into two-dimensional array.
## Find unique features and construct feature array to be used
## in Perceptron.

def BinarizeData(dataSet):

    if dataSet == "train" or "dev" or "test":

        rawData = np.genfromtxt("income-data/income." + dataSet + ".txt",
               dtype=[('f0', '<U4'), ('f1', 'U17'),
                      ('f2', 'U13'), ('f3', 'U22'),
                      ('f4', 'U18'), ('f5', 'U19'),
                      ('f6', 'U7'), ('f7', '<i4'),
                      ('f8', 'U27'), ('f9', 'U6')],
               delimiter=", ")

        data = np.array(rawData.tolist())

        for i in range(0, len(data)):
            data[i, 0] = 'Age ' + data[i, 0]
            data[i, 7] = data[i, 7] + ' Hours'

        if dataSet == 'train':

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
                        gender, workhours, country, ['Bias']))

            return featureArray, data

        else:

            return data

    else:

        print("Invalid file name")
        return
