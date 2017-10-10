import numpy as np

def Predictor(weightVector, testData):
    with open("income-data/income.test.txt", "r") as f:
        output = open('income-data/income.test.predicted.txt', 'w') 
        i = 0
        for line in f:
            testRow = testData[i, 0:-1]
            
            if np.dot(testRow.astype(int), weightVector) <= 0:
                # <= 50
                output.write(line.replace('\n',' <=50\n'))
            else:
                # >50
                output.write(line.replace('\n',' >50\n'))
            i += 1
        output.close()
