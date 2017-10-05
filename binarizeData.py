import numpy as np

data = np.genfromtxt("income-data/income.train.txt",
           dtype=None,
           delimiter=", ")

age = set(data['f0'])
work = set(data['f1'])
education = set(data['f2'])
relationship = set(data['f3'])
occupation = set(data['f4'])
race = set(data['f5'])
gender = set(data['f6'])
workhours = set(data['f7'])
country = set(data['f8'])
salary = set(data['f9'])

columns = len(age) + len(work) + len(education) + len(relationship) + len(occupation) + len(race) + len(gender) + len(workhours) + len(country) + len(salary)

newdata = [[0 for x in range(columns)] for y in range(data.size)]

for index,row in enumerate(data):
    count = 0
    for i in age:
        if i == row[0]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in work:
        if i == row[1]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in education:
        if i == row[2]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in relationship:
        if i == row[3]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in occupation:
        if i == row[4]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in race:
        if i == row[5]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in gender:
        if i == row[6]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in workhours:
        if i == row[7]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in country:
        if i == row[8]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
    for i in salary:
        if i == row[9ter]:
            newdata[index][count] = 1
        else:
            newdata[index][count] = 0
        count += 1
