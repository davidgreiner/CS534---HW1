import numpy as np

data = np.genfromtxt("income-data/income.train.txt",
           dtype=[('f0', '<i4'), ('f1', 'U17'),
                  ('f2', 'U13'), ('f3', 'U22'),
                  ('f4', 'U18'), ('f5', 'U19'),
                  ('f6', 'U7'), ('f7', '<i4'),
                  ('f8', 'U27'), ('f9', 'U6')],
           delimiter=", ")

age = np.unique(data['f0'])
work = np.unique(data['f1'])
education = np.unique(data['f2'])
maritalstatus = np.unique(data['f3'])
occupation = np.unique(data['f4'])
race = np.unique(data['f5'])
gender = np.unique(data['f6'])
workhours = np.unique(data['f7'])
country = np.unique(data['f8'])
salary = np.unique(data['f9'])

featureVector = np.hstack((age, work, education,
                maritalstatus, occupation, race,
                gender, workhours, country))

## The genfromtxt function only produces a 1-dimensional
## array of data.  np.column_stack breaks 'data' into
## 2-dimensional array.

arrayData = np.column_stack((data['f0'],data['f1'],
                             data['f2'],data['f3'],
                             data['f4'],data['f5'],
                             data['f6'],data['f7'],
                             data['f8'],data['f9']))
