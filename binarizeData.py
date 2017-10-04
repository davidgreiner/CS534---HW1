import numpy as np

data = np.genfromtxt("income-data/income.train.txt",
           dtype=None,
           delimiter=", ")

age = set(data['f0'])
work = set(data['f1']))
education = set(data['f2']))
relationship = set(data['f3']))
occupation = set(data['f4']))
race = set(data['f5']))
gender = set(data['f6']))
workhours = set(data['f7']))
country = set(data['f8']))
salary = set(data['f9']))
