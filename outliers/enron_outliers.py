#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

# use matplotlib.pyplot to visualize our data
for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

### your code below
# size = len(data)
# bonus_train = []
# salary_train = []
# for i in range(size):
#     bonus_train.append(data[i][0])
#     salary_train.append(data[i][1])
# print salary_train,'\n',bonus_train
#
# from sklearn import linear_model
# reg = linear_model.LinearRegression()
# reg.fit()





