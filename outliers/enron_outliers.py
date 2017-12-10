#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop( 'TOTAL', 0 )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

for d in data_dict:
    sa = data_dict[d]['salary']
    try:
        if int(sa) >= 1000000:
            print d
    except:
        print 'error'

    # if d['salary'] > 25000000:
    #     print d
### your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    # print point, '::',salary,':',bonus
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



