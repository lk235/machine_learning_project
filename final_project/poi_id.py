#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from collections import Counter
import matplotlib.pyplot

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','restricted_stock_deferred','total_stock_value','expenses',
                 'long_term_incentive','restricted_stock','from_poi_to_this_person','to_messages',
                 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print len(data_dict)
print 'features_count_count: ',len(data_dict['METTS MARK'])
poi_count = 0
non_poi_count = 0
nan_list = []
for key,value in data_dict.iteritems():

    for k,v in value.iteritems():
        if v == 'NaN':
            nan_list.append(k)
    if value['poi'] == True:
        print key
        poi_count = poi_count + 1
    else:
        non_poi_count = non_poi_count + 1

print 'poi count: ',poi_count
print 'non_poi_count: ',non_poi_count
print 'feature_nan_list: ',Counter(nan_list)

features = ["total_payments", "total_stock_value"]
data_dict.pop( 'TOTAL', 0 )
data = featureFormat(data_dict, features)
for point in data:
    total_payments= point[0]
    total_stock_value = point[1]
    # print point, '::',total_payments,':',total_stock_value
    matplotlib.pyplot.scatter( total_payments, total_stock_value )

matplotlib.pyplot.xlabel("total_payments")
matplotlib.pyplot.ylabel("total_stock_value")
matplotlib.pyplot.show()

for key,value in data_dict.iteritems():
     if value['total_payments'] > 100000000 and value['total_payments'] != 'NaN':
         print key
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)