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
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

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
def computeFraction( poi_messages, all_messages ):
    if poi_messages == 'NaN' or all_messages == 'NaN':
        fraction = 0
    else:
        fraction = float(poi_messages) / float(all_messages)

    return fraction

for name in data_dict:

    data_point = data_dict[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi
    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi

features_list_add_new_features = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                 'director_fees','to_messages', 'from_poi_to_this_person', 'from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi']


### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list_add_new_features, sort_keys = True)
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)
from sklearn import linear_model
reg = linear_model.Lasso()
reg.fit(features,labels)
# reg.predict([1,2])
print reg.coef_
print reg.intercept_

from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2

selector = SelectKBest(k=9).fit(features,labels)
print selector.scores_
print selector.pvalues_

new_features = []
# for bool, feature in zip(selector.get_support(), features_list_add_new_features):
for bool, feature in zip(selector.get_support(), features_list):
    if bool:
        new_features.append(feature)
print new_features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
data = featureFormat(data_dict, new_features)
labels, features = targetFeatureSplit(data)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,test_size=0.3,random_state=42)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train,labels_train)
pred_bayes = clf.predict(features_test)
print accuracy_score(labels_test,pred_bayes)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(features_train,labels_train)
pred_tree = clf.predict(features_test)
print accuracy_score(labels_test,pred_tree)

from sklearn.svm import SVC
clf = SVC()
clf.fit(features_train,labels_train)
pred_svm = clf.predict(features_test)
print accuracy_score(labels_test,pred_svm)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(features_train,labels_train)
pred_knn = clf.predict(features_test)
print accuracy_score(labels_test,pred_knn)

from sklearn.model_selection import GridSearchCV
# features_train = features_train[:len(features_train)/10]
# labels_train = labels_train[:len(labels_train)/10]

# parameters = {'kernel':('linear', 'poly', 'rbf', 'sigmoid'), 'C':[1, 1000]}
parameters = {'C': [1, 10], 'kernel': ['linear']},
svc = SVC()
clf = GridSearchCV(svc, parameters)
print clf.fit(features_train,labels_train)
# print sorted(clf.cv_results_.keys())

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