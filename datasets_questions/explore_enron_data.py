#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
print len(enron_data)
for i in enron_data.keys():
    if 'LAY' in i:
        print i
print len(enron_data['METTS MARK'])
count = 0
for k in enron_data:
    # if enron_data[k]["poi"] == 1:
    #     count = count + 1
    # print enron_data[k]
    if enron_data[k]["poi"] == 1:
        if enron_data[k]['total_payments'] == 'NaN':
            count = count + 1


print 'total_payments',float(count)/len(enron_data)
print enron_data['METTS MARK'].keys()

print enron_data['Prentice James'.upper()]['total_stock_value']
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print enron_data['SKILLING JEFFREY K'][ 'total_payments']
print enron_data['FASTOW ANDREW S'][ 'total_payments']
print enron_data['LAY KENNETH L'][ 'total_payments']







