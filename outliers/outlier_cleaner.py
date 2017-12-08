#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy
    predictions.np = numpy.array(predictions)
    net_worths.np = numpy.array(net_worths)
    diff = net_worths.np - predictions.np
    # for i in predictions:



    return cleaned_data

import numpy
list1 = [1,2,3,4,5]
list2 = [2,3,4,5,7]
np1 = numpy.array(list1)
np2 = numpy.array(list2)
for i in np1:
    print i
# print np2 - np1
print 'OK'
