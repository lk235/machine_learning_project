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
    print 'START'
    predictions_np = numpy.array(predictions)
    net_worths_np = numpy.array(net_worths)
    diff = net_worths_np - predictions_np

    abs_diff = numpy.absolute(diff)
    print type(abs_diff)
    sored_diff = numpy.sort(abs_diff,axis=0)
    print sored_diff
    print '9',sored_diff[81]

    for i in range(81):
        cleaned_data_item = []
        if abs(net_worths[i] - predictions[i]) <= sored_diff[81] :

            cleaned_data_item.append(ages[i])
            cleaned_data_item.append(net_worths[i])
            cleaned_data_item.append(net_worths[i] - predictions[i])
            cleaned_data.append(cleaned_data_item)


    print cleaned_data
    return cleaned_data

# import numpy
# list1 = [5,8,6,55,41]
# list2 = [2,3,4,5,7]
# np1 = numpy.array(list1)
# np2 = numpy.array(list2)
# for i in np1:
#     print i
# print np1 - np2
# print 'OK'
# print numpy.sort(np1 - np2)

