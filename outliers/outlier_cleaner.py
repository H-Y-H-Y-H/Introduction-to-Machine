#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    # cleaned_data = []
    #
    # ### your code goes here
    #
    # for i in range(90):
    #     error = pow(predictions[i][0] - net_worths[i][0],2)
    #     cleaned_data.append((ages[i][0],net_worths[i][0],error))
    #
    # cleaned_data.sort(key = lambda x:x[2])
    # cleaned_data = cleaned_data[:81]

    data = []
    size = len(predictions)
    for i in range(size):
        error = (predictions[i][0] - net_worths[i][0]) * (predictions[i][0] - net_worths[i][0])
        data.append((ages[i][0],
                     net_worths[i][0],
                     error))

    data.sort(key=lambda tup: tup[2])
    ### your code goes here
    #    print data
    cleaned_data = data[:81]
    return cleaned_data
