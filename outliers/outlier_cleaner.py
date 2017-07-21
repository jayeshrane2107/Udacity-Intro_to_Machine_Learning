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
    for i in range(len(ages)):
        a = ages[i]
        n = net_worths[i]
        e = abs(predictions[i] - net_worths[i])
        
        x = [a[0],n[0],e[0]]
        
        cleaned_data.append(x)
        cleaned_data= sorted(cleaned_data, key = lambda k : k[2])

    
    return cleaned_data[:81]

