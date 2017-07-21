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

print len(enron_data.keys())
#print enron_data.keys()

#print len(enron_data["PRENTICE JAMES"].keys())
#print enron_data["PRENTICE JAMES"].keys()


cnt = 0
for i in enron_data:
    if enron_data[i]["total_payments"] == "NaN":
       cnt = cnt+1

print cnt 

cnt2 = 0
cnt1 = 0
for ii in enron_data:
    if enron_data[ii]["poi"] == True:
        cnt1 = cnt1+1
        if enron_data[ii]["total_payments"] == "NaN":
            cnt2 = cnt2+1

print cnt1
print cnt2
#print enron_data["LAY KENNETH L"]["total_payments"]
#print enron_data["SKILLING JEFFREY K"]["total_payments"]
#print enron_data["FASTOW ANDREW S"]["total_payments"]

print enron_data["ALLEN PHILLIP K"]