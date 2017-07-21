#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.3, random_state = 42)


### your code goes here 
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)

pred = clf.predict(features_test)

from sklearn.metrics import accuracy_score
print 'accuracy = ', accuracy_score(pred, labels_test)

print sum(pred)

cnt =0
for x in pred:
    cnt = cnt+1
print cnt

pred1 = []
for i in range(0,len(pred)):
    pred1.append(0)
print pred1

from sklearn.metrics import accuracy_score
print 'accuracy1 = ', accuracy_score(pred1, labels_test)

pred_both = []
for i in range(len(labels_test)):
    pred_both.append([labels_test[i], pred[i]])


print len(labels_test)
print len(pred)
print len(pred_both)

print pred_both

cnt1 = 0

import numpy as np
import pandas as pd
df = pd.DataFrame(pred_both, columns=['actual','prediction'])

#for i in pred_both:
#    if pred_both[i]["actual"]==1.0 and pred_both[i]["prediction"]==1.0:
#        cnt1 =cnt1+1


from sklearn.metrics import precision_score
print 'precision = ', precision_score(labels_test, pred)

from sklearn.metrics import recall_score
print 'recall = ', recall_score(labels_test, pred)