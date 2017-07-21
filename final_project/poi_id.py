#!/usr/bin/python

import sys
import pickle
import matplotlib

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
#----------------------------------------------------------------------------------------------------------

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','total_payments','total_stock_value', 'from_this_person_to_poi', 
'to_messages', 'from_poi_to_this_person', 'from_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print 'Counting the NaN values for selecting features from finance dataset -'

def cnt_all(str):
    cnt_all_arg = 0
    for i in data_dict:
        if data_dict[i][str]=='NaN':
            cnt_all_arg += 1
    return cnt_all_arg
          
def cnt_poi(str):
    cnt_poi_arg = 0
    for i in data_dict:
        if data_dict[i]['poi']==1.0:
            if data_dict[i][str]=='NaN':
                cnt_poi_arg += 1 
    return cnt_poi_arg

print 'all_salary_nan ',cnt_all('salary')
print 'poi_salary_nan ',cnt_poi('salary')
print 'all_bonus_nan ',cnt_all('bonus')
print 'poi_bonus_nan ',cnt_poi('bonus')
print 'all_total_payments_nan ',cnt_all('total_payments')
print 'poi_total_payments_nan ',cnt_poi('total_payments')
print 'all_total_stock_value_nan ',cnt_all('total_stock_value')
print 'poi_total_stock_value_nan ',cnt_poi('total_stock_value')
#--------------------------------------------------------------------------------------------------------

### Task 2: Remove outliers
  
print '\n','Before outlier data_dict length - ',len(data_dict)
data_dict.pop('TOTAL')
print 'After outlier data_dict length - ',len(data_dict)
data = featureFormat(data_dict, features_list)
print 'Available data points after removing all_zero rows - ', len(data)  # 2rows - all features are zero (27,90)
def out(xxx,yyy):
    for pt in data:
        x = pt[xxx] 
        y = pt[yyy]
        matplotlib.pyplot.scatter(x, y)
    matplotlib.pyplot.xlabel(features_list[xxx])
    matplotlib.pyplot.ylabel(features_list[yyy])
    matplotlib.pyplot.show()
print '\n','Plotting data_dict with combinations of x,y'
out(1,2) # Salary vs Bonus
#---------------------------------------------------------------------------------------------------------

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#print float(data_dict['ALLEN PHILLIP K']['from_this_person_to_poi']) / float(data_dict['ALLEN PHILLIP K']['from_messages'])
my_dataset = data_dict
for i in data_dict:
    my_dataset[i]['to_messages_ratio'] = float(my_dataset[i]['from_poi_to_this_person']) / float(my_dataset[i]['to_messages'])
    my_dataset[i]['from_messages_ratio'] = float(my_dataset[i]['from_this_person_to_poi']) / float(my_dataset[i]['from_messages'])

for i in my_dataset:
    if(my_dataset[i]['to_messages_ratio'] == 'nan'):
        my_dataset[i]['to_messages_ratio'] = '0.0'


### Extract features and labels from dataset for local testing
#features_list = ['poi','salary','bonus','total_payments','total_stock_value', 'from_this_person_to_poi', 'to_messages', 'from_poi_to_this_person', 'from_messages', 'to_messages_ratio', 'from_messages_ratio']
#data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf_gaussian = GaussianNB()  #Accuracy: 0.84133       Precision: 0.33216      Recall: 0.18800 F1: 0.24010     F2: 0.20587

#from sklearn.svm import SVC
#clf_svm = SVC(kernel = "rbf", C = 10000)

from sklearn.tree import DecisionTreeClassifier
clf_dtree = DecisionTreeClassifier()  #Accuracy: 0.79080       Precision: 0.24277      Recall: 0.26850 F1: 0.25499     F2: 0.26293

from sklearn.neighbors import KNeighborsClassifier
clf_neighbor = KNeighborsClassifier()  #Accuracy: 0.85867       Precision: 0.38764      Recall: 0.10350 F1: 0.16338     F2: 0.12128

from sklearn.ensemble import RandomForestClassifier
clf_random = RandomForestClassifier() #Accuracy: 0.85280       Precision: 0.32724      Recall: 0.09850 F1: 0.15142     F2: 0.11451

from sklearn.ensemble import AdaBoostClassifier
clf_adaboost = AdaBoostClassifier()
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


dump_classifier_and_data(clf_adaboost, my_dataset, features_list)