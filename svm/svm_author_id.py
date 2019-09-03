#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# speed up the algorithm, by using the 1% of the data, waste 99%
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]

clf = SVC(C=10000,kernel='rbf',degree=3,gamma='auto',coef0=0.0)
t0 = time()
clf.fit(features_train, labels_train)
print "\ntraining time:",round(time()-t0,3),"s"

t0 = time()
prediction = clf.predict(features_test)
sub = 0
for a in prediction:
    if a == 1:
         sub +=1
print sub
print "predicting time:",round(time()-t0,3),"s"

#accuracy = accuracy_score(pred,labels_test)
#print '\naccuracy = {0}'.format(accuracy)
#########################################################


