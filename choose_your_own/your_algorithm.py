#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from time import time
from sklearn.metrics import accuracy_score

clf_KNN = KNeighborsClassifier()
clf_RFC = RandomForestClassifier()
clf_ABC = AdaBoostClassifier()

time_KNN = time()
clf_KNN.fit(features_train, labels_train)
print "\nKNN training time:",round(time() - time_KNN, 3), "s"
pred_KNN = clf_KNN.predict(features_test)
acc_KNN = accuracy_score(labels_test, pred_KNN)
print "KNN Accuracy:",acc_KNN

time_RFC = time()
clf_RFC.fit(features_train, labels_train)
print "\nRFC training time:",round(time() - time_RFC, 3), "s"
pred_RFC = clf_RFC.predict(features_test)
acc_RFC = accuracy_score(labels_test, pred_RFC)
print "RFC Accuracy:",acc_RFC

time_ABC = time()
clf_ABC.fit(features_train, labels_train)
print "\nABC training time:",round(time() - time_ABC, 3), "s"
pred_ABC = clf_ABC.predict(features_test)
acc_ABC = accuracy_score(labels_test, pred_ABC)
print "ABC Accuracy:",acc_ABC

pred_CYOA = []
for i in range(len(pred_KNN)):
    pred_CYOA.append((pred_KNN[i]+pred_RFC[i]+pred_ABC[i])//2)
acc_CYOA = accuracy_score(labels_test,pred_CYOA)
print "\nCYOA Accuracy:",acc_CYOA

try:
    prettyPicture(clf_KNN, features_test, labels_test)
except NameError:
    pass
