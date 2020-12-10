#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 23:26:15 2020

@author: ziruiwang
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets


MMM = pd.read_csv('MMM_rdForest.csv')
MMM_label = pd.read_csv('MMM_label.csv')
MMM = MMM.drop(columns=['%Chg'])
MMM = MMM.astype(float)

# Chronological train test split
cr_xtrain = MMM.iloc[int(0.3*len(MMM)):-1]
cr_xtest = MMM.iloc[0:int(0.3*len(MMM))]
cr_ytrain = MMM_label.iloc[int(0.3*len(MMM_label)):-1]
cr_ytest = MMM_label.iloc[0:int(0.3*len(MMM_label))]

# Random train test split
rd_xtrain, rd_xtest, rd_ytrain, rd_ytest = train_test_split(MMM, MMM_label, test_size = 0.3)

clf = DecisionTreeClassifier()
clf.fit(cr_xtrain, cr_ytrain)
predict = clf.predict(cr_xtest)

param_dict = {
 "criterion": ['gini', 'entropy'],
 "max_depth": range(1,10),
 "min_samples_split": range(1,10),
 "min_samples_leaf": range(1,5)
}

grid = GridSearchCV(clf, param_grid = param_dict, cv=None, verbose=1, n_jobs=-1)
grid.fit(cr_xtrain, cr_ytrain)

best_param = grid.best_params_
best_est = grid.best_estimator_

print(best_param)
#Calculate accuracy of prediction
accu = accuracy_score(target_test, predict)
