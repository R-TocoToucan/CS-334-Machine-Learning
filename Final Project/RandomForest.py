import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression

from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

data = pd.read_csv('df_elasticnet.csv')
label = pd.read_csv('df_label.csv')
data = data.drop(columns=['Turnover - USD'])
#data = data.astype(float)

data['Volume'] = data['Volume'].str.replace(",","").astype(float)

# Chronological train test split
cr_xtrain = data.iloc[int(0.3*len(data)):-1]
cr_xtest = data.iloc[0:int(0.3*len(data))]
cr_ytrain = label.iloc[int(0.3*len(label)):-1]
cr_ytest = label.iloc[0:int(0.3*len(label))]

# Random train test split
rd_xtrain, rd_xtest, rd_ytrain, rd_ytest = train_test_split(data, label, test_size = 0.3)

clf_tree = DecisionTreeClassifier()

clf_tree.fit(cr_xtrain, cr_ytrain)

predictions = clf_tree.predict(cr_xtest)


# Tuning Hyperparameters
param_dict = {
 "criterion": ['gini', 'entropy'],
 "max_depth": range(1,10),
 "min_samples_split": range(2,10),
 "min_samples_leaf": range(1,5)
}

grid = GridSearchCV(clf_tree, param_grid = param_dict, cv=None, verbose=1, n_jobs=-1)
grid.fit(cr_xtrain, cr_ytrain)

best_param = grid.best_params_
best_est = grid.best_estimator_

print(best_param)

#Calculate accuracy of prediction
accu = accuracy_score(cr_ytest, predictions)
print("accuracy: ", accu)

# roc
auc_score = roc_auc_score(cr_ytest, predictions)
print('auc score: ', auc_score)

false_positive_rate, true_positive_rate, thresholds = roc_curve(cr_ytest, predictions)
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.figure()
plt.plot(false_positive_rate, true_positive_rate, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# Create a Gaussian Classifier
clf2=RandomForestClassifier(n_estimators=100)

# Train the model using the training sets
clf2.fit(cr_xtrain, cr_ytrain)

# Test the model
pred=clf2.predict(cr_xtest)


# See how accurate the predictions are
report = classification_report(cr_ytest, pred)
print('Model accuracy', accuracy_score(cr_ytest, pred, normalize=True))
print(report)
