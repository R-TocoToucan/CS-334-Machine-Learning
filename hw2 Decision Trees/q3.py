import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, cross_val_score
import warnings

def knn(xTrain, yTrain):
        warnings.filterwarnings("ignore")
        xTrain,xTest,yTrain,yTest = train_test_split(xTrain, yTrain, test_size=0.3)

        knn1 = KNeighborsClassifier(n_neighbors=3)
        knn1.fit(xTrain,yTrain)
        y_pred = knn1.predict(xTest)

        #hyper parameter tuning.Selecting best K
        neighbors = [x for x in range(1,50) if x % 2 != 0]
    # empty list that will hold cv scores
        cv_scores = []
        for k in neighbors:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, xTrain, yTrain, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())
        # graphical view
        # misclassification error
        MSE = [1-x for x in cv_scores]
        # optimal K
        optimal_k_index = MSE.index(min(MSE))
        optimal_k = neighbors[optimal_k_index]

        return optimal_k


def tree(xTrain, yTrain):
    warnings.filterwarnings("ignore")
    tree = DecisionTreeClassifier(max_depth=5, random_state=17)

    tree_params = {'max_depth': range(1,11),
               'max_features': range(4,19)}

    tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True)

    tree_grid.fit(xTrain, yTrain)

    return tree_grid.best_params_



def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create the decision tree classifier


    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)


#-Q1--------------------------
    print("Q3 a:", knn(xTrain, yTrain))

    print("Q3 b:", tree(xTrain, yTrain))

    print("Q3 c:")

if __name__ == "__main__":
    main()
