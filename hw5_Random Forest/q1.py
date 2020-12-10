import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt

def lr(xtrain, ytrain, xtest, ytest):
    xnorm = preprocessing.StandardScaler().fit(xtrain).transform(xtrain)
    logmodel = LogisticRegression()
    logmodel.fit(xnorm, ytrain)

    testnorm = preprocessing.StandardScaler().fit(xtest).transform(xtest)

    predictions = logmodel.predict(testnorm)

    acc = accuracy_score(predictions, ytest)
    return acc

def pca(xFeat, n):
    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(xFeat)
    pca = PCA(n_components = n)
    principalComp = pca.fit_transform(data)

    print(abs(pca.components_))

    return(pca.n_components_, pca.explained_variance_ratio_)


def pca_lr(xtrain, xtest, ytrain, ytest):

    scaler = preprocessing.MinMaxScaler()
    data = scaler.fit_transform(xtrain)
    pca = PCA(n_components = 0.95)
    pca.fit(data)

    xtrain_pca = pca.transform(xtrain)
    xtest_pca = pca.transform(xtest)

    model = LogisticRegression().fit(xtrain_pca, ytrain)

    y_pred_train = model.predict(xtrain_pca)
    y_pred_test = model.predict(xtest_pca)

    train_score = accuracy_score(ytrain, y_pred_train)
    test_score = accuracy_score(ytest, y_pred_test)

    y_pred = model.predict_proba(xtest)
    y_pred_pca = pca.predict_proba(xtest_pca)
    fpr, tpr, _ = metrics.roc_curve(ytest,  y_pred_proba)
    fpr2, tpr2 = metrics.roc_curve(ytest, y_pred)
    auc = metrics.roc_auc_score(ytest, y_pred_proba)
    auc2 = metrics.roc_auc_score(ytet, y_pred)

    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.show()

    return train_score, test_score

def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        default="xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        default="yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        default="xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        default="yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    lr_prob = lr(xTrain, yTrain, xTest, yTest)
    print(lr_prob)

    principalComp, variance = pca(xTrain, 0.95)
    print(principalComp)
    print(variance)

    acc1, acc2 = pca_lr(xTrain, xTest, yTrain, yTest)
    print(acc1)
    print(acc2)


if __name__ == "__main__":
    main()
