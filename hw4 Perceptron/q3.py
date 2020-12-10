import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score


def bayesModel(xtrain, xtest, y):

    model = MultinomialNB()
    model.fit(xtrain, y)

    yHat = model.predict(xtest)

    return yHat


def lrModel(xtrain, xtest, y):
    model = LogisticRegression()
    model.fit(xtrain, y)

    yHat = model.predict(xtest)

    return yHat

def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.

    Returns
    -------
    err : int
        The number of mistakes that are made
    """
    count = 0
    for i in range(len(yHat)):
        if yHat[i] != yTrue[i]:
            count += 1
    return count

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
    parser.add_argument("xTrain_bin",
                        default="xtrain_bin.csv",
                        help="filename for features of the training data")
    parser.add_argument("xTest_bin",
                        default="xtest_bin.csv",
                        help="filename for features of the training data")
    parser.add_argument("xTrain_cnt",
                        default="xtrain_cnt.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest_cnt",
                        default="xtest_cnt.csv",
                        help="filename for features of the test data")
    parser.add_argument("yTrain",
                        default="ytrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("yTest",
                        default="ytest.csv",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xtrain_bin = file_to_numpy(args.xTrain_bin)
    xtrain_cnt = file_to_numpy(args.xTrain_cnt)
    xtest_bin = file_to_numpy(args.xTest_bin)
    xtest_cnt = file_to_numpy(args.xTest_cnt)
    ytrain = file_to_numpy(args.yTrain)
    ytest = file_to_numpy(args.yTest)


    bayes_bin = bayesModel(xtrain_bin, xtest_bin, ytrain)
    bayes_cnt = bayesModel(xtrain_cnt, xtest_cnt, ytrain)

    lr_bin = lrModel(xtrain_bin, xtest_bin, ytrain)
    lr_cnt = lrModel(xtrain_cnt, xtest_cnt, ytrain)

    print("Number of mistakes on the test dataset - Naive Bayes Model")
    print("binary: ", calc_mistakes(bayes_bin, ytest))
    print("count: ", calc_mistakes(bayes_cnt, ytest))

    print("Number of mistakes on the test dataset - Linear Regression Model")
    print("binary: ", calc_mistakes(lr_bin, ytest))
    print("count: ", calc_mistakes(lr_cnt, ytest))

if __name__ == "__main__":
    main()
