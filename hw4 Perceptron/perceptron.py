import argparse
import numpy as np
import pandas as pd
import time
# imported for word counting in 2c
from heapq import nlargest, nsmallest

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def activation(self, x):
        if (np.dot(self.w, x) >= self.bias):
            return 1
        else:
            return 0

    def train(self, xFeat, y):
        """
        Train the perceptron using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the epochs and values the number of mistakes
        """
        stats = {}

        self.w = np.ones(xFeat.shape[1])
        self.bias = 0
        mistakes = 0
        for i in range(self.mEpoch):
            for j in range(len(xFeat)):
                x = xFeat[j]
                label = y[j]
                y_pred = self.activation(x)
                # false positive
                if label == 1 and y_pred == 0:
                    self.w = self.w + x
                    mistakes += 1
                    self.bias -= 1
                # false negative
                elif label == 0 and y_pred == 1:
                    self.w = self.w - x
                    mistakes += 1
                    self.bias += 1
            stats[i] = mistakes

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []
        for x in xFeat:
            result = self.activation(x)
            yHat.append(result)

        return np.array(yHat)

    def outputWords(self, x):
        dict = {}
        i = 0
        for col in x.columns:
            dict[col] = self.w[i]
            i += 1
        pos = nlargest(15, dict, key = dict.get)
        neg = nsmallest(15, dict, key = dict.get)

        return pos, neg


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

    model = Perceptron(args.epoch)
    trainStats_bin = model.train(xtrain_bin, ytrain)
    trainStats_cnt = model.train(xtrain_cnt, ytrain)

    print(trainStats_bin)
    print(trainStats_cnt)

    yHat_bin = model.predict(xtest_bin)
    yHat_cnt = model.predict(xtest_cnt)
    # print out the number of mistakes

    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat_bin, ytest))
    print(calc_mistakes(yHat_cnt, ytest))

    # to get column data
    df = pd.read_csv(args.xTest_cnt)
    pos, neg = model.outputWords(df)
    print("The top 15 positive weighted words are: " + str(pos))
    print("The top 15 negative weighted words are: " + str(neg))
if __name__ == "__main__":
    main()
