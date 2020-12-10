import argparse
import numpy as np
import pandas as pd
import math
import operator
import random


class Knn(object):
    k = 0    # number of neighbors to use

    def __init__(self, k, distance=0):
        """
        Knn constructor

        Parameters
        ----------
        k : int
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        # no training needed
        if isinstance(xFeat, pd.DataFrame):
            xFeat = xFeat.to_numpy()
        self.x = xFeat
        self.y = y


    # function for euclidean distance
    @staticmethod
    def euclidean_distance(x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)

        return np.linalg.norm(x1 - x2)

    def predict(self, test_set):
        if isinstance(test_set, pd.DataFrame):
            test_set = test_set.to_numpy()
        yHat = []
        for xtest in test_set:
            distances = []
            for idx, xtrain in enumerate(self.x):
                dist = self.euclidean_distance(xtest, xtrain)
                # linked with q4. because irrelevant features does not add rows for y, if extra fetures are added, just give them random values
                if idx < len(self.y):
                    distances.append((self.y[idx], dist))
                else:
                    distances.append((random.randint(0, 1), dist))
            distances.sort(key=operator.itemgetter(1))

            # calculate weight. if equal, append 1
            total_weight = 0
            for i in range(self.k):
                total_weight += distances[i][0]
            if total_weight >= self.k/2 :
                yHat.append(1)
            else:
                yHat.append(0)
        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0
    correct = 0
    for x in range(len(yHat)):
        if yHat[x] == yTrue[x]:
            correct += 1
    acc = (correct/float(len(yHat))) * 100.0
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
