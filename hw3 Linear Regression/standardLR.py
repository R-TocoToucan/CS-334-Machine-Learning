import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SOMETHING
        ## TODO FILL IN
        timeElapse = 0
        start = time.time()
        self.fit(xTrain, yTrain)

        train_mse = self.mse(xTrain, yTrain)

        test_mse = self.mse(xTest, yTest)
        trainStats['0'] = {}

        end = time.time()
        timeElapse = end - start
        trainStats['0']['time'] = {timeElapse}
        trainStats['0']['train_mse'] = {train_mse}
        trainStats['0']['test_mse'] = {test_mse}

        return trainStats

    def fit(self, x, y):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        x = self.concatenate_(x)
        self.beta = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
    def concatenate_(self, x):
        ones = np.ones(shape = x.shape[0]).reshape(-1, 1)
        return np.concatenate((ones, x), 1)


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    model.fit(xTrain, yTrain)

    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
