import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch


    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SGD
        itercount = 0
        timeElapse = 0

        m = xTrain.shape[0]
        n = xTest.shape[0]
        n_batches = int(m/self.bs)+1
        start = time.time()
        self.beta = np.random.rand(xTrain.shape[1]+1)

        for i in range(self.mEpoch):
            x_train = np.concatenate((np.ones((m, 1)), xTrain), axis = 1)
            x_test = np.concatenate((np.ones((n, 1)), xTest), axis = 1)
            # new train/test dataset contatenated with y
            xyTrain = np.concatenate((x_train, yTrain), axis = 1)
            for j in range(n_batches):
                xyTrain = np.random.permutation(xyTrain)
                shuffle = xyTrain[j*self.bs : (j+1)*self.bs]
                gradient = gradient_desc(self, shuffle, self.beta)
                self.beta = self.beta + (self.lr * gradient)
                itercount += 1
            train_mse = self.mse(xTrain, yTrain)
            test_mse = self.mse(xTest, yTest)
            end = time.time()
            timeElapse = end - start
            dict = {'time': timeElapse, 'train_mse': train_mse, 'test_mse': test_mse}
            trainStats[itercount-n_batches] = dict
        return trainStats

def gradient_desc(self, x, theta):
    total_gradient = 0.0
    for row in x:
        hypothesis = np.matmul(row[:-1], theta)
        error = np.subtract(row[-1], hypothesis)
        total_gradient += row[:-1].dot(error)

    avg_gradient = total_gradient/self.bs

    return avg_gradient


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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)

    print(trainStats)


if __name__ == "__main__":
    main()
