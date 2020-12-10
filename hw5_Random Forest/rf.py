import argparse
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from statistics import mode
import random
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def decisionTree(dataset, criterion, maxDepth, minLeafSample):
    xtrain = dataset.iloc[:, :-1]
    ytrain = dataset.iloc[:,-1]

    model = DecisionTreeClassifier(criterion = criterion, max_depth = maxDepth, min_samples_leaf = minLeafSample)
    model = model.fit(xtrain, ytrain)

    return model

# create random subsample from data
def bootstrap(data):
    index = list()
    sample = list()
    oob = list()

    data = data.to_numpy() # changed to numpy array for ease of iteration

    while len(index) < len(data):
        index.append(random.randrange(len(data)))

    for i in index:
        sample.append((data[i], i))

    for idx in range(len(data)):
        if idx not in index:
            oob.append((data[idx], idx))

    index = list(set(index)) # contains only unique index values
    return sample, oob

def subsample(data, maxFeat):
    index = random.sample(range(0, len(data.columns)-1), maxFeat) # generates non-duplicate random index from 0 to column length -1 (since dataset includes label y) giving range 0 - 10

    return index

def oob_err(trees, samples, features, oob, data):
    results = list()
    oob = pd.DataFrame(oob)

    sample_df = pd.DataFrame(samples).T
    sample = [] # contains datapoint indices for each tree

    for i in range(np.array(samples).shape[0]): # 0~10
        temp = []
        for j in range(np.array(samples).shape[1]): # i = 0~1119
            temp.append(sample_df[i][j][1])
        sample.append(temp)

    for row in oob[1]: # index of oob datapoints
        treeno = [] # this array stores index of trees that does not contain oob sample
        for k in range(np.array(samples).shape[0]):
            if row in sample[k]:
                treeno.append(k)
        votes = list()

        for idx in treeno: # for each tree get prediction
            feature = (data.iloc[row, features[idx]])
            prediction = trees[idx].predict([feature])
            votes.append(prediction[0])
            results.append(np.argmax(np.bincount(votes))) # append to results the value with highest votes

    y = data.iloc[:, -1]
    count = 0
    for j in range(len(oob)):
        if y[j] == results[j]:
            count += 1

    return count/len(oob)

class RandomForest(object):
    nest = 0           # number of trees
    maxFeat = 0        # maximum number of features
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int
            Maximum depth of the decision tree
        minLeafSample : int
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.maxFeat = maxFeat

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        trees = list() # keeps track of trees and datas
        sample_list = list()
        feature_list = list()
        oob_list = list()
        a = None
        dataset = np.concatenate((xFeat, y), axis=1)
        df = pd.DataFrame(dataset)

        for i in range(self.nest):
            sample1, oob = bootstrap(df)
            sample_list.append(sample1) # this stores array of dataset used to create tree to a list
            oob_list.append(oob)

            features = subsample(df, self.maxFeat)
            feature_list.append(features) # stores index of features to a separate list
            sample_df=pd.DataFrame(sample1)[0]
            sample_df = pd.DataFrame(sample_df.values.tolist())

            y = pd.DataFrame(y)

            xFeat = pd.DataFrame(xFeat)
            a = xFeat.iloc[1,features]

            sample2 = sample_df.iloc[:, features]
            sample2 = pd.concat([sample2, y], axis=1)

            tree = decisionTree(sample2, 'gini', self.maxDepth, self.minLeafSample)

            trees.append(tree)

        self.trees = trees
        self.features = feature_list
        oob_error = 0.0

        for data in oob_list:
            oob_error += oob_err(self.trees, sample_list, feature_list, data, df)

        return oob_error/len(oob_list)

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
        for row in xFeat:
            row_df = pd.DataFrame(row)
            votes = []
            for tree, features in zip(self.trees, self.features):
                xtest = row_df.iloc[features]
                votes.append(tree.predict(xtest.T)[0])
            yHat.append(np.argmax(np.bincount(votes)))

        return yHat


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
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)
    model = RandomForest(args.epoch, 5, 'gini', 5, 5)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    count = 0
    for i in range(len(yTest)):
        if yHat[i] == yTest[i]:
            count += 1
    print(count/len(yTest))

if __name__ == "__main__":
    main()
