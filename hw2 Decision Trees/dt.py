import argparse
import numpy as np
import pandas as pd
from math import log
from sklearn.metrics import accuracy_score

def ispure(s):
    return(len(set(s)) == 1)

def partition(data, column, value):

    set1 = data[data[column] <= value].index
    set2 = data[data[column] > value].index

    return set1, set2

def entropy(label, idx): # in format of data['label']
    log2 = lambda x:log(x)/log(2)
    unique_label, unique_label_count = np.unique(label.loc[idx], return_counts = True)
    entropy_ = 0.0
    for i in range(len(unique_label)):
        p_i = unique_label_count[i] / sum(unique_label_count)
        entropy_ -= p_i * log2(p_i)
    return entropy_

def entr_gain(set1, set2, label, total_entropy):
    # calculate dataset entropy

    p1 = float(len(set1)) / (len(set1) + len(set2))
    p2 = float(len(set2)) / (len(set1) + len(set2))
    #calculate info gain
    weighted_entropy = p1 * entropy(label, set1) + p2 * entropy(label, set2)

    info_gain = total_entropy - weighted_entropy

    return info_gain

def gini_impurity(label, idx):

    # the unique labels and counts in the data
    unique_label, unique_label_count = np.unique(label.loc[idx], return_counts=True)

    impurity = 1.0
    for i in range(len(unique_label)):
        p_i = unique_label_count[i] / sum(unique_label_count)
        impurity -= p_i ** 2
    return impurity

def gini_gain(set1, set2, label, impurity):

    p = float(len(set1)) / (len(set1) + len(set2))
    info_gain = impurity - p * gini_impurity(label, set1) - (1 - p) * gini_impurity(label, set2)
    return info_gain

def entr_best_split(xFeat, y, idx):

    best_gain = 0
    best_col = None
    best_value = None

    df = xFeat.loc[idx] # converting training data to pandas dataframe
    y_idx = y.loc[idx].index # getting the index of the labels

    entr = entropy(y, y_idx)

    # go through the columns and store the unique values in each column
    for col in df.columns:
        unique_values = set(df[col])
        # loop thorugh each value and partition the data into left_index and right_index
        for value in unique_values:

            left_idx, right_idx = partition(df, col, value)
            # ignore if the index is empty
            if len(left_idx) == 0 | len(right_idx) == 0:
                continue
            # determine the info gain at the node
            info_gain = entr_gain(left_idx, right_idx, y, entr)
            # if the info gain is higher then our current best gain then that becomes the best gain
            if info_gain > best_gain:
                best_gain, best_col, best_value = info_gain, col, value

    return best_gain, best_col, best_value

def gini_best_split(xFeat, y, idx):

    best_gain = 0
    best_col = None
    best_value = None

    df = xFeat.loc[idx] # converting training data to pandas dataframe
    y_idx = y.loc[idx].index # getting the index of the labels

    impurity = gini_impurity(y, y_idx) # determining the impurity at the current node

    # go through the columns and store the unique values in each column
    for col in df.columns:
        unique_values = set(df[col])
        # loop thorugh each value and partition the data into left_index and right_index
        for value in unique_values:

            left_idx, right_idx = partition(df, col, value)
            # ignore if the index is empty
            if len(left_idx) == 0 | len(right_idx) == 0:
                continue
            # determine the info gain at the node
            info_gain = gini_gain(left_idx, right_idx, y, impurity)
            # if the info gain is higher then our current best gain then that becomes the best gain
            if info_gain > best_gain:
                best_gain, best_col, best_value = info_gain, col, value

    return best_gain, best_col, best_value



# Create a terminal node value
def to_terminal(y, idx):
    # gives dictionary of label and counts for leaf
    unique_label, unique_label_counts = np.unique(y.loc[idx], return_counts = True)
    return unique_label[unique_label_counts.argmax()]


class Node():

    def __init__(self, column, value, left, right, depth):
        self.column = column
        self.value = value
        self.left = left
        self.right = right
        self.depth = None

class Leaf:

    def __init__(self, label, idx):
        self.predictions = to_terminal(label, idx)

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int
            Maximum depth of the decision tree
        minLeafSample : int
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.tree = None


    # Build a decision tree
    def buildTree(self, data, label, idx, max_depth, min_size, depth = 0):

        currentdepth = depth

        if self.criterion == 'gini':
            best_gain, best_col, best_value = gini_best_split(data, label, idx)

        if self.criterion == 'entropy':
            best_gain, best_col, best_value = entr_best_split(data, label, idx)
        # if there are no more info gains from split, return leaf

        if (currentdepth == max_depth) | (best_gain == 0):
            return Leaf(label, label.loc[idx].index)

        left_idx, right_idx = partition(data, best_col, best_value)

        currentdepth += 1
        # when depth reaches max, return leaf

        # when set becomes too small, return leaf

        if len(left_idx) <= min_size:
            left_branch = Leaf(label, label.loc[idx].index)
        else:
            left_branch = self.buildTree(data, label, left_idx, max_depth, min_size, depth = currentdepth)

        if len(right_idx <= min_size):
            right_branch = Leaf(label, label.loc[idx].index)
        else:
            right_branch = self.buildTree(data, label, right_idx, max_depth, min_size, depth = currentdepth)

        return Node(best_col, best_value, left_branch, right_branch, depth)

    def fit(self, data, label, idx):
        return self.buildTree(data, label, idx, self.maxDepth, self.minLeafSample)


    def train(self, xFeat, y):
        """
        Train the decision tree model.

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
        self.tree = self.fit(xFeat, y, y.index)
        return self.tree


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
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label
        # TODO
        mytree = self.tree
        for data in xFeat.iterrows():
            yHat.append(self.predict_assist(data[1], mytree))

        return yHat

    def predict_assist(self, data, tree):

        if isinstance(tree, Leaf):
            return tree.predictions

        feature_name, feature_value = tree.column, tree.value
            # pass the observation through the nodes recursively
        if data[feature_name] <= feature_value:
            return self.predict_assist(data, tree.left)
        else:
            return self.predict_assist(data, tree.right)

def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
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
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)

    #print(gini_gain(set1, set2, yTrain, impurity))
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
