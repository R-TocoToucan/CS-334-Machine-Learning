import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import collections
#from nltk.tokenize import word_tokenize


def model_assessment(filename):
    """
    Given the entire data, decide how
    you want to assess your different models
    to compare perceptron, logistic regression,
    and naive bayes, the different parameters,
    and the different datasets.
    """

    train, test = train_test_split(filename, test_size = 0.3)

    # separating train dataset
    train[['label', 'email']] = train['data'].str.split(n=1,expand=True)
    train = train.drop('data', 1)

    xtrain = train.loc[:, 'email']
    ytrain = train.loc[:, 'label']
    # separating test dataset
    test[['label', 'email']] = test['data'].str.split(n=1,expand=True)
    test = test.drop('data', 1)

    xtest = test.loc[:, 'email']
    ytest = test.loc[:, 'label']
    #xtrain = ytrain = xtest = ytest = np.array([])

    return xtrain, ytrain, xtest, ytest

def build_vocab_map(xtrain):

    wordlist = []

    for line in xtrain:
        words = line.split()
        for word in words:
            if word is not wordlist:
                wordlist.append(word)
    counter = collections.Counter(wordlist)
    output = {k : c for k, c in counter.items() if c >= 30}

    return output

def construct_binary(dataset, vocabmap):
    """
    Construct the email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the
    vocabulary occurs in the email,
    or 0 otherwise
    """
    # create column labels
    columns = []
    for item in vocabmap:
        columns.append(item)
    # create blank data and append values to convert it into pandas dataframe
    data = []
    # separates each email
    for row in dataset:
        # separates words in email string
        tokens = row.split()
        # a stores binary values for words in vocabmap
        a = [0] * len(columns)
        for i in range(len(columns)):
            if columns[i] in tokens:
                a[i] = 1
        data.append(a)

    df_binary = pd.DataFrame(data, columns = columns)


    return df_binary


def construct_count(dataset, vocabmap):
    """
    Construct the email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the
    vocabulary occurs in the email,
    or 0 otherwise
    """
    # create column labels
    columns = []
    for item in vocabmap:
        columns.append(item)
    # create blank data and append values to convert it into pandas dataframe
    data = []
    # separates each email
    for row in dataset:
        # separates words in email string
        tokens = row.split()
        a = [0] * len(columns)
        #
        b = collections.Counter(tokens)
        for i in range(len(columns)):
            if columns[i] in b:
                a[i] = b[columns[i]]

        data.append(a)

    df_count = pd.DataFrame(data, columns = columns)

    return df_count


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    df.columns = ['data']
    #df.to_csv(args.outTrain, index=False)

    xtrain, ytrain, xtest, ytest = model_assessment(df)

    vocabMap = build_vocab_map(xtrain)

    xtrain_bin = construct_binary(xtrain, vocabMap)
    xtest_bin = construct_binary(xtest, vocabMap)
    xtrain_cnt = construct_count(xtrain, vocabMap)
    xtest_cnt = construct_count(xtest, vocabMap)

    # to csv x6
    xtrain_bin.to_csv("xtrain_bin.csv", index=False)
    xtest_bin.to_csv("xtest_bin.csv", index=False)
    xtrain_cnt.to_csv("xtrain_cnt.csv", index=False)
    xtest_cnt.to_csv("xtest_cnt.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)


if __name__ == "__main__":
    main()
