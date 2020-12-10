from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def standardize(data_features):
    data_features = (data_features - data_features.mean()) /   data_features.std()
    return data_features

class LinearRegression(ABC):
    """
    Base Linear Regression class from which all
    linear regression algorithm implementations are
    subclasses. Can not be instantiated.
    """
    beta = None      # Coefficients

    @abstractmethod
    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        Train the linear regression and predict the values

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : dictionary
            key refers to the batch number
            value is another dictionary with time elapsed and mse
        """
        pass


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
        # TODO

        for row in xFeat:
            b0 = self.beta[0]
            betas = self.beta[1:]
            prediction = None
            for xi, bi in zip(row, betas):
                prediction = b0 + (xi * bi)

            yHat.append(prediction)
        return yHat

    def mse(self, xFeat, y):
        """
        """
        yHat = self.predict(xFeat)
        
        return mean_squared_error(y, yHat)


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()
