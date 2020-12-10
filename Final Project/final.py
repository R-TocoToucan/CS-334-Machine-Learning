# make pandas to print dataframes nicely
pd.set_option('expand_frame_repr', False)

import panda as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import datetime
import time

# newest yahoo API
import yfinance as yahoo_finance
from sklearn import preprocessing

# keras imports
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

import lstm, time

# Data normalization through Minmax Scaling
def normalize(train, test):
    scaler = preprocessing.MinMaxScaler()
    train_data_sc=scaler.fit_transform(train)
    test_data_sc= scaler.transform(test)

    # create pandas dataframe
    train_sc_df = pd.DataFrame(train_data_sc, columns=['Scaled'], index=train.index)
    test_sc_df = pd.DataFrame(test_data_sc, columns=['Scaled'], index=test.index)

    # pearson correlation
    pearson_train = train_data_sc.corr(method='pearson')
    pearson_test = test_data_sc.corr(method='pearson')

    # pcc visualization
    sb.heatmap(pearson_train,
            xticklabels=pearsonc_train.columns,
            yticklabels=pearson_train.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)

    train_sc_df = select_featurers(pearson_train)
    test_sc_df = select_features(pearson_test)

    return train_sc_df, test_sc_df

def select_features(df):

    corr_matrix = d.abs()
    # drop highly correlated features because they give redundent info
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

    df.drop(df[to_drop], axis=1)

    return df

# getting data from yahoo finanace. Not needed when using csv data
def get_data(stock_ticker, start_time, end_time):

    ticker = stock_ticker # eg. 'TSLA'
    connected = False
    while not connected:
        try:
            ticker_df = web.get_data_yahoo(ticker, start=start_time, end=end_time)
            connected = True
            print('connected to yahoo')
        except Exception as e:
            print("type error: " + str(e))
            time.sleep(10)
            pass

    ticker_df = ticker_df.reset_index()
    return ticker_df

# USE LSTM model from keras to predict stock price





def main():

#read the file
df = pd.read_csv('aa.csv')
startdate = datetime.datetime(1999, 1, 1)
enddate = datetime.datetime(2019, 12, 31)



if __name__ == "__main__":
    main()
