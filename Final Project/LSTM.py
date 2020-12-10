import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

# keras imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dropout

import os


data = pd.read_csv('MMM_daily.csv')
#label = pd.read_csv('MMM_label.csv')

data = data[::-1]
data2 = pd.read_csv('MMM_daily %Chg.csv') # save for other uses

#data = data[["Exchange Date", "Open", "High", "Low", "Close"]]
#timeframe = data["Exchange Date"]

# processing the closing price with MinMaxScaler
scaler = MinMaxScaler()
data[["Close"]] = scaler.fit_transform(data[["Close"]])

# save the price as separate list
price = data["Close"].values.tolist()

# Chronological train test split
train = data.iloc[int(0.2*len(data)):-1]
test = data.iloc[0:int(0.2*len(data))]
#ytrain = label.iloc[int(0.3*len(label)):-1]
#ytest = label.iloc[0:int(0.3*len(label))]
train = pd.DataFrame(train)
test = pd.DataFrame(test)

# creating dataset
window_size = 50 # store 52 weeks worth of data
seq_length = window_size + 1 # predict next week's price
ans = []
for i in range(len(price)-seq_length):
    ans.append(price[i:i+seq_length])
ans = np.array(ans)
row = int(round(ans.shape[0] * 0.8))
train = ans[:row,:]
np.random.shuffle(train)

xtrain = train[:,:-1]
xtrain = np.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
ytrain = train[:,-1]

xtest = ans[row:,:-1]
xtest = np.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))
ytest = ans[row:,-1]


# creating LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences = True, input_shape=(50,1)))

model.add(LSTM(30, return_sequences = False))

model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')
model.summary()

model.fit(xtrain, ytrain, validation_data = (xtest, ytest), epochs=50, batch_size=10)


# visualizing
train_predict = model.predict(xtrain)
test_predict = model.predict(xtest)

test_predict = pd.DataFrame(test_predict)
ytest = pd.DataFrame(ytest)

pred_perc_change = pd.DataFrame(test_predict.pct_change()*100)
#data2 = data2["%Chg"]
true_perc_change = data2.iloc[0:len(pred_perc_change)]
true_perc_change['%Chg'] = true_perc_change['%Chg'].str.replace('%', '')
true_perc_change['%Chg'] = true_perc_change['%Chg'].str.replace('+', '')
true_perc_change['%Chg'] = true_perc_change['%Chg'].str.replace('-', '').astype(float)

print(pred_perc_change)
print(true_perc_change)

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Price prediction(top) / %change in price(bottom)')
ax1.plot(ytest, label = 'True')
ax1.plot(test_predict, label = "Prediction")
ax2.plot(true_perc_change[1:], label = 'True')
ax2.plot(pred_perc_change[1:], label = "Prediction")


"""
fig = plt.figure(facecolor = 'white')
ax = fig.add_subplot(111)
ax.plot(ytest, label = 'True')
ax.plot(test_predict, label = "Prediction")
ax.legend()
"""

plt.show()
