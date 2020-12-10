#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 20:40:50 2020

@author: ziruiwang
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('stock.csv')
df = df[df['Exchange Date']>='2015-12-25']
#df = df.drop(df.index[347:358])
df_chg = df['%Chg']

#df = df.drop(columns = ['Exchange Date','fiscalDateEnding','commonStockSharesOutstanding'])
DJIA_chg = pd.read_csv('DJIA %chg History.csv')
DJIA_chg = DJIA_chg['%Chg']
label = list()
for i in range(len(DJIA_chg)):
    #print(df_chg.iloc[i])
    #print(DJIA_chg.iloc[i])
    if df_chg.iloc[i] > DJIA_chg.iloc[i]:
        label.append(1)
    else:
        label.append(0)
label = pd.DataFrame(label,columns=['Label'])
label.to_csv("df_label.csv", index=False)
df = df.drop(['Unnamed: 0','deferredLongTermLiabilities',
       'preferredStockTotalEquity', 'nonRecurring', 'changeInExchangeRate',
       'accumulatedDepreciation', 'extraordinaryItems',
       'discontinuedOperations', 'preferredStockAndOtherAdjustments',
       'warrants', 'capitalLeaseObligations'],axis=1)
df = df.fillna(0)
df = df.drop_duplicates()
df = df.drop(['Exchange Date','%Chg','fiscalDateEnding'],axis=1)
df.to_csv("df_rdForest.csv", index=False)
print(df)