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

MMM = pd.read_csv('MMM.csv')
MMM = MMM.drop(columns = ['Exchange Date','fiscalDateEnding','commonStockSharesOutstanding'])
DJIA_chg = pd.read_csv('DJIA %chg History.csv')
DJIA_chg = DJIA_chg['%Chg']
MMM_chg = MMM['%Chg']
print(DJIA_chg.loc[1])
# Compare 3M stock price percent change with DJIA percentage
# Label 1 if 3M beats DJIA, 0 otherwise 
label = list()
for i in range(len(DJIA_chg)):
    print(MMM_chg.iloc[i])
    print(DJIA_chg.iloc[i])
    if MMM_chg.iloc[i] > DJIA_chg.iloc[i]:
        label.append(1)
    else:
        label.append(0)
# convert label to pandas 
label = pd.DataFrame(label,columns=['Label'])
label.to_csv("MMM_label.csv", index=False)
MMM.to_csv("MMM_rdForest.csv", index=False)

#Pearson Correlation
corr_matrix = MMM.corr()
#fig,ax = plt.subplots(figsize=(20,20))
#sns.heatmap(corr_matrix,annot=True, ax=ax) 

#Reform the Pearson Correlation matrix
#print(corr_matrix)
#print(corr_matrix[(corr_matrix>0.5).any()])
for row in range(len(corr_matrix)):
    for col in range(corr_matrix.shape[1]):
        if(corr_matrix.iloc[row][col]>=0.5 or corr_matrix.iloc[row][col]<=-0.5):
            corr_matrix.iloc[row][col] = 1
        elif(corr_matrix.iloc[row][col]<0.5 and corr_matrix.iloc[row][col]>-0.5):
            corr_matrix.iloc[row][col] = 0
fig,ax = plt.subplots(figsize=(20,20))
sns.heatmap(corr_matrix,annot=True, ax=ax) 
# Eliminate all from balance sheet but short term investments and netreceivables

# First, use correlation table to preprocess
