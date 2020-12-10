#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 23:51:28 2020

@author: ziruiwang
"""
import requests
import pandas as pd
from pandas import json_normalize
import numpy as np
import datetime
#from datetime import datetime

# First, we want to get all the stock fundamental data 
# From 2015 third quarter 
base_url = 'https://www.alphavantage.co/query?'
params = {'function': 'INCOME_STATEMENT',
         'symbol': 'MMM',
         'apikey': 'BQGTYJNQBIZ3NNZT'}

response = requests.get(base_url, params=params)
raw = response.json()
#fundamental_list = list()
#for key in raw['annualReports'][0].keys():
#    if key = 'fiscalDateEnding':
MMM = pd.read_csv('3M.csv')
#MMM["Exchange Date"] = pd.to_datetime(MMM["Exchange Date"], format="%d-%b-%Y")
#MMM["Exchange Date"] = datetime.strptime(MMM["Exchange Date"], '%d-%b-%Y')
for i in range(len(MMM)):
#   print(MMM.iloc[i][0])
#    print(type(MMM.iloc[i][0]))
    a = datetime.datetime.strptime(MMM.iloc[i][0], '%d-%b-%Y').strftime('%Y-%m-%d')
    MMM.at[i,'Exchange Date'] = a
#    print(MMM.at[i,'Exchange Date'])
#    print(MMM.iloc[i][0])
print(MMM)
original_feat = list(MMM.columns)
MMM = MMM.to_numpy()
col_init = MMM.shape[1]
income_by_q = raw['quarterlyReports']
# Create a dataframe for fundamental 
dict_index = 0
a = [[0]*len(income_by_q[0])]*len(MMM)
MMM = np.concatenate((MMM,a),axis=1)

# Name of all features in the income statement
feat_name = list()
for key in income_by_q[0].keys():
    feat_name.append(key)
#print(feat_name)

for i in range(len(MMM)):
    #print(len(MMM))
    if dict_index == 0:
        temp_index = col_init
        if(MMM[i][0]>=income_by_q[dict_index]['fiscalDateEnding']):
            for key in income_by_q[dict_index].keys():
                #print(temp_index)
                #print(key)
                MMM[i][temp_index] = income_by_q[dict_index][key]
                temp_index = temp_index+1
        else:
            dict_index = dict_index+1
            for key in income_by_q[dict_index].keys():
                MMM[i][temp_index] = income_by_q[dict_index][key]
                temp_index = temp_index+1
    elif dict_index < len(income_by_q)-1:
        temp_index = col_init
        if(MMM[i][0]>=income_by_q[dict_index]['fiscalDateEnding'] and 
           MMM[i][0]<income_by_q[dict_index-1]['fiscalDateEnding']):
            for key in income_by_q[dict_index].keys():
                MMM[i][temp_index] = income_by_q[dict_index][key]
                temp_index = temp_index+1
        else:
            dict_index = dict_index+1
            for key in income_by_q[dict_index].keys():
                MMM[i][temp_index] = income_by_q[dict_index][key]
                temp_index = temp_index+1


# Next append the features to the MMM data 
#print(MMM.shape)
#print(MMM)
#print(MMM_fundamental)
   

base_url_cash= 'https://www.alphavantage.co/query?'
params_cash = {'function': 'CASH_FLOW',
         'symbol': 'MMM',
         'apikey': 'BQGTYJNQBIZ3NNZT'}

response_cash = requests.get(base_url_cash, params=params_cash)
raw_cash = response_cash.json()     
        
col_init_cash = MMM.shape[1]
cash_by_q = raw_cash['quarterlyReports']
# Create a dataframe for fundamental 
dict_index = 0
b = [[0]*len(cash_by_q[0])]*len(MMM)
MMM = np.concatenate((MMM,b),axis=1)

# Name of all features in the income statement
feat_name_cash = list()
for key in cash_by_q[0].keys():
    feat_name_cash.append(key)
#print(feat_name)

for i in range(len(MMM)):
    #print(len(MMM))
    if dict_index == 0:
        temp_index = col_init_cash
        if(MMM[i][0]>=cash_by_q[dict_index]['fiscalDateEnding']):
            for key in cash_by_q[dict_index].keys():
                #print(temp_index)
                #print(key)
                MMM[i][temp_index] = cash_by_q[dict_index][key]
                temp_index = temp_index+1
        else:
            dict_index = dict_index+1
            for key in cash_by_q[dict_index].keys():
                MMM[i][temp_index] = cash_by_q[dict_index][key]
                temp_index = temp_index+1
    elif dict_index < len(cash_by_q)-1:
        temp_index = col_init_cash
        if(MMM[i][0]>=cash_by_q[dict_index]['fiscalDateEnding'] and 
           MMM[i][0]<cash_by_q[dict_index-1]['fiscalDateEnding']):
            for key in cash_by_q[dict_index].keys():
                MMM[i][temp_index] = cash_by_q[dict_index][key]
                temp_index = temp_index+1
        else:
            dict_index = dict_index+1
            for key in cash_by_q[dict_index].keys():
                MMM[i][temp_index] = cash_by_q[dict_index][key]
                temp_index = temp_index+1


base_url_bs= 'https://www.alphavantage.co/query?'
params_bs = {'function': 'BALANCE_SHEET',
         'symbol': 'MMM',
         'apikey': 'BQGTYJNQBIZ3NNZT'}

response_bs = requests.get(base_url_bs, params=params_bs)
raw_bs = response_bs.json()     

col_init_bs = MMM.shape[1]
bs_by_q = raw_bs['quarterlyReports']
#print(bs_by_q)
# Create a dataframe for fundamental 
dict_index = 0
c = [[0]*len(bs_by_q[0])]*len(MMM)
MMM = np.concatenate((MMM,c),axis=1)

# Name of all features in the income statement
feat_name_bs = list()
for key in bs_by_q[0].keys():
    feat_name_bs.append(key)
#print(feat_name)

for i in range(len(MMM)):
    #print(len(MMM))
    if dict_index == 0:
        temp_index = col_init_bs
        print(MMM[i][0])
        print(bs_by_q[dict_index]['fiscalDateEnding'])
        if(MMM[i][0]>=bs_by_q[dict_index]['fiscalDateEnding']):
            for key in bs_by_q[dict_index].keys():
                #print(temp_index)
                #print(key)
                MMM[i][temp_index] = bs_by_q[dict_index][key]
                temp_index = temp_index+1
        else:
            dict_index = dict_index+1
            for key in bs_by_q[dict_index].keys():
                MMM[i][temp_index] = bs_by_q[dict_index][key]
                temp_index = temp_index+1
    elif dict_index < len(bs_by_q)-1:
        temp_index = col_init_bs
        print(MMM[i][0])
        print(bs_by_q[dict_index]['fiscalDateEnding'])
        print(MMM[i][0]>=bs_by_q[dict_index]['fiscalDateEnding'] )
        if(MMM[i][0]>=bs_by_q[dict_index]['fiscalDateEnding'] and 
           MMM[i][0]<bs_by_q[dict_index-1]['fiscalDateEnding']):
            for key in bs_by_q[dict_index].keys():
                MMM[i][temp_index] = bs_by_q[dict_index][key]
                temp_index = temp_index+1
        else:
            dict_index = dict_index+1
            for key in bs_by_q[dict_index].keys():
                MMM[i][temp_index] = bs_by_q[dict_index][key]
                temp_index = temp_index+1

feat = original_feat+feat_name+feat_name_cash+feat_name_bs
df_MMM = pd.DataFrame(MMM,columns=feat)
print(df_MMM.shape[1])
df_MMM = df_MMM.loc[:,~df_MMM.columns.duplicated()]
print(df_MMM.shape[1])
#Here is case by case 
df_MMM = df_MMM.drop(columns=['reportedCurrency']) 
df_MMM = df_MMM.fillna(0)
df_MMM = df_MMM.drop(columns=df_MMM.columns[(df_MMM == 'None').any()])
#df_MMM = df_MMM.dropna(axis=1)
print(df_MMM.shape[1])
df_MMM.to_csv("MMM.csv", index=False)

#print(df_MMM)



#print(MMM)
#print(raw['quarterlyReports'])
#print(MMM)
#print(MMM['Exchange Date'].iloc[0]<raw['quarterlyReports'][0]['fiscalDateEnding'])
#test = response.json()['Name']
#print(test)

