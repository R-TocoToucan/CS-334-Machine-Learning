#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 20:41:30 2020

@author: ziruiwang
"""
import pandas as pd
import numpy as np
import os 
import glob 

all_files = glob.glob("*.csv")     # advisable to use os.path.join as this makes concatenation OS independent
df_from_each_file = (pd.read_csv(f) for f in all_files)
for f in all_files:
    print(f)
    print(pd.read_csv(f).shape)
concatenated_df   = pd.concat(df_from_each_file, ignore_index=True)
concatenated_df.to_csv("stock.csv")
print(concatenated_df)