# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:18:28 2019

@author: Brandon
"""

# EDA snippets

# imports
import pandas as pd
from matplotlib import pyplot as plt
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# read in data with index column
df = pd.read_csv('BreastCancerDetection/breast-cancer-wisconsin.txt',index_col=0)
# read in large data, specify column names
df = pd.read_csv("~/Documents/Insight/Data - Insight/oppexp06/oppexp.txt", 
                 delimiter="|", low_memory=False, names=df_header.columns, index_col=False)
# read in, parse timestamps
test_results = pd.read_csv('test_results.csv', index_col=0, parse_dates=['timestamp'])

# cast data as an integer, returning NaN if error
pd.to_numeric(df[column],downcast='integer',errors='coerce')
# cast data as a datetime, returning NaN if error
df['TRANSACTION_DT']=pd.to_datetime(df['TRANSACTION_DT'],errors = 'coerce')

# drop NaNs
df = df.dropna()

# combine dataframes
# append
df = happy.append(unhappy)
# join
df = ( df.set_index('CMTE_ID') ).join(cmte_names.set_index('CMTE_ID')) 

# substrings of Dataframe/Series
df['chest_size'] = df['bust size'].str.slice(start=0,stop=2)

# Dataframe type conversion (all elements must be convertible)
df['chest_size'] = df['chest_size'].astype('int64')

?sklearn
import sklearn
dir(sklearn)
?sklearn.linear_model

