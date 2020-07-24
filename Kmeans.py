#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 11:01:13 2019

@author: robin
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans




def split_data(data, pecentage):
    #training_sample = int(len(data) * pecentage)
    y = data['UCR Literal']
    data = data.drop('UCR Literal',axis = 1)
    x = np.array(data.iloc[:,0:6])
    return x, y



file_name = input('file_name:')
data = pd.read_csv('data/cleaned/' + file_name)
data = data.drop('Unnamed: 0',axis = 1)
data['Occur Time'] = data['Occur Time'].astype(str).str.rjust(4, fillchar="0")
data = data[data['Occur Time'].astype(int) < 2359]  
data = data[(data['Occur Time'].astype(int) % 100) < 59]
date = data['Occur Date'] + ' ' + data['Occur Time'] 
data['Occur Time'] = pd.to_datetime(date).dt.floor('h')
data = data.drop('Occur Date', axis=1)
data = data.drop(['Location'], axis = 1)
encoder = LabelEncoder()
Neighborhood = data['Neighborhood']
zipcode = data['zipcode']
data = data.drop(['zipcode'], axis = 1)
data['Neighborhood'] = encoder.fit_transform(data['Neighborhood'].astype(str))
data['NPU'] = encoder.fit_transform(data['NPU'].astype(str))
pd.to_datetime(data['Occur Time'], unit='s')
dates = data['Occur Time'].apply(lambda x: x.strftime('%Y%m%d%H'))
data['Occur Time'] = (dates.astype(int) % 100).to_list()
data = data.drop(['UCR #','IBR Code'], axis = 1)
K = len(np.unique(data['UCR Literal']))
x, y = split_data(data, 0.9)
kmeans = KMeans(n_clusters=K, random_state=0).fit(x.astype(int))


