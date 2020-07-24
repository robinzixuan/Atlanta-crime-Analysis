#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:51:04 2019

@author: robin
"""

import pandas as pd
import numpy as np
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
sns.set()
file_name = input('file_name:')
data = pd.read_csv('data/cleaned/' + file_name)
data = data.drop('Unnamed: 0',axis = 1)
data['Occur Time'] = data['Occur Time'].astype(str).str.rjust(4, fillchar="0")
data = data[data['Occur Time'].astype(int) < 2359]  
data = data[(data['Occur Time'].astype(int) % 100) < 59]
encoder = LabelEncoder()
data['Location'] = encoder.fit_transform(data['Location'].astype(str))
data['Neighborhood'] = encoder.fit_transform(data['Neighborhood'].astype(str))
data['UCR Literal'] = encoder.fit_transform(data['UCR Literal'].astype(str))
data['NPU'] = encoder.fit_transform(data['NPU'].astype(str))
data['Occur Date'] = pd.to_datetime(data['Occur Date'])
data['IBR Code'] = encoder.fit_transform(data['IBR Code'].astype(str))
dates = data['Occur Date'].apply(lambda x: x.strftime('%Y%m%d'))
data['Occur Date'] = dates.astype(int).to_list()
for i in range(data.shape[0]):
    try:    
        if data['zipcode'][i] == 'None':
                data = data.drop(i, axis=0)
    except:
        pass
index = list(data.columns)
ax = sns.heatmap(data.astype(float).corr(), xticklabels= index, yticklabels=index)

#plt.imshow(data.astype(float), cmap='hot', interpolation='nearest')
#plt.show()
