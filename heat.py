#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:15:28 2019

@author: robin
"""

import seaborn as sns
from sklearn import svm, metrics
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file_name = input('file_name:')
data = pd.read_csv('data/cleaned/' + file_name)
data = data.drop('Unnamed: 0',axis = 1)
real = data['UCR Literal']
year = []
accuracy = []
#for i in range(1,21):
#year.append(i)
file_name1 = 'data_test_with_beat/test_result_with_C_' + str(5) +'.csv'
data_get = pd.read_csv(file_name1)
#data['Occur Date'] = pd.to_datetime(data['Occur Date'])
#dates = data['Occur Date'].apply(lambda x: x.strftime('%Y'))
#time = int(file_name1.split('.')[0][-4:])
#data = data[dates.astype(int) == time]
#accuracy.append(metrics.confusion_matrix(real, data_get, average='micro'))
metric = metrics.confusion_matrix(real, data_get)
temp = (metric - metric.mean()) / (metric.max() - metric.min())
#result.append(metric)
index = list(np.unique(real))
sns.heatmap(temp, cmap="Blues", xticklabels= index, yticklabels=index)

#plt.plot(year, accuracy)
plt.xlabel('Predict label')
plt.ylabel('True label')

