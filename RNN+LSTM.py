#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 01:23:31 2019

@author: robin
"""

import pandas as pd
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense, Embedding, LSTM
import tensorflow as tf
import keras
from sklearn import svm, metrics



def split_data(data, pecentage = 0.9):
    x = np.array(data.iloc[:,0:1])
    y = np.array(data.iloc[:, 1:])
    return x, y 


location = input('sad location:')
file_name = input('file_name:')
data = pd.read_csv('data/cleaned/' + file_name)
data = data.drop('Unnamed: 0',axis = 1)
data['Occur Time'] = data['Occur Time'].astype(str).str.rjust(4, fillchar="0")
data = data[data['Occur Time'].astype(int) < 2359]  
data = data[(data['Occur Time'].astype(int) % 100) < 59]
date = data['Occur Date'] + ' ' + data['Occur Time'] 
data['Occur Time'] = pd.to_datetime(date).dt.floor('d')
data = data.drop('Occur Date', axis=1)
data = data.drop(['Location','UCR Literal'], axis = 1)
encoder = LabelEncoder()
Neighborhood = data['Neighborhood']
zipcode = data['zipcode']
data = data.drop(['zipcode'], axis = 1)
#data['Neighborhood'] = encoder.fit_transform(data['Neighborhood'].astype(str))
data['NPU'] = encoder.fit_transform(data['NPU'].astype(str))
pd.to_datetime(data['Occur Time'], unit='s')
dates = data['Occur Time'].apply(lambda x: x.strftime('%Y%m%d'))
data['Occur Time'] = dates.astype(int).to_list()
IBR = list(data['IBR Code'])
for j in range(len(IBR)) :
    try:
        if len(IBR[j]) > 4:
            IBR[j] = IBR[j][0:4]
    except:
        pass
IBR = pd.Series(IBR)
IBR = IBR[IBR.isna() == False]
data['IBR Code'] = IBR.astype(int)
data = data.drop(['UCR #','IBR Code','Longitude','Latitude', 'NPU'], axis = 1)
Neighborhood_train = np.unique(data['Neighborhood'].astype(str))


test_file_name = input('file_test_name:')
testdata = pd.read_csv('data/cleaned/' + test_file_name)
testdata = testdata.drop('Unnamed: 0',axis = 1)
testdata['Occur Time'] = testdata['Occur Time'].astype(str).str.rjust(4, fillchar="0")
testdata = testdata[testdata['Occur Time'].astype(int) < 2359]  
testdata = testdata[(testdata['Occur Time'].astype(int) % 100) < 59]
date = testdata['Occur Date'] + ' ' + testdata['Occur Time'] 
testdata['Occur Time'] = pd.to_datetime(date).dt.floor('d')
testdata = testdata.drop('Occur Date', axis=1)
testdata = testdata.drop(['Location','UCR Literal'], axis = 1)
encoder = LabelEncoder()
Neighborhood = testdata['Neighborhood']
zipcode = testdata['zipcode']
testdata = testdata.drop(['zipcode'], axis = 1)
#testdata['Neighborhood'] = encoder.fit_transform(testdata['Neighborhood'].astype(str))
testdata['NPU'] = encoder.fit_transform(testdata['NPU'].astype(str))
pd.to_datetime(testdata['Occur Time'], unit='s')
dates = testdata['Occur Time'].apply(lambda x: x.strftime('%Y%m%d'))
testdata['Occur Time'] = dates.astype(int).to_list()
IBR = list(testdata['IBR Code'])
for j in range(len(IBR)) :
    try:
        if len(IBR[j]) > 4:
            IBR[j] = IBR[j][0:4]
    except:
        pass
IBR = pd.Series(IBR)
IBR = IBR[IBR.isna() == False]
testdata['IBR Code'] = IBR.astype(int)
testdata = testdata.drop(['UCR #','IBR Code','Longitude','Latitude', 'NPU'], axis = 1)
Neighborhood_test = np.unique(testdata['Neighborhood'].astype(str))   
'''                      
for i in Neighborhood_test.astype(str):
    if i in Neighborhood_train.astype(str):
        data_train = data[data['Neighborhood']==i]
        data_test = testdata[testdata['Neighborhood']==i]
        data_train = data_train.drop(['Neighborhood'], axis = 1)
        data_test = data_test.drop(['Neighborhood'], axis = 1)
        x_train, y_train = split_data(data)
        x_test, y_test = split_data(data)
        maxlen = 2
        num_classes = max(np.max(y_train),np.max(y_test))  + 1
        model = Sequential()
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True, input_shape=(1, 2)))
        lstm_layers = 3
        for i in range(lstm_layers - 1):
                model.add(LSTM(32,activation='tanh',return_sequences=True))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='MSE', 
                      optimizer='adam', metrics=['accuracy'])
'''
data_train = data[data['Neighborhood']==location]
data_train = data_train.drop(['Neighborhood'], axis = 1)

data_test = testdata[testdata['Neighborhood']==location]
data_test = data_test.drop(['Neighborhood'], axis = 1)
data_train_set = pd.DataFrame([])
data_train_set['Occur Time'] = np.unique(data_train['Occur Time'])
y = data_train.iloc[:, 2:]
labels = y.columns
for i in labels:
    data_train_set[i] = np.array(data_train.groupby('Occur Time')[i].sum())
    data_train_set[i] = np.array(data_train.groupby('Occur Time')[i].sum())
data_test_set = pd.DataFrame([])
data_test_set['Occur Time'] = np.unique(data_test['Occur Time'])
y = data_test.iloc[:, 2:]
labels = y.columns
for i in labels:
    data_test_set[i] = np.array(data_test.groupby('Occur Time')[i].sum())
    data_test_set[i] = np.array(data_test.groupby('Occur Time')[i].sum())
x_train, y_train = split_data(data_train_set)
x_test, y_test = split_data(data_test_set)
maxlen = 2
num_classes = max(np.max(y_train),np.max(y_test))  + 1
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True, input_shape=(1, 1)))
lstm_layers = 3
for i in range(lstm_layers - 1):
        model.add(LSTM(32,activation='tanh',return_sequences=True))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='MSE', 
              optimizer='adam', metrics=['accuracy'])
x_train = x_train.reshape(-1, 1, 1)
x_test = x_test.reshape(-1, 1, 1)
x_train = sequence.pad_sequences(x_train)
x_test = sequence.pad_sequences(x_test)
#types = []
#accuracy = []
f1_score= []
epoch = []
results = []
result = []
for i in range(y_train.shape[1]):
    y_train_t = y_train[:,i]
    y_test_t = y_test[:,i]
    y_train_t = y_train_t.reshape(-1, 1, 1)
    y_test_t = y_test_t.reshape(-1, 1, 1)
    y_train_t = keras.utils.to_categorical(y_train_t, num_classes)
    y_test_t = keras.utils.to_categorical(y_test_t, num_classes)
    model.fit(x_train, y_train_t, batch_size=32, epochs=8)
    score = model.evaluate(x_test, y_test_t, batch_size=32)
    y_result = model.predict(tf.cast(x_test, tf.float32))
    results.append(y_result)
    result.append(np.argmax(y_result))
    #result = mean_squared_error(y_test_t.reshape(y_test_t.shape[0], num_classes), y_result.reshape(y_result.shape[0], num_classes))
    #accuracy.append(score[1])
    #model.save('lstm'+str(i)+'.h5')
    #results.append(result)
    #types.append(i)

from numpy.random import choice
import random
re=[]
s = np.array(results)
non = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
for i in range(s.shape[0]):
    res=[]
    for j in range(s.shape[1]):
        res.append(random.choices(population=non, weights=s[i][j][0], k = 1))
                #choice(non, 1, p= s[i][j][0]))
    re.append(res)
re=np.array(re)
re = np.transpose(re)
re = re[0]
p = pd.DataFrame(re)
p['Occur Time'] = data_test_set['Occur Time']


#file_name = input('file_name:')
file_name = 'sampleCOBRA-2019.csv'
data = pd.read_csv('data/cleaned/' + file_name)
data = data.drop('Unnamed: 0',axis = 1)
data['Occur Time'] = data['Occur Time'].astype(str).str.rjust(4, fillchar="0")
data = data[data['Occur Time'].astype(int) < 2359]  
data = data[(data['Occur Time'].astype(int) % 100) < 59]
date = data['Occur Date'] + ' ' + data['Occur Time'] 
data['Occur Time'] = pd.to_datetime(date).dt.floor('d')
data = data.drop('Occur Date', axis=1)
data = data.drop(['Location','UCR Literal'], axis = 1)
encoder = LabelEncoder()
Neighborhood = data['Neighborhood']
zipcode = data['zipcode']
data = data.drop(['zipcode'], axis = 1)
#data['Neighborhood'] = encoder.fit_transform(data['Neighborhood'].astype(str))
data['NPU'] = encoder.fit_transform(data['NPU'].astype(str))
pd.to_datetime(data['Occur Time'], unit='s')
dates = data['Occur Time'].apply(lambda x: x.strftime('%Y%m%d'))
data['Occur Time'] = dates.astype(int).to_list()
IBR = list(data['IBR Code'])
for j in range(len(IBR)) :
    try:
        if len(IBR[j]) > 4:
            IBR[j] = IBR[j][0:4]
    except:
        pass
IBR = pd.Series(IBR)
IBR = IBR[IBR.isna() == False]
data['IBR Code'] = IBR.astype(int)
data = data.drop(['UCR #','IBR Code','Longitude','Latitude', 'NPU'], axis = 1)
#Neighborhood = np.unique(data['Neighborhood'].astype(str))
data = data[data['Neighborhood']==location]
data = data.drop(['Neighborhood','Beat'], axis = 1)
data = data.reset_index(drop=True)
real = []
predicts = []
for i in range(data.shape[0]):
    t = data.iloc[i]
    predict = p[p['Occur Time'] == t['Occur Time']]
    predict = predict.drop(['Occur Time'], axis = 1)
    predict = np.array(predict.iloc[0,:].values.tolist())
    if np.sum(predict>0) < 3:
        pred = predict.argsort()[-1 * np.sum(predict>0):][::-1]
    else:
        pred = predict.argsort()[-3:][::-1]
    pred = list(pred)
    t = np.array(t)[1:]
    real.append(np.argmax(t))
    if np.argmax(t) in pred:
        predicts.append(np.argmax(t))
    else:
        predicts.append(pred[0])
        
print(metrics.f1_score(real, predicts, average='micro'))
    

    
