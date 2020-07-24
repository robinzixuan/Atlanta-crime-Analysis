#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 03:04:52 2019

@author: robin
"""

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



def split_data(data, pecentage = 0.9):
    x = np.array(data.iloc[:,0:6])
    y = np.array(data.iloc[:, 6:])
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
data = data.drop(['Location','UCR Literal'], axis = 1)
encoder = LabelEncoder()
Neighborhood = data['Neighborhood']
zipcode = data['zipcode']
data = data.drop(['zipcode'], axis = 1)
data['Neighborhood'] = encoder.fit_transform(data['Neighborhood'].astype(str))
data['NPU'] = encoder.fit_transform(data['NPU'].astype(str))
pd.to_datetime(data['Occur Time'], unit='s')
dates = data['Occur Time'].apply(lambda x: x.strftime('%Y%m%d%H'))
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
data = data.drop(['UCR #','IBR Code'], axis = 1)
x_train, y_train = split_data(data)

test_file_name = input('file_test_name:')
testdata = pd.read_csv('data/cleaned/' + test_file_name)
testdata = testdata.drop('Unnamed: 0',axis = 1)
testdata['Occur Time'] = testdata['Occur Time'].astype(str).str.rjust(4, fillchar="0")

testdata = testdata[testdata['Occur Time'].astype(int) < 2359]  
testdata = testdata[(testdata['Occur Time'].astype(int) % 100) < 59]
date = testdata['Occur Date'] + ' ' + testdata['Occur Time'] 
testdata['Occur Time'] = pd.to_datetime(date).dt.floor('h')
testdata = testdata.drop('Occur Date', axis=1)
testdata = testdata.drop(['Location','UCR Literal'], axis = 1)
encoder = LabelEncoder()
Neighborhood = testdata['Neighborhood']
zipcode = testdata['zipcode']
testdata = testdata.drop(['zipcode'], axis = 1)
testdata['Neighborhood'] = encoder.fit_transform(testdata['Neighborhood'].astype(str))
testdata['NPU'] = encoder.fit_transform(testdata['NPU'].astype(str))
pd.to_datetime(testdata['Occur Time'], unit='s')
dates = testdata['Occur Time'].apply(lambda x: x.strftime('%Y%m%d%H'))
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
testdata = testdata.drop(['UCR #','IBR Code'], axis = 1)
x_test, y_test = split_data(data)
maxlen = 9
num_classes = max(np.max(y_train),np.max(y_test))  + 1

'''
lstm = nn.LSTM(input_size=20,  
               hidden_size= 25,
               num_layers = 6,
               batch_first= True)
'''
model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2,return_sequences=True, input_shape=(1, 6)))
lstm_layers = 3
for i in range(lstm_layers - 1):
        model.add(LSTM(32 * (i+1),activation='tanh',return_sequences=True))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='MSE', 
              optimizer='adam', metrics=['accuracy'])


x_train = x_train.reshape(-1, 1, 6)
x_test = x_test.reshape(-1, 1, 6)
x_train = sequence.pad_sequences(x_train)
x_test = sequence.pad_sequences(x_test)
types = []
accuracy = []
results = []
y_res = []
#accuracys = []
#result = []
for i in range(y_train.shape[1]):
    y_train_t = y_train[:,i]
    y_test_t = y_test[:,i]
    y_train_t = y_train_t.reshape(-1, 1, 1)
    y_test_t = y_test_t.reshape(-1, 1, 1)
    y_train_t = keras.utils.to_categorical(y_train_t, num_classes)
    y_test_t = keras.utils.to_categorical(y_test_t, num_classes)
    model.fit(x_train, y_train_t, batch_size=32, epochs=6)
    score = model.evaluate(x_test, y_test_t, batch_size=32)
    y_result = model.predict(tf.cast(x_test, tf.float32))
    result = mean_squared_error(y_test_t.reshape(y_test_t.shape[0], num_classes), y_result.reshape(y_result.shape[0], num_classes))
    accuracy.append(score[1])
    y_res.append(y_result)
   # model.save('lstm'+str(i)+'.h5')
    results.append(result)
    types.append(i)
print(types)
print(accuracy)
print(results)
#plt.plot(types, accuracy)
#plt.plot(types, results)