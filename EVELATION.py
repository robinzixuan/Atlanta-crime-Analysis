#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:14:59 2019

@author: robin
"""

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
#Neighborhood = np.unique(data['Neighborhood'].astype(str))
data = data[data['Neighborhood']=='Midtown']
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
        


    