#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:05:25 2019

@author: robin
"""
#AIzaSyDQwfXw28VkcWPR8TYpIMU0Sj5y-N6A1JU
import pandas as pd
import numpy as np
import scipy
from geopy.geocoders import Nominatim
from uszipcode import Zipcode
from uszipcode import SearchEngine
from geopy.extra.rate_limiter import RateLimiter
def one_hot(data_clean_p1, col):
    data_clean_p1[col] = pd.Categorical(data_clean_p1[col])   
    result = pd.get_dummies(data_clean_p1[col],prefix = col)
    data_clean_p1 = pd.concat([data_clean_p1, result], axis=1)
    return data_clean_p1
    
def clean_data(file_name):
    data = pd.read_csv(file_name)
    return data

geolocator = Nominatim()
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
search = SearchEngine(simple_zipcode=True)
filedate = input('clean data date:')
file_name = input("Please input file name need clean:")
data = clean_data(filedate+'/'+file_name)
temp = data['Occur Date']
temp = pd.to_datetime(temp).dt.floor('d')
temp = temp.apply(lambda x: x.strftime('%Y%m%d%H'))
temp = temp.astype(str).to_list()
for i in range(len(temp)):
    if int(temp[i][:4]) < 2009:
        data = data.drop(i, axis=0)
col = ['Occur Date', 'Occur Time', 'Beat', 'Location', 'UCR Literal', 'UCR #', 'IBR Code', 'Neighborhood', 'NPU', 'Latitude', 'Longitude' ]
condition = pd.DataFrame([data['Location'].isnull(), data['Neighborhood'].isnull(), pd.DataFrame([data['Longitude'].isnull(), data['Latitude'].isnull()]).any()]).all()
data = data[condition == False]
data_clean_p1 = data.loc[:, col]
array = []


def clean_col(data_clean_p1, col):
    for i in range(data_clean_p1.shape[0]):
        try:
            if np.isnan(data_clean_p1[col][i]) == True:
                data_clean_p1 = data_clean_p1.drop(i, axis=0)
        except:
            pass
    return data_clean_p1

for i in range(data_clean_p1.shape[0]):
    try:
        result = search.by_coordinates(data_clean_p1['Latitude'][i],data_clean_p1['Longitude'][i], radius=5, returns=1)
        array.append(result[0].zipcode)
    except:
        array.append('None')
data_clean_p1['zipcode'] = array
for i in range(data_clean_p1.shape[0]):
    '''
    if data_clean_p1['Location'][i].isnull():
        try:
            location = geolocator.geocode(str(data_clean_p1['Location'][i]))
            try:
                if (abs(location.latitude - float(data_clean_p1['Latitude'][i]))) > 3.0 or (abs(location.longitude - float(data_clean_p1['Longitude'][i]))) > 3.0:
                    data_clean_p1.drop(data_clean_p1.index[i])
                #time.sleep(10)
            except:
                pass
        except:
            pass
        '''
    try:
        if len(data_clean_p1['Occur Time'][i]) < 4 or not str(data_clean_p1['Occur Time'][i]).isnumeric():
            data_clean_p1 = data_clean_p1.drop(i, axis=0)
        if np.isnan(data_clean_p1['Neighborhood'][i]):
            data_clean_p1['Neighborhood'][i] = 'other'
    except:
        pass         
data_clean_p1 = one_hot(data_clean_p1, 'UCR Literal')       
data_clean_p1.to_csv('sample'+filedate+'.csv')


