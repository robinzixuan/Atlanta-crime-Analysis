#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 00:14:21 2019

@author: robin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from datetime import datetime
from sklearn.linear_model import Ridge,BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn import linear_model
# KernelRidge 4
from sklearn.metrics import accuracy_score
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,BayesianRidge,LogisticRegression,LinearRegression, TheilSenRegressor,RANSACRegressor
from mlens.ensemble import SuperLearner

seed = 2017
np.random.seed(seed)
data = pd.read_csv("training.csv")

def date_time(data):
    data['Year'] = data['Occur Date'].dt.year
    data['Month'] = data['Occur Date'].dt.month
    data['Day'] = data['Occur Date'].dt.day
    data['Hour'] = data['Occur Time'].dt.hour
    data['Day Of Week'] = data['Occur Time'].dt.dayofweek
    return data

date =  pd.DataFrame(data[['Occur Date']].groupby('Occur Date').size().sort_values(ascending=False).rename('date_counts').reset_index())
data['Occur Date'] = pd.to_datetime(data['Occur Date'],format="%Y/%m/%d")
data['Occur Time'] = pd.to_datetime(data['Occur Time'],format="%H:%M")

new_data = date_time(data)
downtown = new_data[(new_data['Neighborhood'] == 'Downtown') & (new_data['UCR Literal'] == 'LARCENY-FROM VEHICLE') & (new_data['Year'] >= 2009)]
ys = []
for i in downtown['Year'].unique().tolist():
    downtown_year = downtown[downtown['Year'] == i]
    downtown_month = pd.DataFrame((downtown_year[['Month']]).groupby('Month').size().rename('counts').reset_index())
    x = downtown_month['Month']
    y = downtown_month['counts']
    ys.append(y)
#     plt.plot(x,y)
temp = []
ys = np.array(ys)
ys = ys.reshape(1,120)[0].tolist()

ensemble = SuperLearner(scorer=accuracy_score, random_state=seed, verbose=2)
ensemble.add([ a, b])
#a is one regression
#b is another regression
#ensemble.fit(X[:75], y[:75])
#y_plot = ensemble.predict(X[75:])

