#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:59:28 2019

@author: robin
"""
import numpy as np
import pandas as pd
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
p.to_csv('result.csv')
