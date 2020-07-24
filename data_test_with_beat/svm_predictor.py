
import numpy as np
from sklearn import svm, metrics
import os
from os.path import join

import math
def convertToNumber (s):
    return int.from_bytes(s.encode(), 'little')

def convertFromNumber (n):
    return n.to_bytes(math.ceil(n.bit_length() / 8), 'little').decode()

list_root = '../data_test_with_beat'

with open(join(list_root, 'sample2009-2018.txt')) as f:
    lines = f.read()
    data_list = lines.split('\n')
for i in range(len(data_list)):
    data_list[i] = data_list[i].split(',')
    if i > 0:
        date = data_list[i][1].split('/')[0:2]
        data_list[i][1] = date[0] + '/' + date[1]
        data_list[i][1] = convertToNumber(data_list[i][1])
        if data_list[i][2] == '':
            data_list[i][2] = int(0)
        else:
            data_list[i][2] = int(data_list[i][2])
        data_list[i][5] = convertToNumber(data_list[i][5])
head_title = data_list[0]
data_list = data_list[1:]
data_list = np.asarray(data_list)


with open(join(list_root, 'sample.txt')) as f:
    lines = f.read()
    data_list_sample = lines.split('\n')
for i in range(len(data_list_sample)):
    data_list_sample[i] = data_list_sample[i].split(',')
    if i > 0:
        date2 = data_list_sample[i][1].split('/')[0:2]
        data_list_sample[i][1] = date2[0] + '/' + date2[1]
        data_list_sample[i][1] = convertToNumber(data_list_sample[i][1])
        if data_list_sample[i][2] == '':
            data_list_sample[i][2] = int(0)
        else:
            data_list_sample[i][2] = int(data_list_sample[i][2])
        data_list_sample[i][5] = convertToNumber(data_list_sample[i][5])
data_list_sample = data_list_sample[1:]
data_list_sample = np.asarray(data_list_sample)

train_data = data_list[:, [1, 2, 5]]
data_label = data_list[:, 3]
train_data = train_data.tolist()
data_label = data_label.tolist()

sample_test_data = data_list_sample[:, [1, 2, 5]]
sample_test_label = data_list_sample[:, 3]
sample_test_data = sample_test_data.tolist()
sample_test_label = sample_test_label.tolist()

l = len(train_data)
split_points = [0, 39396, 74898, 109773, 143355, 175621, 206755, 236844, 265865, 292276, l]
years = ['2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018']

for i in range(1,21):
    classifer = svm.SVC(C= 0.5  ,kernel = 'rbf', gamma= i)
    file = open(join(list_root, 'test_result_with_C_' + str(i) + '.csv'), 'w')
    classifer.fit(train_data[split_points[9]: split_points[10]], data_label[split_points[9]: split_points[10]])

    predicted_labels = classifer.predict(sample_test_data)
    file.write('index\n')
    for i in predicted_labels:
        file.write(i + '\n')

    #file.write('\n' + str(np.sum(predicted_labels == np.asarray(sample_test_label))))
    file.close()
