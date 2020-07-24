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