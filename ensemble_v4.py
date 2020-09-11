#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: ensemble.py 
@time: 2020/9/8 1:36 上午
@description:
"""
import pandas as  pd
import numpy as np
from tqdm import tqdm
import os
from utils import label_inverse

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = train['Book-Rating']
from sklearn.metrics import *

oof_stack = np.zeros(train.shape[0])
predictions = np.zeros(test.shape[0])

no_numclass=[4,5,11]
cnt=0
for file in tqdm(os.listdir('result')):
    if 'train' in file:
        num_classes_c = file.split('_')[1]
        if '11' in num_classes_c:
            num_classes = 11
        else:
            num_classes = int(num_classes_c[-1])
        if num_classes not in no_numclass:
            oof = pd.read_csv('result/' + file, header=None)
            oof['pred'] = np.argmax(oof.values, axis=1)
            oof_stack += oof['pred'].apply(lambda x: label_inverse(x, num_classes))
            cnt+=1
for file in os.listdir('result'):
    if 'test' in file:
        num_classes_c = file.split('_')[1]
        if '11' in num_classes_c:
            num_classes = 11
        else:
            num_classes = int(num_classes_c[-1])
        if num_classes not in no_numclass:
            oof = pd.read_csv('result/' + file, header=None)
            oof['pred'] = np.argmax(oof.values, axis=1)
            predictions += oof['pred'].apply(lambda x: label_inverse(x, num_classes))


print("ensemble", mean_absolute_error(oof_stack / cnt, target))

result = pd.DataFrame(test['id'])
result['Book-Rating'] = predictions / cnt
print(cnt)
result['Book-Rating']=result['Book-Rating'].astype(int)
print(result['Book-Rating'].value_counts())
result[['id', 'Book-Rating']].to_csv('result/en_v2.csv', index=False,
                                     header=False)
