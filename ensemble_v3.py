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

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = train['Book-Rating']
from sklearn.metrics import *


def label_inverse(x, num_classes):
    if num_classes == 3:
        if x == 0:
            return 0
        elif x == 1:  # 1 5
            return 3.5
        else:
            return 7  # 6 10


lgb3 = pd.read_csv('result/' + 'lgb_numclasses3_0.7224976566146037trainoof.csv', header=None)
lgb3['pred'] = np.argmax(lgb3.values, axis=1)
lgb3['pred'] = lgb3['pred'].apply(lambda x: label_inverse(x, 3))
print("lgb3_train",mean_absolute_error(target, lgb3['pred']))

lgb3_test = pd.read_csv('result/' + 'lgb_numclasses3_0.7224976566146037testoof.csv', header=None)
lgb3_test['pred'] = np.argmax(lgb3_test.values, axis=1)
lgb3_test['pred'] = lgb3_test['pred'].apply(lambda x: label_inverse(x, 3))
print(lgb3_test['pred'].value_counts())

cat3 = pd.read_csv('result/' + 'cat_numclasses3_0.7290964243513188trainoof.csv', header=None)
cat3['pred'] = np.argmax(cat3.values, axis=1)
cat3['pred'] = cat3['pred'].apply(lambda x: label_inverse(x, 3))
print(cat3['pred'].value_counts())
print("cat3_train",mean_absolute_error(target, cat3['pred']))

cat3_test = pd.read_csv('result/' + 'cat_numclasses3_0.7290964243513188testoof.csv', header=None)
cat3_test['pred'] = np.argmax(cat3_test.values, axis=1)
cat3_test['pred'] = cat3_test['pred'].apply(lambda x: label_inverse(x, 3))
print(cat3_test['pred'].value_counts())

xgb3 = pd.read_csv('result/' + 'xgb_numclasses3_0.7257421538079711trainoof.csv', header=None)
xgb3['pred'] = np.argmax(xgb3.values, axis=1)
xgb3['pred'] = xgb3['pred'].apply(lambda x: label_inverse(x, 3))
print(xgb3['pred'].value_counts())
print("xgb3_train",mean_absolute_error(target, xgb3['pred']))

xgb3_test = pd.read_csv('result/' + 'xgb_numclasses3_0.7257421538079711testoof.csv', header=None)
xgb3_test['pred'] = np.argmax(xgb3_test.values, axis=1)
xgb3_test['pred'] = xgb3_test['pred'].apply(lambda x: label_inverse(x, 3))
print(xgb3_test['pred'].value_counts())

print("ensemble",mean_absolute_error((lgb3['pred'] + cat3['pred']+xgb3['pred']) / 3, target))

result = pd.DataFrame(test['id'])
result['Book-Rating'] = (lgb3_test['pred'] + cat3_test['pred']+xgb3_test['pred'])/3
# result['Book-Rating']=result['Book-Rating'].astype(int)
print(result['Book-Rating'].value_counts())
result[['id', 'Book-Rating']].to_csv('result/en_v2.csv', index=False,
                                     header=False)
