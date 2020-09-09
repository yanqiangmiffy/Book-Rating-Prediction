#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: stacking.py 
@time: 2020/9/9 4:39 上午
@description:
"""
import os
import pandas as  pd
import numpy as np
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from gen_feas import label_inverse
from tqdm import  tqdm
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = train['Book-Rating']

# 将lgb和xgb的结果进行stacking

train_oofs=[]
for file in tqdm(os.listdir('models')):
    if 'train' in file:
        num_classes_c=file.split('_')[1]
        if '11' in num_classes_c:
            num_classes=11
        else:
            num_classes=int(num_classes_c[-1])
        oof=pd.read_csv('models/'+file,header=None)
        oof['pred']=np.argmax(oof.values,axis=1)
        oof['pred']=oof['pred'].apply(lambda x:label_inverse(x,num_classes))
        train_oofs.append(oof['pred'].values.tolist())
# print(train_oofs)
test_oofs=[]
for file in os.listdir('models'):
    if 'test' in file:
        num_classes_c=file.split('_')[1]
        if '11' in num_classes_c:
            num_classes=11
        else:
            num_classes=int(num_classes_c[-1])
        oof=pd.read_csv('models/'+file,header=None)
        oof['pred']=np.argmax(oof.values,axis=1)
        oof['pred']=oof['pred'].apply(lambda x:label_inverse(x,num_classes))
        test_oofs.append(oof['pred'].values)

# train_stack = np.hstack(train_oofs)
train_stack = pd.DataFrame(train_oofs).values.T
# print(train_stack.shape)
# print(pd.DataFrame(train_stack))
# print(pd.DataFrame(train_stack).isnull().sum())
# test_stack = np.hstack(test_oofs)
test_stack = pd.DataFrame(test_oofs).values.T

scaler = MinMaxScaler()
train_stack=scaler.fit_transform(train_stack)
test_stack=scaler.transform(test_stack)

folds_stack = StratifiedKFold(n_splits=5, shuffle=True, random_state=4590)
oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])


for fold_, (trn_idx, val_idx) in enumerate(folds_stack.split(train_stack, target)):
    print("fold {}".format(fold_))
    trn_data, trn_y = train_stack[trn_idx], target.iloc[trn_idx].values
    val_data, val_y = train_stack[val_idx], target.iloc[val_idx].values

    clf_3 = BayesianRidge()
    # clf_3 = SGDRegressor()
    # clf_3 = LinearRegression()
    # clf_3 = RandomForestRegressor()
    # clf_3 = SVR()
    clf_3.fit(trn_data, trn_y)

    oof_stack[val_idx] = clf_3.predict(val_data)
    predictions += clf_3.predict(test_stack) / 10

print(mean_squared_error(target.values, oof_stack))
from pandas import DataFrame

result = DataFrame()
result['id'] = test['id']
result['Book-Rating'] = predictions
print(result['Book-Rating'].describe())
result['Book-Rating'] = result['Book-Rating'].apply(lambda x: 0 if x < 0 else x)
result['Book-Rating'] = result['Book-Rating'].apply(lambda x: 10 if x >10 else x)
result['Book-Rating']=result['Book-Rating'].astype(int)
print(result['Book-Rating'].value_counts())
result.to_csv('result/stacking_re.csv', index=False, sep=",", header=False)
