#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: stacking.py 
@time: 2020/9/9 4:39 上午
@description:
"""
# !/usr/bin/env python
# -*- coding:utf-8 _*-
import os
import pandas as  pd
import numpy as np
from sklearn.model_selection import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import lightgbm as   lgb
from sklearn.metrics import mean_squared_error
from gen_feas import label_inverse
from tqdm import tqdm
from sklearn.metrics import *

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = train['Book-Rating']

# 将lgb和xgb的结果进行stacking

train_oofs = []
for file in tqdm(os.listdir('models')):
    if 'train' in file:
        num_classes_c = file.split('_')[1]
        if '11' in num_classes_c:
            num_classes = 11
        else:
            num_classes = int(num_classes_c[-1])
        oof = pd.read_csv('models/' + file, header=None)
        oof['pred'] = np.argmax(oof.values, axis=1)
        oof['pred'] = oof['pred'].apply(lambda x: label_inverse(x, num_classes))
        train_oofs.append(oof['pred'].values.tolist())
# print(train_oofs)
test_oofs = []
for file in os.listdir('models'):
    if 'test' in file:
        num_classes_c = file.split('_')[1]
        if '11' in num_classes_c:
            num_classes = 11
        else:
            num_classes = int(num_classes_c[-1])
        oof = pd.read_csv('models/' + file, header=None)
        oof['pred'] = np.argmax(oof.values, axis=1)
        oof['pred'] = oof['pred'].apply(lambda x: label_inverse(x, num_classes))
        test_oofs.append(oof['pred'].values)

# train_stack = np.hstack(train_oofs)
train_stack = pd.DataFrame(train_oofs).values.T
# print(train_stack.shape)
# print(pd.DataFrame(train_stack))
# print(pd.DataFrame(train_stack).isnull().sum())
# test_stack = np.hstack(test_oofs)
test_stack = pd.DataFrame(test_oofs).values.T

oof_stack = np.zeros(train_stack.shape[0])
predictions = np.zeros(test_stack.shape[0])

param = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 11,
    'metric': {'multi_logloss'},
    'max_depth': 4,
    'min_child_weight': 6,
    'num_leaves': 64,
    'learning_rate': 0.1,  # 0.05
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1
}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_stack, target)):
    print("fold {}".format(fold_))
    x_train, y_train = train_stack[trn_idx], target.iloc[trn_idx].values
    x_valid, y_valid = train_stack[val_idx], target.iloc[val_idx].values
    trn_data = lgb.Dataset(x_train, y_train)
    val_data = lgb.Dataset(x_valid, y_valid)
    # clf_3 = BayesianRidge()
    # clf_3 = SGDRegressor()
    # clf_3 = LinearRegression()
    # clf_3 = RandomForestRegressor()
    # clf_3 = SVR()
    num_round = 10000
    clf_3 = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                      early_stopping_rounds=100)

    oof_stack[val_idx] = np.argmax(clf_3.predict(x_valid), axis=1)
    print(accuracy_score(oof_stack[val_idx], y_valid))
    predictions += np.argmax(clf_3.predict(test_stack), axis=1) / 5

print(mean_squared_error(target.values, oof_stack))
from pandas import DataFrame

result = DataFrame()
result['id'] = test['id']
result['Book-Rating'] = predictions
print(result['Book-Rating'].describe())
result['Book-Rating']=result['Book-Rating'].astype(int)
print(result['Book-Rating'].value_counts())
result.to_csv('result/stacking_cls.csv', index=False, sep=",", header=False)
