#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: stacking.py 
@time: 2020/9/9 4:39 上午
@description:
"""
import pandas as  pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
target = train['Book-Rating']

# 将lgb和xgb的结果进行stacking
oof_lgb1 = pd.read_csv('result/lgb_numclasses3_0.7296304108130445trainoof.csv', header=None).values
oof_lgb2 = pd.read_csv('result/lgb_numclasses4_0.7032108989748551trainoof.csv', header=None).values
oof_lgb3 = pd.read_csv('result/lgb_numclasses5_0.6825444872129117trainoof.csv', header=None).values
oof_lgb4 = pd.read_csv('result/lgb_numclasses11_0.660923976403941trainoof.csv', header=None).values

oof_cat1 = pd.read_csv('result/cat_numclasses3_0.7373865732541545trainoof.csv', header=None).values
oof_cat2 = pd.read_csv('result/cat_numclasses4_0.7079516030610012trainoof.csv', header=None).values
oof_cat3 = pd.read_csv('result/cat_numclasses5_0.6850273270411618trainoof.csv', header=None).values
oof_cat4 = pd.read_csv('result/cat_numclasses11_0.6650222875388965trainoof.csv', header=None).values

predictions_lgb1 = pd.read_csv('result/lgb_numclasses3_0.7296304108130445testoof.csv', header=None).values
predictions_lgb2 = pd.read_csv('result/lgb_numclasses4_0.7032108989748551testoof.csv', header=None).values
predictions_lgb3 = pd.read_csv('result/lgb_numclasses5_0.6825444872129117testoof.csv', header=None).values
predictions_lgb4 = pd.read_csv('result/lgb_numclasses11_0.660923976403941testoof.csv', header=None).values

predictions_cat1 = pd.read_csv('result/cat_numclasses3_0.7373865732541545testoof.csv', header=None).values
predictions_cat2 = pd.read_csv('result/cat_numclasses4_0.7079516030610012testoof.csv', header=None).values
predictions_cat3 = pd.read_csv('result/cat_numclasses5_0.6850273270411618testoof.csv', header=None).values
predictions_cat4 = pd.read_csv('result/cat_numclasses11_0.6650222875388965testoof.csv', header=None).values


train_stack = np.hstack([oof_lgb1, oof_lgb2, oof_lgb3, oof_lgb4,
                         oof_cat1,oof_cat2,oof_cat3,oof_cat4])
print(train_stack.shape)
test_stack = np.hstack([predictions_lgb1, predictions_lgb2, predictions_lgb3, predictions_lgb4,
                        predictions_cat1,predictions_cat2,predictions_cat3,predictions_cat4])

folds_stack = RepeatedKFold(n_splits=5, n_repeats=2, random_state=4590)
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
# result['Book-Rating']=result['Book-Rating'].astype(int)
print(result['Book-Rating'].value_counts())
result.to_csv('result/stacking.csv', index=False, sep=",", header=False)
