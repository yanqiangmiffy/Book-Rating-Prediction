#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: ensemble.py 
@time: 2020/9/3 4:48 上午
@description:
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
# from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.ensemble import ExtraTreesClassifier
from gen_feas import load_data

train, _, test, features = load_data()
train.fillna(value=-1,inplace=True)
test.fillna(value=-1,inplace=True)
scaler = MinMaxScaler()
scaler.fit(train[features].values)
X = scaler.transform(train[features].values)
X_test = scaler.transform(test[features].values)
y=train['Book-Rating'].values


n_splits=5
k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=7)
n_fold = 0
perds=np.zeros(X_test.shape[0])
for train_index, val_index in k_fold.split(X, y):
    x_train, y_train=X[train_index],y[train_index]
    x_val, y_val=X[val_index],y[val_index]

    clf0 = ('clf0',LinearRegression(n_jobs=-1))
    clf1 = ('clf1',SGDRegressor())
    clf2 = ('clf2',RandomForestRegressor(n_jobs=-1))
    clf3 = ('clf3',SVR())
    clf4 = ('clf4',CatBoostRegressor())
    clf5 = ('clf5',KNeighborsRegressor(n_jobs=-1))
    clf6 = ('clf6',XGBRegressor(n_jobs=-1))
    clf7 = ('clf7',LGBMRegressor(n_jobs=-1))

    model = VotingRegressor([clf0,clf1,clf2,clf3,clf4,clf5,clf6,clf7],n_jobs=-1)
    model.fit(x_train, y_train)
    pred=model.predict(X_test)
    perds+=pred[:,0]
    print('验证集正确率为：{}'.format(mean_absolute_error(model.predict(x_val),y_val)))
    print('训练集正确率为：{}'.format(mean_absolute_error(model.predict(x_train),y_train)))

    dataframe = pd.DataFrame({'index': range(len(X_test)), 'label': model.predict(X_test)})

label=perds/n_splits
dataframe = pd.DataFrame({'index':range(len(X_test)),'label':label})
dataframe.to_csv("submisson.csv",index=False,header=False,sep=',')
