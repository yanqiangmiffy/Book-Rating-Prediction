#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: cat.py 
@time: 2020/9/5 10:13 下午
@description:
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

data_train = pd.read_csv('data/train.csv')
data_test = pd.read_csv('data/test.csv')
data_item = pd.read_csv("data/book.csv", encoding='ISO-8859-1')
data_user = pd.read_csv("data/user.csv", encoding='ISO-8859-1')

# 重命名列
data_train.columns = ['userid', 'bookid', 'score']
data_test.drop(columns=['id'], inplace=True)
data_test.columns = ['userid', 'bookid']

# 删除多余列
df_item = data_item.drop(columns=['Book-Title', 'Book-Author', 'Publisher'])
df_user = data_user.drop(columns=['Location'])

# 数据合并
df_merge_train = pd.merge(data_train, df_item, how='inner', left_on='bookid', right_on='ISBN', sort=True)
df_merge_train.drop(columns=['ISBN'], inplace=True)

dftrain = pd.merge(df_merge_train, df_user, how='inner', left_on='userid', right_on='User-ID', sort=True)
dftrain.drop(columns=['User-ID'], inplace=True)

# set(dftrain['Year-Of-Publication'].value_counts().index)
# 发现到年龄中存在多个字符串，处理脏数据
# dftrain[dftrain['Year-Of-Publication'] == 'Amit Chaudhuri']
# 脏数据数据值 采用 众数 填补
dftrain.iloc[dftrain[dftrain['Year-Of-Publication'] == 'Amit Chaudhuri'].index[0], 3] = \
dftrain['Year-Of-Publication'].value_counts().index[0]
# print(set(dftrain['Year-Of-Publication'].value_counts().index))
dftrain['Year-Of-Publication'] = dftrain['Year-Of-Publication'].astype(int)

# ISBN编码处理
lbe = LabelEncoder()
## 建立一个ISBN的编码索引
index_ISBN = pd.DataFrame({'ISBN': dftrain['bookid'].values, 'code': lbe.fit_transform(dftrain['bookid'])})
index_ISBN.drop_duplicates(subset=['ISBN', 'code'], keep='first', inplace=True)
# 训练集ISBN编码处理
dftrain['bookid'] = lbe.fit_transform(dftrain['bookid'])

# 查看现有年龄分布
#plt.figure(figsize=(16, 8))
#plt.plot(dftrain['Age'].value_counts().sort_index().index, dftrain['Age'].value_counts().sort_index().values, 'o-')
#plt.xticks(np.arange(0, 250, 10))
#plt.yticks(np.arange(0, 25000, 2000))
#plt.show()

# 使用随机森林填补缺失值

# 不再原有列表进行操作，创建一个新的表（无sorce）
df = dftrain.copy()
df.drop(['score'], axis=1, inplace=True)

# 获取填补值 和 原有特征
fillc = df.iloc[:, 3]
feature = df.iloc[:, df.columns != 'Age']

# 构成训练集和测试集
ytrain = fillc[fillc.notnull()]
ytest = fillc[fillc.isnull()]
xtrain = feature.iloc[ytrain.index, :]
xtest = feature.iloc[ytest.index, :]

### ---Time : 3min 40 second --
start_time = time.time()

# 建模 预测年龄
rfc = RandomForestRegressor(n_estimators=100).fit(xtrain, ytrain)
ypredict = rfc.predict(xtest)
# 填补缺失值
dftrain.iloc[ytest.index, 4] = ypredict

print(f'预计耗时:{time.time() - start_time}')

# 全部转变为整型变量
dftrain = dftrain.astype(int)
# 查看数据
dftrain.head()

### ---Time : xxx  --

start_time = time.time()

# 获取 标签和特征
y = dftrain[['score']]

x = dftrain.iloc[:, dftrain.columns != 'score']

# 交叉验证
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

categorical_features_indices = [0, 1]

clf = CatBoostClassifier(iterations=50, learning_rate=0.1, od_type="Iter", l2_leaf_reg=3, depth=10,
                         cat_features=categorical_features_indices,
                         random_seed=2020)

clf.fit(x_train, y_train, eval_set=(x_test, y_test))
pred = clf.predict(x_test)

print(f'预计耗时:{time.time() - start_time}')

# 查看预测结果
cfm = pd.DataFrame(metrics.confusion_matrix(y_test, pred))
print("train_accuracy：", clf.score(x, y), "\n", "test_accuracy：", clf.score(x_test, y_test))

# 测试集数据预处理
df_tset_merge = pd.merge(data_test, df_item, how='inner', left_on='bookid', right_on='ISBN', sort=True)
df_tset_merge = df_tset_merge.drop(columns=['ISBN'])

df_test = pd.merge(df_tset_merge, df_user, how='inner', left_on='userid', right_on='User-ID', sort=True)
df_test = df_test.drop(columns=['User-ID'])

df_test.iloc[df_test[df_test['Year-Of-Publication'] == 'Amit Chaudhuri'].index[0], 2] = \
df_test['Year-Of-Publication'].value_counts().index[0]
df_test['Year-Of-Publication'] = df_test['Year-Of-Publication'].astype(int)

test_set = pd.merge(df_test, index_ISBN, how='left', on=None, left_on='bookid', right_on='ISBN', left_index=True)
test_set = test_set[['userid', 'code', 'Year-Of-Publication', 'Age']]
test_set.columns = ['userid', 'bookid', 'Year-Of-Publication', 'Age']
test_set['bookid'] = test_set['bookid'].fillna(test_set['bookid'].median()).astype(int)
test_set.index = np.arange(0, 206235)

# 获取填补值 和 原有特征
fillc = test_set['Age']
ytest = fillc[fillc.isnull()]
xtest = feature.iloc[ytest.index, :]

ypredict = rfc.predict(xtest)
# 填补缺失值
test_set.iloc[ytest.index, 3] = ypredict
test_set = test_set.astype(int)

# 预测测试集
result = clf.predict(test_set)
save = pd.DataFrame({'id': np.arange(0, 206235), 'acc': result.ravel()})
# 查看 评分 状态
print(save['acc'].value_counts())
# 保存
save.to_csv(r'reslut02.csv', index=False, header=False)
