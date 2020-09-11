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

pred1=pd.read_csv('result/lgb_numclasses3_0.7195052410657701.csv',header=None)
pred1.columns=['id','score']
pred2=pd.read_csv('result/cat_numclasses3_0.7294094916964116.csv',header=None)
pred2.columns=['id','score']
pred=pd.DataFrame(pred1['id'])
# pred['score']=((pred1['score']+pred2['score'])/2) # 76.22
pred['score']=pred1['score']*0.7+pred2['score']*0.3
print(pred['score'].value_counts())
pred[['id', 'score']].to_csv('result/en.csv', index=None, header=None)
