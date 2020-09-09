#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:quincyqiang 
@license: Apache Licence 
@file: utils.py 
@time: 2020/9/10 2:07 上午
@description:
"""
import pickle
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from sklearn.ensemble import RandomForestRegressor

stop_words = stopwords.words('english')
tqdm.pandas()


def label(x, num_classes):
    if num_classes == 3:
        if x == 0:
            return 0
        elif 1 <= x <= 6:
            return 1
        else:
            return 2
    elif num_classes == 4:
        if x == 0:
            return 0
        elif 1 <= x <= 4:
            return 1
        elif 5 <= x <= 7:
            return 2
        else:
            return 3
    elif num_classes == 5:
        if x == 0:
            return 0
        elif 1 <= x <= 4:
            return 1
        elif 5 <= x <= 6:
            return 2
        elif 7 <= x <= 8:
            return 3
        else:
            return 4
    elif num_classes == 11:
        return x


def label_inverse(x, num_classes):
    if num_classes == 3:
        if x == 0:
            return 0
        elif x == 1:  # 1 5
            return 3.5
        else:
            return 7  # 6 10
    elif num_classes == 4:
        if x == 0:
            return 0
        elif x == 1:  # 1  4
            return 2.5
        elif x == 2:  # 5 7
            return 6
        else:  # 8  10
            return 9
    elif num_classes == 5:
        if x == 0:
            return 0
        elif x == 1:  # 1 <= x <= 4:
            return 2.5
        elif x == 2:  # 5 <= x <= 6:
            return 5.5
        elif x == 3:  # 7 8
            return 7.5
        else:  # 9  10
            return 9.5
    elif num_classes == 11:
        return x
