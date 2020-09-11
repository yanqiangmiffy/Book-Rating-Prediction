# -*- coding: utf-8 -*-
"""
@author: quincyqiang
@software: PyCharm
@file: gen_feas.py
@time: 2020/9/2 23:36
@description：
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
from utils import label, label_inverse
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
import os

stop_words = stopwords.words('english')
tqdm.pandas()


def load_data(num_classes):
    # ===================== 数据读取 ================
    book = pd.read_csv('data/book.csv', encoding='ISO-8859-1')
    user = pd.read_csv('data/user.csv', encoding='ISO-8859-1')
    train = pd.read_csv('data/train.csv')
    train['id']=['train_{}'.format(i) for i in range(train.shape[0])]
    test = pd.read_csv('data/test.csv')

    # ===================== 数据预处理 ================
    # 预处理
    # 年份处理
    book.loc[book['ISBN'] == '0330482750', 'Year-Of-Publication'] = 2002
    book['Year-Of-Publication'] = book['Year-Of-Publication'].astype(int)
    book.loc[book['Year-Of-Publication'] > 2020, 'Year-Of-Publication'] = \
        book['Year-Of-Publication'].value_counts().index[0]
    #book.loc[book['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = \
    #    book['Year-Of-Publication'].value_counts().index[0]
    #book['year_lag'] = 2020 - book['Year-Of-Publication']  # 数据集发表年份距今日期

    # 作者处理
    book.loc[book['ISBN'] == '0330482750', 'Book-Author'] = 'Amit Chaudhuri'
    book.loc[book['ISBN'] == '0330482750', 'Publisher'] = 'Vintage Books USA'
    book['Publisher'].fillna(value=book.Publisher.mode().values[0], inplace=True)

    train = pd.merge(train, user, how='left', on='User-ID')
    train = pd.merge(train, book, how='left', on='ISBN')
    test = pd.merge(test, user, how='left', on='User-ID')
    test = pd.merge(test, book, how='left', on='ISBN')
    train_size = len(train)

    # =================== 评价低分特征 ==================
    select_index = (train['Book-Rating'] <= 4) & (train['Book-Rating'] != 0)
    low_ratings_years = train[select_index]['Year-Of-Publication'].unique()
    low_ratings_authors = train[select_index]['Book-Author'].unique()
    low_ratings_publishers = train[select_index]['Publisher'].unique()
    low_ratings_users = train[select_index]['User-ID'].unique()
    low_ratings_books = train[select_index]['ISBN'].unique()

    select_index = train['Book-Rating'] >= 7
    high_ratings_years = train[select_index]['Year-Of-Publication'].unique()
    high_ratings_authors = train[select_index]['Book-Author'].unique()
    high_ratings_publishers = train[select_index]['Publisher'].unique()
    high_ratings_users = train[select_index]['User-ID'].unique()
    high_ratings_books = train[select_index]['ISBN'].unique()

    unique_authors = set(low_ratings_authors) - set(high_ratings_years)
    unique_publishers = set(low_ratings_publishers) - set(high_ratings_publishers)
    unique_users = set(low_ratings_users) - set(high_ratings_users)
    unique_books = set(low_ratings_books) - set(high_ratings_books)

    train['Book-Rating'] = train['Book-Rating'].astype(int)
    train['Book-Rating'] = train['Book-Rating'].apply(lambda x: label(x, num_classes))
    data = pd.concat([train, test], axis=0).reset_index(drop=True)

    data['low_ratings_year'] = data['Year-Of-Publication'].isin(low_ratings_years).astype(int)
    data['low_ratings_author'] = data['Book-Author'].isin(low_ratings_authors).astype(int)
    data['low_ratings_publisher'] = data['Publisher'].isin(low_ratings_publishers).astype(int)
    data['low_ratings_users'] = data['User-ID'].isin(low_ratings_publishers).astype(int)
    data['low_ratings_books'] = data['ISBN'].isin(low_ratings_publishers).astype(int)

    data['high_ratings_year'] = data['Year-Of-Publication'].isin(high_ratings_years).astype(int)
    data['high_ratings_authors'] = data['Book-Author'].isin(high_ratings_authors).astype(int)
    data['high_ratings_publisher'] = data['Publisher'].isin(high_ratings_publishers).astype(int)

    data['unique_authors'] = data['Book-Author'].isin(unique_authors).astype(int)
    data['unique_publishers'] = data['Publisher'].isin(unique_publishers).astype(int)
    data['unique_users'] = data['User-ID'].isin(unique_users).astype(int)
    data['unique_books'] = data['ISBN'].isin(unique_books).astype(int)

    #  ===================== 位置特征 ==========================
    def get_locations(row):
        """
        用户 location 处理
        :param row:
        :return:
        """
        x = row['Location']
        locations = x.split(',')
        country, state, city = '', '', ''
        if len(locations) == 3:
            country = locations[2]
            if len(country) == 0:  # 例如： weston, ,，将国家设置成美国
                country = 'usa'
            state = locations[1]
            if len(state) == 0:
                state = locations[0]
            city = locations[0]
        elif len(locations) == 4:
            # eg：ray, michigan usa, ,
            # eg:ivanhoe, melbourne, , australia
            country = locations[3]  #
            if len(country) == 0:  # 例如： weston, ,，将国家设置成美国
                country = 'usa'
            state = locations[1]
            if len(state) == 0:
                state = locations[2]
            city = locations[0]
        elif len(locations) == 2:
            country = locations[1]
            state = locations[0]
            city = locations[0]
        elif len(locations) == 1:
            country = locations[0]
            state = locations[0]
            city = locations[0]
        else:  # 大于4
            country = locations[-1]
            if len(country) == 0:
                country = 'usa'
            state = locations[1]
            if len(state) == 0:
                state = locations[2]
            city = locations[0]
        return country, state, city

    if not os.path.exists('data/locations.csv'):
        data[['country', 'state', 'city']] = data.progress_apply(lambda x: get_locations(x), axis=1,
                                                                 result_type="expand")
        data[['country', 'state', 'city']].to_csv('data/locations.csv', index=None)
    else:
        location_df = pd.read_csv('data/locations.csv')
        print(location_df.isnull().sum())
        data = pd.concat([data, location_df], axis=1)
    lb = LabelEncoder()

    data['country'] = lb.fit_transform(data['country'])
    data['state'] = lb.fit_transform(data['state'])
    del data['city']
    # data['city'] = lb.fit_transform(data['city'])

    data['Location'] = lb.fit_transform(data['Location'])
    data['Location_count'] = data.groupby('Location')['ISBN'].transform('count')
    # data['country_count'] = data.groupby('country')['ISBN'].transform('count')
    # data['state_count'] = data.groupby('state')['ISBN'].transform('count')
    # data['city_count'] = data.groupby('city')['ISBN'].transform('count')

    # ===================== 出版商 Publisher特征 ===================
    data['Publisher'] = data['Publisher'].fillna(value='None')
    # data['Publisher_count'] = data.groupby('Publisher')['ISBN'].transform('count')
    data['Publisher'] = lb.fit_transform(data['Publisher'].astype(str))

    # 年份 Year-Of-Publication
    data['Year-Of-Publication_count'] = data.groupby('Year-Of-Publication')['ISBN'].transform('count')
    # 作者 Book-Author
    data['Book-Author'] = data['Book-Author'].fillna(value='None')
    # data['Book-Author_count'] = data.groupby('Book-Author')['ISBN'].transform('count')
    data['Book-Author'] = lb.fit_transform(data['Book-Author'])

    # ======================= ISBN 特征 ===========================
    # https://zhuanlan.zhihu.com/p/96141798
    # 3-16-148410-0  特定书籍的ISBN：978-3-16-148410-0
    data['ISBN'] = data['ISBN'].astype(str)

    def get_ISBN(row):
        x = row['ISBN']
        if len(x) > 10:
            x = x[-10:]
        if len(x) == 10:
            ISBN1, ISBN2, ISBN3, ISBN4 = x[0], x[1:4], x[4:9], x[9]
        else:
            ISBN1, ISBN2, ISBN3, ISBN4 = x[0:3], x[4:7], x[7:10], x[10:12]
        return ISBN1, ISBN2, ISBN3, ISBN4

    if not os.path.exists('data/isbn.csv'):
        data[['ISBN1', 'ISBN2', 'ISBN3', 'ISBN4']] = data.progress_apply(lambda x: get_ISBN(x), axis=1,
                                                                         result_type="expand")
        data[['ISBN1', 'ISBN2', 'ISBN3', 'ISBN4']].to_csv('data/isbn.csv', index=None)
    else:
        isbn_df = pd.read_csv('data/isbn.csv').astype(str)
        data = pd.concat([data, isbn_df], axis=1)

    data['ISBN1'] = lb.fit_transform(data['ISBN1'])
    data['ISBN2'] = lb.fit_transform(data['ISBN2'])
    data['ISBN3'] = lb.fit_transform(data['ISBN3'])
    data['ISBN4'] = lb.fit_transform(data['ISBN4'])
    data['ISBN'] = lb.fit_transform(data['ISBN'])
    data['ISBN_count'] = data.groupby('ISBN')['User-ID'].transform('count')
    data['ISBN1_count'] = data.groupby('ISBN1')['User-ID'].transform('count')
    data['ISBN2_count'] = data.groupby('ISBN2')['User-ID'].transform('count')
    data['ISBN3_count'] = data.groupby('ISBN3')['User-ID'].transform('count')
    data['ISBN4_count'] = data.groupby('ISBN4')['User-ID'].transform('count')

    # 用户特征
    data['User-ID_count'] = data.groupby('User-ID')['ISBN'].transform('count')
    # data['User-ID'] = lb.fit_transform(data['User-ID'])

    # 年龄特征
    # data['Age']=data['Age'].fillna(value=user['Age'].mean())
    print("年龄特征")
    features = ['User-ID', 'ISBN', 'Year-Of-Publication']
    fillc = data.loc[:, 'Age']
    feature = data.loc[:, features]
    # 构成训练集和测试集
    ytrain = fillc[fillc.notnull()]
    # print(ytrain)
    ytest = fillc[fillc.isnull()]
    xtrain = feature.iloc[ytrain.index, :]
    xtest = feature.iloc[ytest.index, :]
    # 建模 预测年龄
    rfc = RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(xtrain, ytrain)
    ypredict = rfc.predict(xtest)
    # 填补缺失值
    data.loc[ytest.index, 'Age'] = ypredict
    data['Age'] = data['Age'].astype(int)
    print(data.loc[ytest.index, 'Age'].value_counts())

    # =================   Book-Title ===================
    # 标题Tfidf特征 Book-Title
    data['Book-Title'] = data['Book-Title'].str.lower()
    data['Book-Title_len'] = data['Book-Title'].map(len)
    data['Book-Title_word_len'] = data['Book-Title'].apply(lambda x: len(x.split()))
    data['Book-Title'] = data['Book-Title'].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))
   

    # # ==============聚类特征==================
    # # 基于tfidf的文本聚类
    # print('文本聚类')
    # modelname = 'data/tfidf.pkl'
    # if not os.path.exists(modelname):
    #     v = TfidfVectorizer(ngram_range=(1, 2))
    #     v.fit(data['Book-Title'])
    #     print(len(v.get_feature_names()))
    #     with open('data/tfidf.pkl', 'wb') as f:
    #         pickle.dump(v, f)
    # else:
    #     with open('data/tfidf.pkl', 'rb') as f:
    #         v = pickle.load(f)
    # df_tfidf = v.transform(data['Book-Title'])
    # kmname = 'data/km1.pkl'
    # if not os.path.exists(kmname):
    #     kmeans = KMeans(init='k-means++', n_clusters=6, n_init=6, n_jobs=-1)
    #     pred_classes = kmeans.fit_predict(df_tfidf)
    #     data['title_label'] = pred_classes
    #     print(data['title_label'].value_counts())
    #     with open(kmname, 'wb') as f:
    #         pickle.dump(kmeans, f)
    # else:
    #     with open(kmname, 'rb') as f:
    #         kmeans = pickle.load(f)
    #         pred_classes = kmeans.predict(df_tfidf)
    #         data['title_label'] = pred_classes

    # kmname = 'data/km2.pkl'
    # user_features = ['ISBN', 'Age', 'Year-Of-Publication', 'Year-Of-Publication_count',
    #                  'ISBN1', 'ISBN2', 'ISBN3', 'ISBN4'
    #                  ]
    # if not os.path.exists(kmname):
    #     print("用户聚类")
    #     kmeans = KMeans(init='k-means++', n_clusters=6, n_init=6, n_jobs=-1)
    #     pred_classes = kmeans.fit_predict(data[user_features])
    #     data['user_label'] = pred_classes
    #     print(data['user_label'].value_counts())
    #     with open(kmname, 'wb') as f:
    #         pickle.dump(kmeans, f)
    # else:
    #     with open(kmname, 'rb') as f:
    #         kmeans = pickle.load(f)
    #         pred_classes = kmeans.predict(data[user_features])
    #         data['user_label'] = pred_classes

    # # 基于doc2vec的文本聚类
    # if not os.path.exists('models/doc2vec.vec'):
    #     texts = book['Book-Title'].values.tolist()
    #     texts = [text.split() for text in texts]
    #     print(texts[:2])
    #     documents = [TaggedDocument(doc, [i]) for i, doc in tqdm(enumerate(texts))]
    #     d2v_model = Doc2Vec(documents, vector_size=200, window=3, min_count=1, workers=10)
    #     d2v_model.save("models/doc2vec.vec")
    #     d2v_model = Doc2Vec.load("models/doc2vec.vec")  # you can continue training with the loaded model!
    #     print(d2v_model.infer_vector(['Classical Mythology']))
    # else:
    #     d2v_model = Doc2Vec.load('models/doc2vec.vec')
    # if not os.path.exists('data/doc2vec.pkl'):
    #     print("正在加载文档向量")
    #     texts = data['Book-Title'].values.tolist()
    #     vecs = []
    #     for text in tqdm(texts):
    #         vec = d2v_model.infer_vector(text.split())
    #         vecs.append(vec)
    #     print(np.array(vecs).shape)
    #     with open('data/doc2vec.pkl', 'wb') as f:
    #         pickle.dump(vecs, f)
    # else:
    #     with open('data/doc2vec.pkl', 'rb') as f:
    #         vecs = pickle.load(f)

    # kmname = 'data/km3.pkl'
    # if not os.path.exists(kmname):
    #     print("基于doc2vec的标题聚类")
    #     kmeans = KMeans(init='k-means++', n_clusters=15, n_init=15, n_jobs=-1)
    #     pred_classes = kmeans.fit_predict(np.array(vecs))
    #     data['doc2vec_label'] = pred_classes
    #     print(data['doc2vec_label'].value_counts())
    #     with open(kmname, 'wb') as f:
    #         pickle.dump(kmeans, f)
    # else:
    #     with open(kmname, 'rb') as f:
    #         kmeans = pickle.load(f)
    #         pred_classes = kmeans.predict(np.array(vecs))
    #         data['doc2vec_label'] = pred_classes

    # 添加转化率特征
    cat_list = ['ISBN', 'User-ID', 'Book-Author', 'Publisher', 'Year-Of-Publication']
    data['ID'] = data.index
    data['fold'] = data['ID'] % 5
    data.loc[data['Book-Rating'].isnull(), 'fold'] = 5
    target_feat = []
    for i in tqdm(cat_list):
        target_feat.extend([i + '_mean_last_1'])
        data[i + '_mean_last_1'] = None
        for fold in range(6):
            data.loc[data['fold'] == fold, i + '_mean_last_1'] = data[data['fold'] == fold][i].map(
                data[(data['fold'] != fold) & (data['fold'] != 5)].groupby(i)['Book-Rating'].mean()
            )
        data[i + '_mean_last_1'] = data[i + '_mean_last_1'].astype(float)

    del data['Book-Title']
    cate_feas = ['User-ID', 'Location', 'Age', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'ISBN']
    for fea1 in cate_feas:
        for fea2 in cate_feas:
            if fea1!=fea2:
                data['{}_{}_nunique'.format(fea1,fea2)] = data.groupby(by=fea1)[fea2].transform('nunique')

    no_fea = ['Book-Title', 'id',
              'Book-Rating', 'ID', 'fold',
              'Book-Title', 'Book-Author', 'Publisher', 'Location',
              'User-ID']
    features = [fea for fea in data.columns if fea not in no_fea]
    data['Age'] = data['Age'].astype(int)
    train = data[:train_size]
    test = data[train_size:]

    del data
    # features=['User-ID', 'ISBN', 'Year-Of-Publication', 'Age']
    print(features, len(features))

    return train, train['Book-Rating'], test, features
