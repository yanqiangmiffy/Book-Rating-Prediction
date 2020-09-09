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

stop_words = stopwords.words('english')
tqdm.pandas()


def emb(df, f1, f2, emb_size=10):
    print("==================={}_{}==================".format(f1, f2))
    tmp = df.groupby(f1, as_index=False)[f2].agg({'{}_{}_list'.format(f1, f2): list})
    sentences = tmp['{}_{}_list'.format(f1, f2)].values.tolist()
    print(sentences)
    for i in range(len(sentences)):
        sentences[i] = [str(x) for x in sentences[i]]
    model = Word2Vec(sentences, size=emb_size, window=5, min_count=1, sg=0, hs=1, seed=2020)
    emb_matrix = []
    for seq in sentences:
        vec = []
        for w in seq:
            if w in model:
                vec.append(model[w])
        if len(vec) > 0:
            emb_matrix.append(np.mean(vec, axis=0))
        else:
            emb_matrix.append([0] * emb_size)
    emb_matrix = np.array(emb_matrix)
    for i in range(emb_size):
        tmp['{}_{}_emb_{}'.format(f1, f2, i)] = emb_matrix[:, i]
    del model, emb_matrix, sentences
    return tmp


# 添加词向量特征
def w2v_feat(data_frame, feat, mode):
    print(f'Start {mode} word2vec ...')
    for i in feat:
        if data_frame[i].dtype != 'object':
            data_frame[i] = data_frame[i].astype(str)
    # data_frame.fillna('nan', inplace=True)
    data_frame = data_frame.replace([np.inf, -np.inf], np.nan).fillna(0)
    model = Word2Vec(data_frame[feat].values.tolist(), size=10, window=2, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=10)
    stat_list = ['min', 'max', 'mean', 'std']
    new_all = pd.DataFrame()
    for m, t in enumerate(feat):
        print(f'Start gen feat of {t} ...')
        tmp = []
        for i in data_frame[t].unique():
            tmp_v = [i]
            tmp_v.extend(model[i])
            tmp.append(tmp_v)
        tmp_df = pd.DataFrame(tmp)
        w2c_list = [f'w2c_{t}_{n}' for n in range(10)]
        tmp_df.columns = [t] + w2c_list
        tmp_df = data_frame[['ISBN', t]].merge(tmp_df,on=t)
        tmp_df = tmp_df.drop_duplicates().groupby('ISBN').agg(stat_list).reset_index()
        tmp_df.columns = ['ISBN'] + [f'{p}_{q}' for p in w2c_list for q in stat_list]
        if m == 0:
            new_all = pd.concat([new_all, tmp_df], axis=1)
        else:
            new_all = pd.merge(new_all, tmp_df, how='left', on='ISBN')
    return new_all


book = pd.read_csv('data/book.csv', encoding='ISO-8859-1')

# 预处理
# 年份处理
book.loc[book['ISBN'] == '0330482750', 'Year-Of-Publication'] = 2002
book['Year-Of-Publication'] = book['Year-Of-Publication'].astype(int)
book.loc[book['Year-Of-Publication'] == 2050, 'Year-Of-Publication'] = \
    book['Year-Of-Publication'].value_counts().index[0]
# book.loc[book['Year-Of-Publication'] == 0, 'Year-Of-Publication'] = \
#     book['Year-Of-Publication'].value_counts().index[0]

book.loc[book['ISBN'] == '0330482750', 'Book-Author'] = 'Amit Chaudhuri'
book.loc[book['ISBN'] == '0330482750', 'Publisher'] = 'Vintage Books USA'
book['Publisher'].fillna(value=book.Publisher.mode().values[0], inplace=True)

user = pd.read_csv('data/user.csv', encoding='ISO-8859-1')

train = pd.read_csv('data/train.csv', )
test = pd.read_csv('data/test.csv')

train = pd.merge(train, user, how='left', on='User-ID')
train = pd.merge(train, book, how='left', on='ISBN')

test = pd.merge(test, user, how='left', on='User-ID')
test = pd.merge(test, book, how='left', on='ISBN')
train_size = len(train)


# 评价低分特征
select_index=(train['Book-Rating']<=4)&(train['Book-Rating']!=0)
low_ratings_years=train[select_index]['Year-Of-Publication'].unique()
low_ratings_authors=train[select_index]['Book-Author'].unique()
low_ratings_publishers=train[select_index]['Publisher'].unique()
low_ratings_users=train[select_index]['User-ID'].unique()
low_ratings_books=train[select_index]['ISBN'].unique()

select_index=train['Book-Rating']>4
high_ratings_years=train[select_index]['Year-Of-Publication'].unique()
high_ratings_authors=train[select_index]['Book-Author'].unique()
high_ratings_publishers=train[select_index]['Publisher'].unique()
high_ratings_users=train[select_index]['User-ID'].unique()
high_ratings_books=train[select_index]['ISBN'].unique()

unique_authors=set(low_ratings_authors)-set(high_ratings_years)
unique_publishers=set(low_ratings_publishers)-set(high_ratings_publishers)
unique_users=set(low_ratings_users)-set(high_ratings_users)
unique_books=set(low_ratings_books)-set(high_ratings_books)


def label(x):
    if x==0:
        return 0
    elif 1<=x<=4:
        return 1
    elif 5<=x<=7:
        return 2
    else:
        return 3
train['Book-Rating']=train['Book-Rating'].astype(int)
train['Book-Rating']=train['Book-Rating'].map(label)

update = True
if update:
    data = pd.concat([train, test], axis=0).reset_index(drop=True)

    data['low_ratings_year']=data['Year-Of-Publication'].isin(low_ratings_years).astype(int)
    data['low_ratings_author']=data['Book-Author'].isin(low_ratings_authors).astype(int)
    data['low_ratings_publisher']=data['Publisher'].isin(low_ratings_publishers).astype(int)
    data['low_ratings_users']=data['User-ID'].isin(low_ratings_publishers).astype(int)
    data['low_ratings_books']=data['ISBN'].isin(low_ratings_publishers).astype(int)

    data['high_ratings_year'] = data['Year-Of-Publication'].isin(high_ratings_years).astype(int)
    data['high_ratings_authors'] = data['Book-Author'].isin(high_ratings_authors).astype(int)
    data['high_ratings_publisher'] = data['Publisher'].isin(high_ratings_publishers).astype(int)

    data['unique_authors'] = data['Book-Author'].isin(unique_authors).astype(int)
    data['unique_publishers'] = data['Publisher'].isin(unique_publishers).astype(int)
    data['unique_users'] = data['User-ID'].isin(unique_users).astype(int)
    data['unique_books'] = data['ISBN'].isin(unique_books).astype(int)


    def get_locations(row):
        """
        用户 location 处理
        :param row:
        :return:
        """
        x = row['Location']
        locations = x.split(',')
        # loc1, loc2, loc3 = '', '', ''
        if len(locations) == 3:
            loc1 = locations[2]
            loc2 = locations[1]
            loc3 = locations[0]
        elif len(locations) == 2:
            loc1 = locations[1]
            loc2 = locations[0]
            loc3 = locations[0]
        elif len(locations) == 1:
            loc1 = locations[0]
            loc2 = locations[0]
            loc3 = locations[0]
        else:
            loc1, loc2, loc3 = locations[2], locations[1], locations[0]
        return loc1, loc2, loc3


    #data[['country', 'state', 'city']] = data.progress_apply(lambda x: get_locations(x), axis=1,
    #                                                          result_type="expand")
    lb = LabelEncoder()

    #data['country'] = lb.fit_transform(data['country'])
    #data['state'] = lb.fit_transform(data['state'])
    #data['city'] = lb.fit_transform(data['city'])
    # del data['state'],data['city']

    data['Location'] = lb.fit_transform(data['Location'])
    data['Location_count'] = data.groupby('Location')['ISBN'].transform('count')
    #data['country_count'] = data.groupby('country')['ISBN'].transform('count')
    #data['state_count'] = data.groupby('state')['ISBN'].transform('count')
    #data['city_count'] = data.groupby('city')['ISBN'].transform('count')

    # 出版商 Publisher特征
    data['Publisher'] = data['Publisher'].fillna(value='None')
    # data['Publisher_count'] = data.groupby('Publisher')['ISBN'].transform('count')
    data['Publisher'] = lb.fit_transform(data['Publisher'].astype(str))

    # 年份 Year-Of-Publication
    data['Year-Of-Publication_count'] = data.groupby('Year-Of-Publication')['ISBN'].transform('count')
    # 作者 Book-Author
    data['Book-Author'] = data['Book-Author'].fillna(value='None')
    # data['Book-Author_count'] = data.groupby('Book-Author')['ISBN'].transform('count')
    data['Book-Author'] = lb.fit_transform(data['Book-Author'])

    # ISBN 特征
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


    data[['ISBN1', 'ISBN2', 'ISBN3', 'ISBN4']] = data.progress_apply(lambda x: get_ISBN(x), axis=1,
                                                                     result_type="expand")

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
    # data['User-ID_count'] = data.groupby('User-ID')['ISBN'].transform('count')
    data['User-ID'] = lb.fit_transform(data['User-ID'])

    # 年龄特征
    # data['Age']=data['Age'].fillna(value=user['Age'].mean())
    print("年龄特征")
    no_fea = ['Book-Title', 'Book-Rating']
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

    print(data.loc[ytest.index, 'Age'])
    # 标题Tfidf特征 Book-Title
    data['Book-Title'] = data['Book-Title'].str.lower()
    data['Book-Title_len'] = data['Book-Title'].map(len)
    data['Book-Title_word_len'] = data['Book-Title'].apply(lambda x: len(x.split()))
    data['Book-Title'] = data['Book-Title'].apply(lambda x: " ".join([w for w in x.split() if w not in stop_words]))

    print('文本聚类')
    import os

    modelname = 'data/tfidf.pkl'
    if not os.path.exists(modelname):
        v = TfidfVectorizer(ngram_range=(1, 2))
        v.fit(data['Book-Title'])
        print(len(v.get_feature_names()))
        with open('data/tfidf.pkl', 'wb') as f:
            pickle.dump(v, f)
    else:
        with open('data/tfidf.pkl', 'rb') as f:
            v = pickle.load(f)
    df_tfidf = v.transform(data['Book-Title'])

    kmname = 'data/km1.pkl'
    if not os.path.exists(kmname):
        kmeans = KMeans(init='k-means++', n_clusters=6, n_init=6, n_jobs=-1)
        pred_classes = kmeans.fit_predict(df_tfidf)
        data['title_label'] = pred_classes
        print(data['title_label'].value_counts())
        with open(kmname, 'wb') as f:
            pickle.dump(kmeans, f)
    else:
        with open(kmname, 'rb') as f:
            kmeans = pickle.load(f)
            pred_classes = kmeans.predict(df_tfidf)
            data['title_label'] = pred_classes

    kmname = 'data/km2.pkl'
    user_features = ['ISBN', 'Age', 'Year-Of-Publication', 'Year-Of-Publication_count',
                     'ISBN1', 'ISBN2', 'ISBN3', 'ISBN4'
                     ]
    if not os.path.exists(kmname):
        print("用户聚类")

        kmeans = KMeans(init='k-means++', n_clusters=6, n_init=6, n_jobs=-1)
        pred_classes = kmeans.fit_predict(data[user_features])
        data['user_label'] = pred_classes
        print(data['user_label'].value_counts())
        with open(kmname, 'wb') as f:
            pickle.dump(kmeans, f)
    else:
        with open(kmname, 'rb') as f:
            kmeans = pickle.load(f)
            pred_classes = kmeans.predict(data[user_features])
            data['user_label'] = pred_classes

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

    #  id embedding
    # print("embedding")
    # tmp1 = emb(data, 'User-ID', 'ISBN')
    # tmp2 = emb(data, 'User-ID', 'Location')
    # tmp3 = emb(data, 'User-ID', 'Age')
    # tmp4 = emb(data, 'User-ID', 'Book-Author')
    # tmp5 = emb(data, 'User-ID', 'Year-Of-Publication')
    # tmp6 = emb(data, 'User-ID', 'Publisher')
    # data = pd.concat([data, tmp1.iloc[:, 2:]], axis=1)
    # data = pd.concat([data, tmp2.iloc[:, 2:]], axis=1)
    # data = pd.concat([data, tmp3.iloc[:, 2:]], axis=1)
    # data = pd.concat([data, tmp4.iloc[:, 2:]], axis=1)
    # data = pd.concat([data, tmp5.iloc[:, 2:]], axis=1)
    # data = pd.concat([data, tmp6.iloc[:, 2:]], axis=1)
    #data=w2v_feat(data,['User-ID','Location','Age','Book-Author','Year-Of-Publication','Publisher'],'data')
    #del data['Book-Title']
    cate_feas=['User-ID','Location','Age','Book-Author','Year-Of-Publication','Publisher','ISBN']
    for fea in cate_feas:
        data['{}_nunique'.format(fea)]=data.groupby(by='User-ID')[fea].transform('nunique')
    data.to_csv('data/df.csv', index=False)
else:
    data = pd.read_csv('data/df.csv')
    print(data.isnull().sum())

no_fea = ['Book-Title', 'id',
          'Book-Rating', 'ID', 'fold',
          'Book-Title', 'Book-Author', 'Publisher', 'Location',
          'User-ID']
features = [fea for fea in data.columns if fea not in no_fea]
data['Age'] = data['Age'].astype(int)
print(features, len(features))
train = data[:train_size]
test = data[train_size:]

del data


def load_data():
    return train, train['Book-Rating'], test, features
