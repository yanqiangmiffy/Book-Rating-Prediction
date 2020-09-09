import pandas as pd
from tqdm import tqdm
tqdm.pandas()
user=pd.read_csv('../data/user.csv',encoding='latin-1')
train=pd.read_csv('../data/train.csv')
print(train['Book-Rating'].value_counts())
print(user.head())
print(user.shape)

print(user['Location'].value_counts())

def get_locations(row):
        """
        用户 location 处理
        :param row:
        :return:
        """
        x = row['Location']
        locations = x.split(',')
        loc1, loc2, loc3 = '', '', ''
        if len(locations) >= 3:
            loc1 = locations[2]
            loc2 = locations[1]
            loc3 = locations[0]
        if len(locations) == 2:
            loc1 = locations[1]
            loc2 = locations[0]
            loc3 = locations[0]
        if len(locations) == 1:
            loc1 = locations[0]
            loc2 = locations[0]
            loc3 = locations[0]
        return loc1, loc2, loc3

user['locations_len']=user['Location'].apply(lambda x:len(x.split(',')))
print(user['locations_len'].value_counts())
user[['country', 'state', 'city']] = user.progress_apply(lambda x: get_locations(x), axis=1,
                                                             result_type="expand")

print(user['country'].value_counts())
print(user['state'].value_counts())

