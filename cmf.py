import numpy as np
import pandas as pd
from cmfrec import CMF_explicit
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import math

now_phase = 4
train_path = 'underexpose_train'
test_path = 'underexpose_test'
recom_item = []

whole_click = pd.DataFrame()
whole_test = pd.DataFrame()
for c in range(now_phase + 1):
    print('phase:', c)
    click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(
        c), header=None,  names=['UserId', 'ItemId', 'Rating'])
    click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(
        c), header=None,  names=['UserId', 'ItemId', 'Rating'])

    all_click = click_train.append(click_test)
    whole_click = whole_click.append(all_click)
    whole_test = whole_test.append(click_test)

# whole_click.insert(3, column='Rating', value=1)
whole_click = whole_click.drop_duplicates(subset=['UserId', 'ItemId'], keep='first')


user_info = pd.read_csv(train_path + '/underexpose_user_feat.csv'.format(c), header=None,
                        names=['UserId', 'user_age_level', 'user_gender', 'user_city_level'])
user_info = user_info.join(pd.get_dummies(user_info.user_gender))
user_info = user_info.drop(columns=['user_gender'])
print(user_info)

item_info = pd.read_csv(
    train_path + '/underexpose_item_feat.csv'.format(c), header=None)
item_info.columns = ['ItemId'] + list(item_info.columns[1:])
item_info[1] = item_info[1].str.replace('[', '').astype('float64')
item_info[128] = item_info[128].str.replace(']', '').astype('float64')
item_info[129] = item_info[129].str.replace('[', '').astype('float64')
item_info[256] = item_info[256].str.replace(']', '').astype('float64')
print(item_info)

# Fit the model
model = CMF_explicit(method="als")
model.fit(X=whole_click, U=user_info, I=item_info)

# Top-5 highest predicted for user 3
test_id = whole_test['UserId'].unique()
result = pd.DataFrame()
for i in range(test_id.shape[0]):
    result = result.append(
        pd.Series(model.topN(user=i, n=50)), ignore_index=True)

result.insert(0, column='id', value=test_id)

for i in range(result.shape[1]-1):
    result[i] = result[i].astype('int')

result.to_csv('baseline1.csv', index=False, header=None)
