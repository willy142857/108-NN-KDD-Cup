import argparse
import math
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation import evaluate


item_feat_dict = pickle.load(open('underexpose_train/item_feat.pkl', 'rb'))
item_feat_set = set(item_feat_dict.keys())
item_img_feat_dict = pickle.load(open('underexpose_train/item_img_feat.pkl', 'rb'))


def get_sim_item(df, user_col, item_col, phase, use_iif=False):  
    user_item_ = df.groupby(user_col)[item_col].agg(list).reset_index()  
    user_item_dict = dict(zip(user_item_[user_col], user_item_[item_col]))  
    
    item_user_ = df.groupby(item_col)[user_col].agg(set).reset_index()  
    item_user_dict = dict(zip(item_user_[item_col], item_user_[user_col]))    

    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    sim_item = {}

    for item, users in tqdm(item_user_dict.items(), desc='calc sim'):
        sim_item.setdefault(item, {}) 

        for u in users:
            loc1 = user_item_dict[u].index(item)
            for loc2, relate_item in enumerate(user_item_dict[u]):
                if item == relate_item:
                    continue
                sim_item[item].setdefault(relate_item, 0)
                
                t1 = user_time_dict[u][loc1]
                t2 = user_time_dict[u][loc2]
                t12_diff = abs(t1 - t2)
                loc12_diff = abs(loc1 - loc2)
                time_weight = ((loc12_diff)**-0.5) * -math.log(t12_diff + 1e-6)

                if item in item_feat_set and relate_item in item_feat_set:
                    sim_item[item][relate_item] += 0.007*np.inner(item_feat_dict[item], item_feat_dict[relate_item]) 
                    sim_item[item][relate_item] += 0.01*np.inner(item_img_feat_dict[item], item_img_feat_dict[relate_item])
                    
                # sim(i, j) = P(j | i) = (Ui & Uj) / Ui
                # ref: https://bit.ly/3dyDuch
                ui = set(users)
                uj = set(item_user_dict[relate_item])
                confidence = len(ui.intersection(uj)) / len(ui)

                if loc2 > loc1:
                    # 正向
                    weight = 1.05 * time_weight
                    sim_item[item][relate_item] += confidence + weight
                else:
                    # 逆向
                    weight = 1.0 * time_weight
                    sim_item[item][relate_item] += confidence + weight
                

    item_cnt = df[item_col].value_counts().to_dict()
    for i, related_items in tqdm(sim_item.items(), desc='normalize'):
        for j, cij in related_items.items():
            sim_item[i][j] = cij / (math.log(item_cnt[j]) * math.log(item_cnt[i]) + 1)

    return sim_item, user_item_dict 


def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda d: d[1], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * ((loc+1)**-0.7)

    return sorted(rank.items(), key=lambda d: d[1], reverse=True)[:item_num]


# fill user to 50 items
def get_predict(df, pred_col, top_fill):
    top_fill = [int(t) for t in top_fill.split(',')]
    scores = [-1 * i for i in range(1, len(top_fill) + 1)]
    ids = list(df['user_id'].unique())
    fill_df = pd.DataFrame(ids * len(top_fill), columns=['user_id'])
    fill_df.sort_values('user_id', inplace=True)
    fill_df['item_id'] = top_fill * len(ids)
    fill_df[pred_col] = scores * len(ids)
    df = df.append(fill_df)
    df.sort_values(pred_col, ascending=False, inplace=True)
    df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
    df['rank'] = df.groupby('user_id')[pred_col].rank(
        method='first', ascending=False)
    df = df[df['rank'] <= 50]
    df = df.groupby('user_id')['item_id'].apply(lambda x: ','.join(
        [str(i) for i in x])).str.split(',', expand=True).reset_index()
    return df


def generate_answer(phase, test_path, submission_name, top_k):
    train_path = 'underexpose_train'
    recom_item = []

    whole_click = pd.DataFrame()
    phase_begin = 7 if phase > 6 else 0
    for c in range(phase_begin, phase + 1):
        print('phase:', c)
        click_train = pd.read_csv(train_path + '/underexpose_train_click-{}.csv'.format(c),
                                  header=None, names=['user_id', 'item_id', 'time'])
        click_test = pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(c),
                                 header=None, names=['user_id', 'item_id', 'time'])

        all_click = click_train.append(click_test)
        whole_click = whole_click.append(all_click)

        all_click = all_click.drop_duplicates(subset=['user_id','item_id','time'], keep='last')
        all_click = all_click.sort_values(by='time')
        
        item_sim_list, user_item = get_sim_item( 
                all_click, 'user_id', 'item_id', c, use_iif=False)

        test_qtime_df = pd.read_csv(test_path + '/underexpose_test_qtime-{}.csv'.format(c),
                                 header=None, names=['user_id', 'item_id', 'time'])
        for i in tqdm(test_qtime_df['user_id'].unique(), desc='recommend'):
            rank_item = recommend(item_sim_list, user_item, i, top_k, 50)
            if len(rank_item) == 0:
                recom_item.append([i, -100, -9999])
            for j in rank_item:
                recom_item.append([i, j[0], j[1]])
    
    # find most popular items
    whole_click = whole_click.drop_duplicates(keep='last')
    top50_click = whole_click['item_id'].value_counts().index[:50].values
    top50_click = ','.join([str(i) for i in top50_click])

    recom_df = pd.DataFrame(recom_item, columns=['user_id', 'item_id', 'sim'])
    result = get_predict(recom_df, 'sim', top50_click)
    result.to_csv(submission_name, index=False, header=None)


# usage: baseline.py <now_phase> <test_path> <submission_name>
# example: python baseline.py 6 underexpose_test submission.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', type=int)
    parser.add_argument('test_path', type=str)
    parser.add_argument('submission_name', type=str)
    args = parser.parse_args()

    # test_path = 'underexpose_test'
    # fake_test_path = 'fake_test'
    top_k = 500
    generate_answer(args.phase, args.test_path, args.submission_name, top_k)
    
    print(f"top k: {top_k}")
    if args.submission_name == 'fake_submission.csv':
        try:
            evaluate(args.phase, args.submission_name)
        except Exception as e:
            print(e)
