import argparse
import math
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from evaluation import evaluate


def get_sim_item(df, user_col, item_col, use_iif=False):
    """
    sim_item = {
        item_id: {relate_item_id: count / log(1 + items_count)}
    }
    """
    user_item_df = df.groupby(user_col)[item_col].agg(list).reset_index()
    user_item_dict = dict(zip(user_item_df[user_col], user_item_df[item_col]))
    
    user_time_ = df.groupby(user_col)['time'].agg(list).reset_index()
    user_time_dict = dict(zip(user_time_[user_col], user_time_['time']))

    item_cnt = df[item_col].value_counts().to_dict()
    
    sim_item = {}
    for user, items in tqdm(user_item_dict.items(), desc='calc sim'):
        for loc1, item in enumerate(items):
            item_cnt[item] += 1
            sim_item.setdefault(item, {})
            for loc2, relate_item in enumerate(items):
                if item == relate_item:
                    continue
                sim_item[item].setdefault(relate_item, 0)

                if use_iif:
                    sim_item[item][relate_item] += 1 / math.log(1 + len(items))
                else:
                    t1 = user_time_dict[user][loc1]
                    t2 = user_time_dict[user][loc2]
                    t12_diff = abs(t1 - t2)
                    loc12_diff = abs(loc1 - loc2)
                    time_weight = (0.8**(loc12_diff-1)) * (1 - t12_diff*10000)
                    if loc2 > loc1:
                        # 正向
                        weight = 1.0 * time_weight
                        sim_item[item][relate_item] += weight * (1 / math.log(1 + len(items)))
                    else:
                        # 逆向
                        weight = 0.7 * time_weight
                        sim_item[item][relate_item] += weight * (1 / math.log(1 + len(items)))
                
    for i, related_items in tqdm(sim_item.items(), desc='normalize'):
        for j, cij in related_items.items():
            sim_item[i][j] = cij / math.log(item_cnt[i]*item_cnt[j])

    return sim_item, user_item_dict


def recommend(sim_item_corr, user_item_dict, user_id, top_k, item_num):
    rank = {}
    interacted_items = user_item_dict[user_id]
    interacted_items = interacted_items[::-1]
    for loc, i in enumerate(interacted_items):
        for j, wij in sorted(sim_item_corr[i].items(), reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)
                rank[j] += wij * (0.7**loc)

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
    for c in range(phase + 1):
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
            all_click, 'user_id', 'item_id', use_iif=False)

        test_qtime_df = pd.read_csv(test_path + '/underexpose_test_qtime-{}.csv'.format(c),
                                 header=None, names=['user_id', 'item_id', 'time'])
        for i in tqdm(test_qtime_df['user_id'].unique(), desc='recommend'):
            rank_item = recommend(item_sim_list, user_item, i, top_k, 50)
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
# example: python baseline.py 4 underexpose_test submission.csv
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', type=int)
    parser.add_argument('test_path', type=str)
    parser.add_argument('submission_name', type=str)
    args = parser.parse_args()

    # test_path = 'underexpose_test'
    # fake_test_path = 'fake_test'
    top_k = 3500
    generate_answer(args.phase, args.test_path, args.submission_name, top_k)
    
    print(f"top k: {top_k}")
    if args.submission_name == 'fake_submission.csv':
        try:
            evaluate(args.phase, args.submission_name)
        except Exception as e:
            print(e)
