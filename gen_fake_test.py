import argparse
import pandas as pd
from tqdm import trange

test_path = 'underexpose_test/'
fake_test_path = 'fake_test/'


def generate_fake_test(now_phase):
    for i in trange(now_phase + 1):
        test_click_df = (pd.read_csv(test_path + '/underexpose_test_click-{}.csv'.format(i),
                                     header=None, names=['user_id', 'item_id', 'time']))

        sort_df = test_click_df.sort_values('time')
        sort_df = sort_df[sort_df.duplicated(subset=['user_id'], keep=False)]
        user_cnt = sort_df['user_id'].unique().shape[0]
        fake_test_qtime_with_answer_df = sort_df.groupby('user_id').tail(1)
        assert user_cnt == fake_test_qtime_with_answer_df['user_id'].unique(
        ).shape[0]

        fake_test_qtime_with_answer_df.to_csv(
            f'{fake_test_path}underexpose_test_qtime_with_answer-{i}.csv', index=False, header=False)

        fake_test_click_df = sort_df.drop(fake_test_qtime_with_answer_df.index)
        fake_test_click_df.to_csv(
            f'{fake_test_path}underexpose_test_click-{i}.csv', index=False, header=False)

        fake_test_qtime_df = fake_test_qtime_with_answer_df.drop(columns=['time'])
        fake_test_qtime_df.to_csv(
            f'{fake_test_path}underexpose_test_qtime-{i}.csv', index=False, header=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('now_phase', type=int)
    args = parser.parse_args()

    generate_fake_test(args.now_phase)
