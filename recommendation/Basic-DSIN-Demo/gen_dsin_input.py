
import os

import numpy as np
import pandas as pd
from deepctr.utils import SingleFeat
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from config import DSIN_SESS_COUNT, DSIN_SESS_MAX_LEN, FRAC

FRAC = FRAC
SESS_COUNT = DSIN_SESS_COUNT


def gen_sess_feature_dsin(row):
    sess_count = DSIN_SESS_COUNT
    sess_max_len = DSIN_SESS_MAX_LEN
    sess_input_dict = {}
    sess_input_length_dict = {}
    for i in range(sess_count):
        sess_input_dict['sess_' + str(i)] = {'cate_id': [], 'brand': []}
        sess_input_length_dict['sess_' + str(i)] = 0
    sess_length = 0
    user, time_stamp = row[1]['user'], row[1]['time_stamp']
    # sample_time = pd.to_datetime(timestamp_datetime(time_stamp ))
    if user not in user_hist_session:
        for i in range(sess_count):
            sess_input_dict['sess_' + str(i)]['cate_id'] = [0]
            sess_input_dict['sess_' + str(i)]['brand'] = [0]
            sess_input_length_dict['sess_' + str(i)] = 0
        sess_length = 0
    else:
        valid_sess_count = 0
        last_sess_idx = len(user_hist_session[user]) - 1
        for i in reversed(range(len(user_hist_session[user]))):
            cur_sess = user_hist_session[user][i]
            if cur_sess[0][2] < time_stamp:
                in_sess_count = 1
                for j in range(1, len(cur_sess)):
                    if cur_sess[j][2] < time_stamp:
                        in_sess_count += 1
                if in_sess_count > 2:
                    sess_input_dict['sess_0']['cate_id'] = [e[0] for e in cur_sess[max(0,
                                                                                       in_sess_count - sess_max_len):in_sess_count]]
                    sess_input_dict['sess_0']['brand'] = [e[1] for e in
                                                          cur_sess[max(0, in_sess_count - sess_max_len):in_sess_count]]
                    sess_input_length_dict['sess_0'] = min(
                        sess_max_len, in_sess_count)
                    last_sess_idx = i
                    valid_sess_count += 1
                    break
        for i in range(1, sess_count):
            if last_sess_idx - i >= 0:
                cur_sess = user_hist_session[user][last_sess_idx - i]
                sess_input_dict['sess_' + str(i)]['cate_id'] = [e[0]
                                                                for e in cur_sess[-sess_max_len:]]
                sess_input_dict['sess_' + str(i)]['brand'] = [e[1]
                                                              for e in cur_sess[-sess_max_len:]]
                sess_input_length_dict['sess_' +
                                       str(i)] = min(sess_max_len, len(cur_sess))
                valid_sess_count += 1
            else:
                sess_input_dict['sess_' + str(i)]['cate_id'] = [0]
                sess_input_dict['sess_' + str(i)]['brand'] = [0]
                sess_input_length_dict['sess_' + str(i)] = 0

        sess_length = valid_sess_count
    return sess_input_dict, sess_input_length_dict, sess_length


if __name__ == "__main__":

    user_hist_session = {}
    FILE_NUM = len(
        list(filter(lambda x: x.startswith('user_hist_session_' + str(FRAC) + '_dsin_'),
                    os.listdir('../sampled_data/'))))

    print('total', FILE_NUM, 'files')

    for i in range(FILE_NUM):
        user_hist_session_ = pd.read_pickle(
            '../sampled_data/user_hist_session_' + str(FRAC) + '_dsin_' + str(i) + '.pkl')  # 19,34
        user_hist_session.update(user_hist_session_)
        del user_hist_session_

    sample_sub = pd.read_pickle(
        '../sampled_data/raw_sample_' + str(FRAC) + '.pkl')

    index_list = []
    sess_input_dict = {}
    sess_input_length_dict = {}
    for i in range(SESS_COUNT):
        sess_input_dict['sess_' + str(i)] = {'cate_id': [], 'brand': []}
        sess_input_length_dict['sess_' + str(i)] = []

    sess_length_list = []
    for row in tqdm(sample_sub[['user', 'time_stamp']].iterrows()):
        sess_input_dict_, sess_input_length_dict_, sess_length = gen_sess_feature_dsin(
            row)
        # index_list.append(index)
        for i in range(SESS_COUNT):
            sess_name = 'sess_' + str(i)
            sess_input_dict[sess_name]['cate_id'].append(
                sess_input_dict_[sess_name]['cate_id'])
            sess_input_dict[sess_name]['brand'].append(
                sess_input_dict_[sess_name]['brand'])
            sess_input_length_dict[sess_name].append(
                sess_input_length_dict_[sess_name])
        sess_length_list.append(sess_length)

    print('done')

    user = pd.read_pickle('../sampled_data/user_profile_' + str(FRAC) + '.pkl')
    ad = pd.read_pickle('../sampled_data/ad_feature_enc_' + str(FRAC) + '.pkl')
    user = user.fillna(-1)
    user.rename(
        columns={'new_user_class_level ': 'new_user_class_level'}, inplace=True)

    sample_sub = pd.read_pickle(
        '../sampled_data/raw_sample_' + str(FRAC) + '.pkl')
    sample_sub.rename(columns={'user': 'userid'}, inplace=True)

    data = pd.merge(sample_sub, user, how='left', on='userid', )
    data = pd.merge(data, ad, how='left', on='adgroup_id')

    sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id', 'final_gender_code', 'age_level',
                       'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level', 'campaign_id',
                       'customer'] # sparse feature for user and ads

    dense_features = ['price'] # dense feature for user and ads

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()  # or Hash
        data[feat] = lbe.fit_transform(data[feat]) # 将不同的取值转换为对应的编号
    mms = StandardScaler()
    data[dense_features] = mms.fit_transform(data[dense_features])

    # class SingleFeat(namedtuple('SingleFeat', ['name', 'dimension', 'hash_flag', 'dtype'])):
    sparse_feature_list = [SingleFeat(feat, data[feat].nunique(
    ) + 1) for feat in sparse_features + ['cate_id', 'brand']]
    dense_feature_list = [SingleFeat(feat, 1) for feat in dense_features]
    sess_feature = ['cate_id', 'brand'] # sess feature for ad

    sess_input = []
    sess_input_length = []
    for i in tqdm(range(SESS_COUNT)):
        sess_name = 'sess_' + str(i)
        for feat in sess_feature:
            sess_input.append(pad_sequences(
                sess_input_dict[sess_name][feat], maxlen=SESS_COUNT, padding='post'))
        sess_input_length.append(sess_input_length_dict[sess_name])

    model_input = [data[feat.name].values for feat in sparse_feature_list] + \
                  [data[feat.name].values for feat in dense_feature_list]
    sess_lists = sess_input + [np.array(sess_length_list)]
    model_input += sess_lists

    if not os.path.exists('../model_input/'):
        os.mkdir('../model_input/')

    pd.to_pickle(model_input, '../model_input/dsin_input_' +
                 str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    pd.to_pickle(data['clk'].values, '../model_input/dsin_label_' +
                 str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    pd.to_pickle({'sparse': sparse_feature_list, 'dense': dense_feature_list},
                 '../model_input/dsin_fd_' + str(FRAC) + '_' + str(SESS_COUNT) + '.pkl')
    print("gen dsin input done")
