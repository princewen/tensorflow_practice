# coding: utf-8
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import FRAC

if __name__ == "__main__":

    user = pd.read_csv('../data/user_profile.csv')
    sample = pd.read_csv('../data/raw_sample.csv')

    if not os.path.exists('../sampled_data/'):
        os.mkdir('../sampled_data/')

    if os.path.exists('../sampled_data/user_profile_' + str(FRAC) + '_.pkl') and os.path.exists(
            '../sampled_data/raw_sample_' + str(FRAC) + '_.pkl'):
        user_sub = pd.read_pickle(
            '../sampled_data/user_profile_' + str(FRAC) + '_.pkl')
        sample_sub = pd.read_pickle(
            '../sampled_data/raw_sample_' + str(FRAC) + '_.pkl')
    else:

        if FRAC < 1.0:
            user_sub = user.sample(frac=FRAC, random_state=1024)
        else:
            user_sub = user
        sample_sub = sample.loc[sample.user.isin(user_sub.userid.unique())]
        pd.to_pickle(user_sub, '../sampled_data/user_profile_' +
                     str(FRAC) + '.pkl')
        pd.to_pickle(sample_sub, '../sampled_data/raw_sample_' +
                     str(FRAC) + '.pkl')

    if os.path.exists('../data/behavior_log_pv.pkl'):
        log = pd.read_pickle('../data/behavior_log_pv.pkl')
    else:
        log = pd.read_csv('../data/behavior_log.csv')
        log = log.loc[log['btag'] == 'pv']
        pd.to_pickle(log, '../data/behavior_log_pv.pkl')

    userset = user_sub.userid.unique()
    log = log.loc[log.user.isin(userset)]
    # pd.to_pickle(log, '../sampled_data/behavior_log_pv_user_filter_' + str(FRAC) + '_.pkl')

    ad = pd.read_csv('../data/ad_feature.csv')
    ad['brand'] = ad['brand'].fillna(-1)

    lbe = LabelEncoder()
    # unique_cate_id = ad['cate_id'].unique()
    # log = log.loc[log.cate.isin(unique_cate_id)]

    unique_cate_id = np.concatenate(
        (ad['cate_id'].unique(), log['cate'].unique()))

    lbe.fit(unique_cate_id)
    ad['cate_id'] = lbe.transform(ad['cate_id']) + 1
    log['cate'] = lbe.transform(log['cate']) + 1

    lbe = LabelEncoder()
    # unique_brand = np.ad['brand'].unique()
    # log = log.loc[log.brand.isin(unique_brand)]

    unique_brand = np.concatenate(
        (ad['brand'].unique(), log['brand'].unique()))

    lbe.fit(unique_brand)
    ad['brand'] = lbe.transform(ad['brand']) + 1
    log['brand'] = lbe.transform(log['brand']) + 1

    log = log.loc[log.user.isin(sample_sub.user.unique())]
    log.drop(columns=['btag'], inplace=True)
    log = log.loc[log['time_stamp'] > 0]

    pd.to_pickle(ad, '../sampled_data/ad_feature_enc_' + str(FRAC) + '.pkl')
    pd.to_pickle(
        log, '../sampled_data/behavior_log_pv_user_filter_enc_' + str(FRAC) + '.pkl')

    print("0_gen_sampled_data done")
