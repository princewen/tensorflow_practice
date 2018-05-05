import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from DataReader import FeatureDictionary, DataParser
from matplotlib import pyplot as plt

import config
from AFM import AFM

def load_data():
    dfTrain = pd.read_csv(config.TRAIN_FILE)
    dfTest = pd.read_csv(config.TEST_FILE)

    def preprocess(df):
        cols = [c for c in df.columns if c not in ['id','target']]
        #df['missing_feat'] = np.sum(df[df[cols]==-1].values,axis=1)
        df["missing_feat"] = np.sum((df[cols] == -1).values, axis=1)
        df['ps_car_13_x_ps_reg_03'] = df['ps_car_13'] * df['ps_reg_03']
        return df

    dfTrain = preprocess(dfTrain)
    dfTest = preprocess(dfTest)

    cols = [c for c in dfTrain.columns if c not in ['id','target']]
    cols = [c for c in cols if (not c in config.IGNORE_COLS)]

    X_train = dfTrain[cols].values
    y_train = dfTrain['target'].values

    X_test = dfTest[cols].values
    ids_test = dfTest['id'].values

    cat_features_indices = [i for i,c in enumerate(cols) if c in config.CATEGORICAL_COLS]

    return dfTrain,dfTest,X_train,y_train,X_test,ids_test,cat_features_indices

def run_base_model_nfm(dfTrain,dfTest,folds,pnn_params):
    fd = FeatureDictionary(dfTrain=dfTrain,
                           dfTest=dfTest,
                           numeric_cols=config.NUMERIC_COLS,
                           ignore_cols = config.IGNORE_COLS)
    data_parser = DataParser(feat_dict= fd)
    # Xi_train ：列的序号
    # Xv_train ：列的对应的值
    Xi_train,Xv_train,y_train = data_parser.parse(df=dfTrain,has_label=True)
    Xi_test,Xv_test,ids_test = data_parser.parse(df=dfTest)

    print(dfTrain.dtypes)

    pnn_params['feature_size'] = fd.feat_dim
    pnn_params['field_size'] = len(Xi_train[0])


    _get = lambda x,l:[x[i] for i in l]



    for i, (train_idx, valid_idx) in enumerate(folds):
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        afm = AFM(**pnn_params)
        afm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)


pnn_params = {
    "embedding_size":8,
    "attention_size":10,
    "deep_layers":[32,32],
    "dropout_deep":[0.5,0.5,0.5],
    "deep_layer_activation":tf.nn.relu,
    "epoch":30,
    "batch_size":1024,
    "learning_rate":0.001,
    "optimizer":"adam",
    "batch_norm":1,
    "batch_norm_decay":0.995,
    "verbose":True,
    "random_seed":config.RANDOM_SEED,
    "deep_init_size":50,
    "use_inner":False

}

# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = load_data()

# folds
folds = list(StratifiedKFold(n_splits=config.NUM_SPLITS, shuffle=True,
                             random_state=config.RANDOM_SEED).split(X_train, y_train))

#y_train_pnn,y_test_pnn = run_base_model_pnn(dfTrain,dfTest,folds,pnn_params)
y_train_pnn, y_test_pnn = run_base_model_nfm(dfTrain, dfTest, folds, pnn_params)

