
import numpy as np
import pandas as pd


class FeatureDictionary(object):
    def __init__(self, trainfile=None,testfile=None,
                 numeric_cols=[],
                 ignore_cols=[],
                 cate_cols=[]):

        self.trainfile = trainfile
        #self.testfile = testfile
        self.testfile = testfile
        self.cate_cols = cate_cols
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        df = pd.concat([self.trainfile,self.testfile])
        self.feat_dict = {}
        self.feat_len = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols or col in self.numeric_cols:
                continue
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc




class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict


    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            dfi.drop(["id"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)

        numeric_Xv = dfi[self.feat_dict.numeric_cols].values.tolist()
        dfi.drop(self.feat_dict.numeric_cols,axis=1,inplace=True)

        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        cate_Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        cate_Xv = dfv.values.tolist()
        if has_label:
            return cate_Xi, cate_Xv,numeric_Xv,y
        else:
            return cate_Xi, cate_Xv,numeric_Xv,ids