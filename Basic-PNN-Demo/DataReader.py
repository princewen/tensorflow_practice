import pandas as pd

class FeatureDictionary(object):
    def __init__(self,trainfile=None,testfile=None,
                 dfTrain=None,dfTest=None,numeric_cols=[],
                 ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"

        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()




    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)

        else:
            dfTrain = self.dfTrain

        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)

        else:
            dfTest = self.dfTest

        df = pd.concat([dfTrain,dfTest])

        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1

            else:
                us = df[col].unique()
                print(us)
                self.feat_dict[col] = dict(zip(us,range(tc,len(us)+tc)))
                tc += len(us)

        self.feat_dim = tc


class DataParser(object):
    def __init__(self,feat_dict):
        self.feat_dict = feat_dict

    def parse(self,infile=None,df=None,has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"


        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            y = dfi['target'].values.tolist()
            dfi.drop(['id','target'],axis=1,inplace=True)
        else:
            ids = dfi['id'].values.tolist()
            dfi.drop(['id'],axis=1,inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col,axis=1,inplace=True)
                dfv.drop(col,axis=1,inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        xi = dfi.values.tolist()
        xv = dfv.values.tolist()

        if has_label:
            return xi,xv,y
        else:
            return xi,xv,ids


