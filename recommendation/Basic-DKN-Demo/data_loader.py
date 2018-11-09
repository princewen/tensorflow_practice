import numpy as np
import pandas as pd
from collections import namedtuple

Data = namedtuple("Data",['size','clicked_words','clicked_entities','news_words','news_entities','labels'])




def load_data(args):
    train_df = read(args.train_file)
    test_df = read(args.test_file)
    uid2words,uid2entities = aggregate(train_df,args.max_click_history)
    train_data = transform(train_df,uid2words,uid2entities)
    test_data = transform(test_df,uid2words,uid2entities)
    print(train_df.head(10))
    return train_data,test_data


def read(file):
    df = pd.read_table(file,sep='\t',header=None,names=['user_id','news_words','news_entities','label'])
    df['news_words'] = df['news_words'].map(lambda x:[int(i) for i in x.split(",")])
    df['news_entities'] = df['news_entities'].map(lambda x:[int(i) for i in x.split(",")])
    return df



def aggregate(train_df,max_click_history):
    uid2words = dict()
    uid2entities = dict()
    pos_df = train_df[train_df['label'] == 1]
    for user_id in set(pos_df['user_id']):
        df_user = pos_df[pos_df['user_id']==user_id]
        words = np.array(df_user['news_words'].tolist())
        entities = np.array(df_user['news_entities'].tolist())
        indices = np.random.choice(list(range(0,df_user.shape[0])),size = max_click_history,replace=True)
        uid2words[user_id] = words[indices]
        uid2entities[user_id] = entities[indices]
    return uid2words,uid2entities

def transform(df,uid2words,uid2entities):
    df['clicked_words'] = df['user_id'].map(lambda x:uid2words[x])
    df['clicked_entities'] = df['user_id'].map(lambda x:uid2entities[x])
    data = Data(size=df.shape[0],
                clicked_words=np.array(df['clicked_words'].tolist()),
                clicked_entities=np.array(df['clicked_entities'].tolist()),
                news_words=np.array(df['news_words'].tolist()),
                news_entities=np.array(df['news_entities'].tolist()),
                labels=np.array(df['label']))
    return data

