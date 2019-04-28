import pandas as pd

import numpy as np

import random

train_data = np.zeros([3,10000],dtype=np.int32)

test_data = np.zeros([3,100],dtype=np.int32)

for i in range(10000):
    train_data[0,i] = random.randint(0,200)
    train_data[1,i] = random.randint(0,200)
    train_data[2,i] = random.randint(0,20000)

for i in range(100):
    test_data[0,i] = random.randint(0, 200)
    test_data[1,i] = random.randint(0, 200)
    test_data[2,i] = random.randint(0, 20000)

train_data = np.transpose(train_data)
test_data = np.transpose(test_data)


train_df = pd.DataFrame(train_data,columns=['SessionId','ItemId','Timestamps']).to_csv('data/train.csv')
test_df = pd.DataFrame(test_data,columns=['SessionId','ItemId','Timestamps']).to_csv('data/test.csv')
