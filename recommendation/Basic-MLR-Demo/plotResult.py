import matplotlib.pyplot as plt
import pandas as pd

mlr5 = pd.read_csv("data/mlr_5.csv",index_col=0)

mlr10 = pd.read_csv("data/mlr_10.csv",index_col=0)

mlr15 = pd.read_csv("data/mlr_15.csv",index_col=0)

mlr20 = pd.read_csv("data/mlr_20.csv",index_col=0)

lr = pd.read_csv("data/lr.csv",index_col=0)




epoch = mlr5['epoch']
train_auc5 = mlr5['test_auc']
train_auc10 = mlr10['test_auc']
train_auc15 = mlr15['test_auc']
train_auc20 = mlr20['test_auc']
train_auclr = lr['train_auc']

l1,= plt.plot(epoch,train_auc5,label='mlr-5')
l2,= plt.plot(epoch,train_auc10,label='mlr-10')
l3, = plt.plot(epoch,train_auc15,label='mlr-15')
l4, = plt.plot(epoch,train_auc20,label='mlr-20')
l5, = plt.plot(epoch,train_auclr,label='lr')
plt.xlabel('epoch')
plt.ylabel('auc')
plt.title('mlr,lr test_auc')
plt.grid()
plt.legend(handles = [l1, l2,l3,l4,l5], labels = ['mlr-5', 'mlr-10','mlr-15','mlr-20','lr'], loc = 'best')
plt.savefig('data/test_zhexian.png')
