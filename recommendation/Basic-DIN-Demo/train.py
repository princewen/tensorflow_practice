import pickle

import numpy as np
import tensorflow as tf
import random

from model import Model
from input import DataInput, DataInputTest

import time

import sys


random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

train_batch_size = 32
test_batch_size = 512

with open('dataset.pkl', 'rb') as f:
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    cate_list = pickle.load(f)
    user_count, item_count, cate_count = pickle.load(f)

print(user_count, item_count, cate_count)

# catelist是item到cate的转换关系
print(len(cate_list))

print(test_set[:5])

best_auc = 0.0

def calc_auc(raw_arr):
    arr = sorted(raw_arr,key=lambda d:d[2])
    auc = 0.0
    fp1,tp1,fp2,tp2 = 0.0,0.0,0.0,0.0

    for record in arr:
        fp2 += record[0]
        tp2 += record[1]

        auc += (fp2 - fp1) * (tp2 + tp1)
        fp1,tp1 = fp2,tp2

    threshold = len(arr) - 1e-3
    if tp2 > threshold or fp2 > threshold:
        return -0.5

    if tp2 * fp2 > 0.0:  # normal auc
        return (1.0 - auc / (2.0 * tp2 * fp2))
    else:
        return None


def _auc_arr(score):
    score_p = score[:,0]
    score_n = score[:,1]

    score_arr = []
    for s in score_p.tolist():
        score_arr.append([0,1,s])
    for s in score_n.tolist():
        score_arr.append([1,0,s])
    return score_arr

def _eval(sess,model):
    auc_sum = 0.0
    score_arr = []
    for _,uij in DataInputTest(test_set,test_batch_size):
        auc_,score_ = model.eval(sess,uij)
        score_arr += _auc_arr(score_)
        auc_sum += auc_ * len(uij[0])

    test_gauc = auc_sum / len(test_set)

    Auc = calc_auc(score_arr)

    global best_auc
    if best_auc < test_gauc:
        best_auc = test_gauc
        model.save(sess, 'save_path/ckpt')
    return test_gauc, Auc



with tf.Session() as sess:
    model = Model(user_count,item_count,cate_count,cate_list)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    lr = 1.0

    start_time = time.time()

    for _ in range(50):

        random.shuffle(train_set)

        epoch_size = round(len(train_set)/ train_batch_size)

        loss_sum = 0.0

        for _,uij in DataInput(train_set,train_batch_size):
            loss = model.train(sess,uij,lr)
            loss_sum += loss

            if model.global_step.eval() % 10 == 0:
                test_gauc,Auc = _eval(sess,model)

                if model.global_step.eval() % 1000 == 0:
                    test_gauc, Auc = _eval(sess, model)
                    print('Epoch %d Global_step %d\tTrain_loss: %.4f\tEval_GAUC: %.4f\tEval_AUC: %.4f' %
                          (model.global_epoch_step.eval(), model.global_step.eval(),
                           loss_sum / 1000, test_gauc, Auc))
                    sys.stdout.flush()
                    loss_sum = 0.0

                if model.global_step.eval() % 336000 == 0:
                    lr = 0.1

            print('Epoch %d DONE\tCost time: %.2f' %
                  (model.global_epoch_step.eval(), time.time() - start_time))
            sys.stdout.flush()
            model.global_epoch_step_op.eval()

        print('best test_gauc:', best_auc)
        sys.stdout.flush()




