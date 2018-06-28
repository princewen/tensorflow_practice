import numpy as np
import tensorflow as tf
import os
import random
from collections import defaultdict


def load_data():
    user_ratings = defaultdict(set)
    max_u_id = -1
    max_i_id = -1
    with open('data/u.data','r') as f:
        for line in f.readlines():
            u,i,_,_ = line.split("\t")
            u = int(u)
            i = int(i)
            user_ratings[u].add(i)
            max_u_id = max(u,max_u_id)
            max_i_id = max(i,max_i_id)


    print("max_u_id:",max_u_id)
    print("max_i_idL",max_i_id)

    return max_u_id,max_i_id,user_ratings

def generate_test(user_ratings):
    """
    对每一个用户u，在user_ratings中随机找到他评分过的一部电影i,保存在user_ratings_test，
    后面构造训练集和测试集需要用到。
    """
    user_test = dict()
    for u,i_list in user_ratings.items():
        user_test[u] = random.sample(user_ratings[u],1)[0]
    return user_test


def generate_train_batch(user_ratings,user_ratings_test,item_count,batch_size=512):
    """
    构造训练用的三元组
    对于随机抽出的用户u，i可以从user_ratings随机抽出，而j也是从总的电影集中随机抽出，当然j必须保证(u,j)不在user_ratings中

    """
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(),1)[0]
        i = random.sample(user_ratings[u],1)[0]
        while i==user_ratings_test[u]:
            i = random.sample(user_ratings[u],1)[0]

        j = random.randint(1,item_count)
        while j in user_ratings[u]:
            j = random.randint(1,item_count)

        t.append([u,i,j])

    return np.asarray(t)


def generate_test_batch(user_ratings,user_ratings_test,item_count):
    """
    对于每个用户u，它的评分电影i是我们在user_ratings_test中随机抽取的，它的j是用户u所有没有评分过的电影集合，
    比如用户u有1000部电影没有评分，那么这里该用户的测试集样本就有1000个
    """
    for u in user_ratings.keys():
        t = []
        i = user_ratings_test[u]
        for j in range(1,item_count + 1):
            if not(j in user_ratings[u]):
                t.append([u,i,j])
        yield np.asarray(t)


def bpr_mf(user_count,item_count,hidden_dim):
    u = tf.placeholder(tf.int32,[None])
    i = tf.placeholder(tf.int32,[None])
    j = tf.placeholder(tf.int32,[None])

    user_emb_w = tf.get_variable("user_emb_w", [user_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))
    item_emb_w = tf.get_variable("item_emb_w", [item_count + 1, hidden_dim],
                                 initializer=tf.random_normal_initializer(0, 0.1))

    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)


    x = tf.reduce_sum(tf.multiply(u_emb,(i_emb-j_emb)),1,keep_dims=True)

    mf_auc = tf.reduce_mean(tf.to_float(x>0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])

    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))

    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)
    return u, i, j, mf_auc, bprloss, train_op


user_count,item_count,user_ratings = load_data()
user_ratings_test = generate_test(user_ratings)

with tf.Session() as sess:
    u,i,j,mf_auc,bprloss,train_op = bpr_mf(user_count,item_count,20)
    sess.run(tf.global_variables_initializer())

    for epoch in range(1,4):
        _batch_bprloss = 0
        for k in range(1,5000):
            uij = generate_train_batch(user_ratings,user_ratings_test,item_count)
            _bprloss,_train_op = sess.run([bprloss,train_op],
                                          feed_dict={u:uij[:,0],i:uij[:,1],j:uij[:,2]})

            _batch_bprloss += _bprloss

        print("epoch:",epoch)
        print("bpr_loss:",_batch_bprloss / k)
        print("_train_op")

        user_count = 0
        _auc_sum = 0.0

        for t_uij in generate_test_batch(user_ratings, user_ratings_test, item_count):
            _auc, _test_bprloss = sess.run([mf_auc, bprloss],
                                              feed_dict={u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]}
                                              )
            user_count += 1
            _auc_sum += _auc
        print("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum / user_count)
        print("")
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    for k, v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        print(v)

#  0号用户对这个用户对所有电影的预测评分
session1 = tf.Session()
u1_dim = tf.expand_dims(values[0][0], 0)
u1_all = tf.matmul(u1_dim, values[1],transpose_b=True)
result_1 = session1.run(u1_all)
print (result_1)

print("以下是给用户0的推荐：")
p = np.squeeze(result_1)
p[np.argsort(p)[:-5]] = 0
for index in range(len(p)):
    if p[index] != 0:
        print (index, p[index])
