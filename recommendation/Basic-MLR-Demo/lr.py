import tensorflow as tf
import time
from sklearn.metrics import roc_auc_score
from data import get_data



x=tf.placeholder(tf.float32,shape=[None,108])
y=tf.placeholder(tf.float32,shape=[None])

m=1
learning_rate=0.3
w=tf.Variable(tf.random_normal([108,m], 0.0, 0.5),name='u')


W=tf.matmul(x,w)
p2=tf.reduce_sum(tf.nn.sigmoid(W),1)

pred=p2

cost1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))

cost=tf.add_n([cost1])
train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)


init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess = tf.Session()
sess.run(init_op)
train_x,train_y,test_x,test_y = get_data()


time_s=time.time()
for epoch in range(0,10000):
    f_dict = {x: train_x, y: train_y}
    _, cost_, predict_= sess.run([train_op, cost, pred],feed_dict=f_dict)
    auc=roc_auc_score(train_y, predict_)
    time_t=time.time()
    if epoch % 100 == 0:
        f_dict = {x: test_x, y: test_y}
        _, cost_, predict_test = sess.run([train_op, cost, pred], feed_dict=f_dict)
        test_auc = roc_auc_score(test_y, predict_test)
        print("%d %ld cost:%f,train_auc:%f,test_auc:%f" % (epoch, (time_t-time_s),cost_,auc,test_auc))