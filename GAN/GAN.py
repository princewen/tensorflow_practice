import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
import os
import shutil

img_width = 28
img_height = 28
img_size = img_height * img_width

to_train = True
to_restore = False
output_path = 'output'

#总迭代次数500次
max_epoch = 500

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256

def build_generator(z_prior):
    w1 = tf.Variable(tf.truncated_normal([z_size,h1_size],stddev=0.1),name='g_w1',dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h1_size]),name='g_b1',dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior,w1) + b1)
    w2 = tf.Variable(tf.truncated_normal([h1_size,h2_size],stddev=0.1),name='g_w2',dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h2_size]),name='g_b2',dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1,w2)+b2)
    w3 = tf.Variable(tf.truncated_normal([h2_size,img_size],stddev=0.1),name='g_w3',dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([img_size]),name='g_b3',dtype=tf.float32)
    h3 = tf.matmul(h2,w3)+b3
    x_generate = tf.nn.tanh(h3)
    g_params = [w1,b1,w2,b2,w3,b3]
    return x_generate,g_params


def build_discriminator(x_data,x_generated,keep_prob):
    #将real img 和 generated img拼在一起
    x_in = tf.concat([x_data,x_generated],0)
    w1 = tf.Variable(tf.truncated_normal([img_size,h2_size],stddev=0.1),name='d_w1',dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([h2_size]),name='d_b1',dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in,w1)+b1),keep_prob)
    w2 = tf.Variable(tf.truncated_normal([h2_size,h1_size],stddev=0.1),name='d_w2',dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([h1_size]),name='d_b2',dtype=tf.float32)
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1,w2)+b2),keep_prob)
    w3 = tf.Variable(tf.truncated_normal([h1_size,1]),name='d_w3',dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]),name='d_b3',dtype=tf.float32)
    h3 = tf.matmul(h2,w3)+b3
    """
    
    1，函数原型 tf.slice(inputs,begin,size,name='')
    2，用途：从inputs中抽取部分内容
     inputs：可以是list,array,tensor
     begin：n维列表，begin[i] 表示从inputs中第i维抽取数据时，相对0的起始偏移量，也就是从第i维的begin[i]开始抽取数据
     size：n维列表，size[i]表示要抽取的第i维元素的数目
     有几个关系式如下:
         （1） i in [0,n]
         （2）tf.shape(inputs)[0]=len(begin)=len(size)
         （3）begin[i]>=0   抽取第i维元素的起始位置要大于等于0
         （4）begin[i]+size[i]<=tf.shape(inputs)[i]
         
    """
    """
    h3的size:[batch_size + batch_size,1]
    所以 y_data 是对 real img的判别结果
    y_generated 是对 generated img 的判别结果
    """
    y_data = tf.nn.sigmoid(tf.slice(h3,[0,0],[batch_size,-1],name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h3,[batch_size,0],[-1,-1],name=None))
    d_params = [w1,b1,w2,b2,w3,b3]
    return y_data,y_generated,d_params


def show_result(batch_res,fname,grid_size=(0,0),grid_pad=5):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)

def train():
    # load data
    mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
    x_data = tf.placeholder(tf.float32,[None,img_size],name='x_data')
    z_prior = tf.placeholder(tf.float32,[None,z_size],name='z_prior')
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    global_step = tf.Variable(0,name="global_step",trainable=False)

    x_generated,g_params = build_generator(z_prior)

    y_data,y_generated,d_params = build_discriminator(x_data,x_generated,keep_prob)

    d_loss =-( tf.log(y_data) + tf.log(1-y_generated))
    g_loss = -(tf.log(y_generated))

    optimizer = tf.train.AdamOptimizer(0.0001)

    d_trainer= optimizer.minimize(d_loss,var_list=d_params)
    g_trainer = optimizer.minimize(g_loss,var_list=g_params)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        if to_restore:
            chkpt_fname = tf.train.latest_checkpoint(output_path)
            saver.restore(sess, chkpt_fname)
        else:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            os.mkdir(output_path)

        z_sample_val = np.random.normal(0,1,size=(batch_size,z_size)).astype(np.float32)
        steps = 60000 / batch_size

        for i in range(sess.run(global_step),max_epoch):
            for j in np.arange(steps):
                print("epoch:%s, iter:%s" % (i, j))
                # 每一步迭代，我们都会加载256个训练样本，然后执行一次train_step
                x_value, _ = mnist.train.next_batch(batch_size)
                x_value = 2 * x_value.astype(np.float32) - 1
                z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
                # 执行生成
                sess.run(d_trainer,
                         feed_dict={x_data: x_value, z_prior: z_value, keep_prob: 0.7})
                # 执行判别
                if j % 1 == 0:
                    sess.run(g_trainer,
                             feed_dict={x_data: x_value, z_prior: z_value, keep_prob: 0.7})
            x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
            show_result(x_gen_val, "output/sample{0}.jpg".format(i))
            z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
            show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
            sess.run(tf.assign(global_step, i + 1))
            saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


train()