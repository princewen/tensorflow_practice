import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)



def compute_accuracy(v_xs,v_ys,sess):
    #prediction 变为全剧变量
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    #预测值每行是10列，tf.argmax(数据，axis），相等为1，不想等为0
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    # 计算平均值，即计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    # 运行我们的accuracy这一步
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    inital = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1,shape=shape)
    return tf.Variable(inital)

def conv2d(x,W):
    #strides是步长，四维的列表 [1,x_movement,y_movement,1]
    # must have strides[0]=strides[3]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')




xs = tf.placeholder(tf.float32,[None,784])
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

#将图片reshape，第一位是张数的意思，二三位是图片的长和宽，第四位是channel，因为只有灰这一个channel，所以是1
x_image = tf.reshape(xs,[-1,28,28,1])

## conv1 layer ##
#patch是5 * 5 的，输入是1个，输出是32，也就是32个feature map
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1= tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1) # 输出28 * 28 * 32
h_pool1 = max_pool_2x2(h_conv1) #输出 14 * 14 * 32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64]) # 64个feature map
b_conv2 = bias_variable([64])
h_conv2= tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2) # 输出14 * 14 * 64
h_pool2 = max_pool_2x2(h_conv2) #输出 7 * 7 * 64

## func1 layer ##
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1 = tf.nn.dropout(h_fc1,keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1,W_fc2)+b_fc2)

#prediction = add_layer(xs,784,10,activation_function=tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)
                                              ,reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
        if i % 50 == 0:
            print(compute_accuracy(mnist.test.images,mnist.test.labels,sess))
