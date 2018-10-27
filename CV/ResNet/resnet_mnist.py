import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib.slim as slim



mnist = input_data.read_data_sets("./MNIST_data/",one_hot=True)

batch_size = 100
learning_rate = 0.01
learning_rate_decay = 0.95
model_save_path = 'model/'


def res_identity(input_tensor,conv_depth,kernel_shape,layer_name):
    with tf.variable_scope(layer_name):
        relu = tf.nn.relu(slim.conv2d(input_tensor,conv_depth,kernel_shape))
        outputs = tf.nn.relu(slim.conv2d(relu,conv_depth,kernel_shape) + input_tensor)
    return outputs

def res_change(input_tensor,conv_depth,kernel_shape,layer_name):
    with tf.variable_scope(layer_name):
        relu = tf.nn.relu(slim.conv2d(input_tensor,conv_depth,kernel_shape,stride=2))
        input_tensor_reshape = slim.conv2d(input_tensor,conv_depth,[1,1],stride=2)
        outputs = tf.nn.relu(slim.conv2d(relu,conv_depth,kernel_shape) + input_tensor_reshape)
    return outputs


def inference(inputs):
    x = tf.reshape(inputs,[-1,28,28,1])
    conv_1 = tf.nn.relu(slim.conv2d(x,32,[3,3])) #28 * 28 * 32
    pool_1 = slim.max_pool2d(conv_1,[2,2]) # 14 * 14 * 32
    block_1 = res_identity(pool_1,32,[3,3],'layer_2')
    block_2 = res_change(block_1,64,[3,3],'layer_3')
    block_3 = res_identity(block_2,64,[3,3],'layer_4')
    block_4 = res_change(block_3,32,[3,3],'layer_5')
    net_flatten = slim.flatten(block_4,scope='flatten')
    fc_1 = slim.fully_connected(slim.dropout(net_flatten,0.8),200,activation_fn=tf.nn.tanh,scope='fc_1')
    output = slim.fully_connected(slim.dropout(fc_1,0.8),10,activation_fn=None,scope='output_layer')
    return output



def train():
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    y_outputs = inference(x)
    global_step = tf.Variable(0, trainable=False)

    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_outputs, labels=tf.argmax(y, 1))
    loss = tf.reduce_mean(entropy)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_outputs, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            x_b, y_b = mnist.train.next_batch(batch_size)
            train_op_, loss_, step = sess.run([train_op, loss, global_step], feed_dict={x: x_b, y: y_b})
            if i % 50 == 0:
                print("training step {0}, loss {1}".format(step, loss_))

                x_b, y_b = mnist.test.images[:500], mnist.test.labels[:500]
                result = sess.run(accuracy, feed_dict={x: x_b, y: y_b})
                print("training step {0},accuracy {1} ".format(step,result))

            saver.save(sess, model_save_path + 'my_model', global_step=global_step)



def main(_):
    train()


if __name__ == '__main__':
    tf.app.run()

