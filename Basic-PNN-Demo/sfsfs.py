import tensorflow as tf




t1 = tf.convert_to_tensor([[2,2],
                           [2,3]])

t1_1 = tf.reshape(t1,shape=[2,2,1])
t1_2 = tf.reshape(t1,shape=[2,1,2])

t = tf.matmul(t1_1,t1_2)

with tf.Session() as sess:
    print(sess.run(t))