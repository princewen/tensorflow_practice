import tensorflow as tf
import numpy as np

arr1 = tf.convert_to_tensor(np.arange(1,25).reshape(2,4,3),dtype=tf.int32)
arr2 = tf.convert_to_tensor(np.arange(1,25).reshape(2,4,3),dtype=tf.int32)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    split_arr1 = tf.split(arr1,[1,1,1],2)
    split_arr2 = tf.split(arr2,[1,1,1],2)
    print(split_arr1)
    print(sess.run(split_arr1))
    print(sess.run(split_arr2))
    res = tf.matmul(split_arr1,split_arr2,transpose_b=True)
    print(sess.run(res))
    res = tf.transpose(res,perm=[1,0,2,3])
    print(sess.run(res))


