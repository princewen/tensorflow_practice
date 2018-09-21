import tensorflow as tf

arr1 = tf.convert_to_tensor([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]],dtype=tf.int32)
arr2 = tf.convert_to_tensor([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]],dtype=tf.int32)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    split_arr1 = tf.split(arr1,[1,1,1],2)
    split_arr2 = tf.split(arr2,[1,1,1],2)
    print(sess.run(split_arr1))
    print(sess.run(split_arr2))
    res = tf.matmul(split_arr1,split_arr2,transpose_b=True)
    print(sess.run(res))


