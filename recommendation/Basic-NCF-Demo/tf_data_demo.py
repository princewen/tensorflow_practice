import numpy as np
import tensorflow as tf


data = np.load('data/test_data.npy').item()

print(data['user'][:10])

dataset = tf.data.Dataset.from_tensor_slices(data)

dataset = dataset.shuffle(10000).batch(100)

user = tf.ones(name='user',shape=[None,],dtype=tf.int32)
item = tf.placeholder(name='item',shape=[None,],dtype=tf.int32)

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                            dataset.output_shapes,
                                           shared_name='user'
                                           )




def getBatch():
    sample = iterator.get_next()
    print(sample)
    user = sample['user']
    item = sample['item']

    return user,item


iterator.get_next()
usersum = tf.reduce_mean(user,axis=-1)
itemsum = tf.reduce_mean(item,axis=-1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(2):
        sess.run(iterator.make_initializer(dataset))
        try:
            while True:
                print(sess.run([usersum,itemsum]))
        except tf.errors.OutOfRangeError:
            print("outOfRange")



