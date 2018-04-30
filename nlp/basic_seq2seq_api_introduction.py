import tensorflow as tf
import numpy as np
ids = tf.constant([[1,2,3],[3,2,1]],dtype=tf.int32)
embed_ids = tf.contrib.layers.embed_sequence(ids,vocab_size=3,embed_dim=10)


data = [[[1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4]],
            [[5, 5, 5], [6, 6, 6]]]
x = tf.strided_slice(data,[0,0,0],[1,1,1])
y = tf.strided_slice(data,[0,0,0],[2,2,2],[1,1,1])
z = tf.strided_slice(data,[0,0,0],[2,2,2],[1,2,1])

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
    print(sess.run(z))

c = np.random.random([10, 1])
b = tf.nn.embedding_lookup(c, [1, 3])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(b))
    print(c)


a = tf.constant([[1, 2],[2, 3],[3, 4]], dtype=tf.float32)
tile_a_1 = tf.tile(a, [1,2])
tile_a_2 = tf.tile(a,[2,1])
tile_a_3 = tf.tile(a,[2,2])

with tf.Session() as sess:
    print(sess.run(tile_a_1))
    print(sess.run(tile_a_2))
    print(sess.run(tile_a_3))