import tensorflow as tf

# embedding
embedding = tf.constant(
    [
        [0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]
    ],dtype=tf.float32)

feature_batch = tf.constant([2,3,1,0])

get_embedding1 = tf.nn.embedding_lookup(embedding,feature_batch)

feature_batch_one_hot = tf.one_hot(feature_batch,depth=4)

get_embedding2 = tf.matmul(feature_batch_one_hot,embedding)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embedding1,embedding2 = sess.run([get_embedding1,get_embedding2])
    print(embedding1)
    print(embedding2)
    print(sess.run(feature_batch_one_hot))


# embedding1
embedding = tf.get_variable(name='embedding',shape=[4,4],dtype=tf.float32,initializer=tf.random_uniform_initializer)

feature_batch = tf.constant([2,3,1,0])

get_embedding1 = tf.nn.embedding_lookup(embedding,feature_batch)

feature_batch_one_hot = tf.one_hot(feature_batch,depth=4)

get_embedding2 = tf.matmul(feature_batch_one_hot,embedding)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    embedding1,embedding2 = sess.run([get_embedding1,get_embedding2])
    print(embedding1)
    print(embedding2)

# 单维索引
embedding = tf.constant(
    [
        [0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]
    ],dtype=tf.float32)

index_a = tf.Variable([2,3,1,0])

gather_a = tf.gather(embedding, index_a)

gather_a_axis1 = tf.gather(embedding,index_a,axis=1)

b = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
index_b = tf.Variable([2, 4, 6, 8])
gather_b = tf.gather(b, index_b)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(gather_a))
    print(sess.run(gather_b))
    print(sess.run(gather_a_axis1))


# 多维索引
a = tf.Variable([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
index_a = tf.Variable([2])

b = tf.get_variable(name='b',shape=[3,3,2],initializer=tf.random_normal_initializer)
index_b = tf.Variable([[0,1,1],[2,2,0]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.gather_nd(a, index_a)))
    print(sess.run(b))
    print(sess.run(tf.gather_nd(b, index_b)))

# sparse embedding
a = tf.SparseTensor(indices=[[0, 0],[1, 2],[1,3]], values=[1, 2, 3], dense_shape=[2, 4])
b = tf.sparse_tensor_to_dense(a)

embedding = tf.constant(
    [
        [0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]
    ],dtype=tf.float32)

embedding_sparse = tf.nn.embedding_lookup_sparse(embedding, sp_ids=a, sp_weights=None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(embedding_sparse))
    print(sess.run(b))



print("""
[[0.41,0.21],
[0.42,0.22],
[0.43,0.23],
[0.44,0.24]]
""")