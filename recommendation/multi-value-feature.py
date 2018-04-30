csv = [
  "1,harden|james|curry",
  "2,wrestbrook|harden|durant",
  "3,|paul|towns",
]
#第二列是不定长的特征。处理如下：

import tensorflow as tf

# Purposefully omitting "bourne" to demonstrate OOV mappings.
TAG_SET = ["harden", "james", "curry", "durant", "paul","towns","wrestbrook"]


def sparse_from_csv(csv):
  ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
  table = tf.contrib.lookup.index_table_from_tensor(
      mapping=TAG_SET, default_value=-1) ## 这里构造了个查找表 ##
  split_tags = tf.string_split(post_tags_str, "|")
  return tf.SparseTensor(
      indices=split_tags.indices,
      values=table.lookup(split_tags.values), ## 这里给出了不同值通过表查到的index ##
      dense_shape=split_tags.dense_shape)

# Optionally create an embedding for this.




TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM]))
tags = sparse_from_csv(csv)
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)

# Test it out
with tf.Session() as s:
  s.run([tf.global_variables_initializer(), tf.tables_initializer()])
  print(s.run([embedded_tags]))