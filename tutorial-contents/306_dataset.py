"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

More information about Dataset: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/programmers_guide/datasets.md
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.data import Dataset


# load your data or create your data in here
npx = np.random.uniform(-1, 1, (1000, 1))                           # x data
npy = np.power(npx, 2) + np.random.normal(0, 0.1, size=npx.shape)   # y data
npx_train, npx_test = np.split(npx, [800])                          # training and test data
npy_train, npy_test = np.split(npy, [800])

# use placeholder, later you may need different data, pass the different data into placeholder
tfx = tf.placeholder(npx_train.dtype, npx_train.shape)
tfy = tf.placeholder(npy_train.dtype, npy_train.shape)

# create dataloader
dataset = Dataset.from_tensor_slices((tfx, tfy))
dataset = dataset.shuffle(buffer_size=1000)   # choose data randomly from this buffer
dataset = dataset.batch(32)                   # batch size you will use
dataset = dataset.repeat(3)                   # repeat for 3 epochs
iterator = dataset.make_initializable_iterator()  # later we have to initialize this one

# your network
bx, by = iterator.get_next()                  # use batch to update
l1 = tf.layers.dense(bx, 10, tf.nn.relu)
out = tf.layers.dense(l1, npy.shape[1])
loss = tf.losses.mean_squared_error(by, out)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
# need to initialize the iterator in this case
sess.run([iterator.initializer, tf.global_variables_initializer()], feed_dict={tfx: npx_train, tfy: npy_train})

for step in range(201):
  try:
    _, trainl = sess.run([train, loss])                       # train
    if step % 10 == 0:
      testl = sess.run(loss, {bx: npx_test, by: npy_test})    # test
      print('step: %i/200' % step, '|train loss:', trainl, '|test loss:', testl)
  except tf.errors.OutOfRangeError:     # if training takes more than 3 epochs, training will be stopped
    print('Finish the last epoch.')
    break