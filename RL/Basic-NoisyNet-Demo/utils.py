import tensorflow as tf
from tensorflow.python.framework import ops


def conv(inputs, kernel_shape, bias_shape, strides, w_i, b_i=None, activation=tf.nn.relu):

    weights = tf.get_variable('weights', shape=kernel_shape, initializer=w_i)
    conv = tf.nn.conv2d(inputs, weights, strides=strides, padding='SAME')
    if bias_shape is not None:
        biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
        return activation(conv + biases) if activation is not None else conv+biases
    return activation(conv) if activation is not None else conv

def noisy_dense(inputs, units, bias_shape, c_names, w_i, b_i=None, activation=tf.nn.relu, noisy_distribution='factorised'):
    def f(e_list):
        return tf.multiply(tf.sign(e_list), tf.pow(tf.abs(e_list), 0.5))

    if not isinstance(inputs, ops.Tensor):
        inputs = ops.convert_to_tensor(inputs, dtype='float')

    if len(inputs.shape) > 2:
        inputs = tf.contrib.layers.flatten(inputs)
    flatten_shape = inputs.shape[1]
    weights = tf.get_variable('weights', shape=[flatten_shape, units], initializer=w_i)
    w_sigma = tf.get_variable('w_sigma', [flatten_shape, units], initializer=w_i, collections=c_names)
    if noisy_distribution == 'independent':
        weights += tf.multiply(tf.random_normal(shape=w_sigma.shape), w_sigma)
    elif noisy_distribution == 'factorised':
        noise_1 = f(tf.random_normal(tf.TensorShape([flatten_shape, 1]), dtype=tf.float32))  # 注意是列向量形式，方便矩阵乘法
        noise_2 = f(tf.random_normal(tf.TensorShape([1, units]), dtype=tf.float32))
        weights += tf.multiply(noise_1 * noise_2, w_sigma)
    dense = tf.matmul(inputs, weights)
    if bias_shape is not None:
        assert bias_shape[0] == units
        biases = tf.get_variable('biases', shape=bias_shape, initializer=b_i)
        b_noise = tf.get_variable('b_noise', [1, units], initializer=b_i, collections=c_names)
        if noisy_distribution == 'independent':
            biases += tf.multiply(tf.random_normal(shape=b_noise.shape), b_noise)
        elif noisy_distribution == 'factorised':
            biases += tf.multiply(noise_2, b_noise)
        return activation(dense + biases) if activation is not None else dense + biases
    return activation(dense) if activation is not None else dense






