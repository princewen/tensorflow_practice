import tensorflow as tf

def dice(_x,axis=-1,epsilon=0.0000001,name=''):

    alphas = tf.get_variable('alpha'+name,_x.get_shape()[-1],
                             initializer = tf.constant_initializer(0.0),
                             dtype=tf.float32)

    input_shape = list(_x.get_shape())
    reduction_axes = list(range(len(input_shape)))

    del reduction_axes[axis] # [0]

    broadcast_shape = [1] * len(input_shape)  #[1,1]
    broadcast_shape[axis] = input_shape[axis] # [1 * hidden_unit_size]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes) # [1 * hidden_unit_size]
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape) #[1 * hidden_unit_size]
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)  # a simple way to use BN to calculate x_p
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x