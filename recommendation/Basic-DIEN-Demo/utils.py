import tensorflow as tf

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

def dice(_x,axis=-1,epsilon=0.000000001,name=""):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha'+name,_x.get_shape()[-1],initializer=tf.constant_initializer(0.0),dtype=tf.float32)
        input_shape = list(_x.get_shape())

        reduction_axis = list(range(len(input_shape)))
        del reduction_axis[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

    mean = tf.reduce_mean(_x, axis=reduction_axis)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axis)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    # x_normed = tf.layers.batch_normalization(_x, center=False, scale=False)
    x_p = tf.sigmoid(x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False, return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])


    mask = tf.equal(mask,tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1] # Hidden size for rnn layer
    query = tf.layers.dense(query,facts_size,activation=None,name='f1'+stag)
    query = prelu(query)

    queries = tf.tile(query,[1,tf.shape(facts)[1]]) # Batch * Time * Hidden size
    queries = tf.reshape(queries,tf.shape(facts))
    din_all = tf.concat([queries,facts,queries-facts,queries*facts],axis=-1) # Batch * Time * (4 * Hidden size)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag) # Batch * Time * 1

    d_layer_3_all = tf.reshape(d_layer_3_all,[-1,1,tf.shape(facts)[1]])  # Batch * 1 * time
    scores = d_layer_3_all

    key_masks = tf.expand_dims(mask,1) # Batch * 1 * Time
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)

    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T] ，没有的地方用paddings填充

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores,facts) # Batch * 1 * Hidden Size
    else:
        scores = tf.reshape(scores,[-1,tf.shape(facts)[1]]) # Batch * Time
        output = facts * tf.expand_dims(scores,-1) # Batch * Time * Hidden Size
        output = tf.reshape(output,tf.shape(facts))
    if return_alphas:
        return output,scores
    else:
        return output








