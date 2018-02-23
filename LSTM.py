import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)


lr= 0.001
training_iters = 1000000
batch_size = 128

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10



xs = tf.placeholder(tf.float32,[None,n_inputs,n_steps])
ys = tf.placeholder(tf.float32,[None,10])


weights = {
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}

biases = {
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}


def RNN(X,weights,biases):
    # hidden layer for input to cell
    # X(128,28,28)
    X = tf.reshape(X,[-1,n_inputs])
    X_in = tf.matmul(X,weights['in']) + biases['in']
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units,forget_bias=1.0,state_is_tuple=True)
    #这里生成的state是tuple类型的，因为声明了state_is_tuple参数
    init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32)
    #time_major指时间点是不是在主要的维度，因为我们的num_steps在次维，所以定义为了false
    outputs,final_states = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=init_state,time_major=False)

    #final_states[1] 就是短时记忆h
    results = tf.matmul(final_states[1],weights['out']) + biases['out']

    return results

prediction = RNN(xs,weights,biases)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=ys))

train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction,axis=1),tf.argmax(ys,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #一个step是一行
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={xs: batch_xs, ys: batch_ys}))

        step = step + 1

