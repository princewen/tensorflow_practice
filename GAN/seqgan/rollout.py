import tensorflow as tf
class rollout():
    """Rollout implementation for generator"""

    def __init__(self,config):
        self.sequence_length = config.sequence_length
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.emb_dim = config.emb_dim
        self.batch_size = config.gen_batch_size
        self.start_token = config.start_token
        self.pred_seq = tf.placeholder(tf.int32,[None,self.sequence_length],name='pred_seq_rollout')
        self.sample_rollout_step = []


        with tf.variable_scope('teller'):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], tf.float32)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.emb_dim, self.num_emb], tf.float32)


            zero_state = lstm1.zero_state([self.batch_size],tf.float32)

            start_token = tf.constant(self.start_token,dtype=tf.int32,shape=[self.batch_size])

            for step in range(1,self.sequence_length):
                if step % 5 == 0:
                    print("Rollout step: {}".format(step))

                sample_rollout_left = tf.reshape(self.pred_seq[:,0:step],shape=[self.batch_size,step])
                sample_rollout_right = []

                # Update the hidden state for i < step to prepare sampling token for i >= step
                for j in range(step):
                    if j==0:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,start_token)
                    else:
                        tf.get_variable_scope().reuse_variables()
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,self.pred_seq[:,j-1])

                    with tf.variable_scope("lstm"):
                        if j==0:
                            output,state = lstm1(lstm1_in,zero_state,scope=tf.get_variable_scope())
                        else:
                            output,state = lstm1(lstm1_in,state,scope=tf.get_variable_scope())

                for j in range(step,self.sequence_length):
                    if j==step:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,self.pred_seq[:,j-1])
                    else:
                        # stop_gradient 停止计算节点的梯度
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,tf.stop_gradient(sample_word))

                    with tf.variable_scope("lstm"):
                        output ,state = lstm1(lstm1_in,state,scope=tf.get_variable_scope())
                        logits = tf.matmul(output,output_W)
                        log_probs = tf.log(tf.nn.softmax(logits))
                        sample_word = tf.to_int32(tf.squeeze(tf.multinomial(log_probs,1)))
                        sample_rollout_right.append(sample_word)


                sample_rollout_right = tf.transpose(tf.stack(sample_rollout_right))
                sample_rollout = tf.concat([sample_rollout_left,sample_rollout_right],axis=1)
                self.sample_rollout_step.append(sample_rollout)

