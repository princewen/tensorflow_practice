import tensorflow as tf


class Generator(object):

    def __init__(self,config):
        """ Basic Set up

        Args:
           num_emb: output vocabulary size
           batch_size: batch size for generator
           emb_dim: LSTM hidden unit dimension
           sequence_length: maximum length of input sequence
           start_token: special token used to represent start of sentence
           initializer: initializer for LSTM kernel and output matrix
        """

        self.num_emb = config.num_emb
        self.batch_size = config.gen_batch_size
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.sequence_length = config.sequence_length
        self.start_token = tf.constant(config.start_token,dtype=tf.int32,shape=[self.batch_size])
        self.initializer = tf.random_normal_initializer(mean=0,stddev=0.1)

    def build_input(self,name):
        """ Buid input placeholder

        Input:
            name: name of network
        Output:
            self.input_seqs_pre (if name == pretrained)
            self.input_seqs_mask (if name == pretrained, optional mask for masking invalid token)
            self.input_seqs_adv (if name == 'adversarial')
            self.rewards (if name == 'adversarial')
        """
        assert name in ['pretrain','adversarial','sample']
        if name == 'pretrain':
            self.input_seqs_pre = tf.placeholder(tf.int32,[None,self.sequence_length],name='input_seqs_pre')
            self.input_seqs_mask = tf.placeholder(tf.float32,[None,self.sequence_length],name='input_seqs_mask')

        elif name == 'adversarial':
            self.input_seqs_adv = tf.placeholder(tf.int32,[None,self.sequence_length],name='input_seqs_adv')
            self.rewards = tf.placeholder(tf.float32,[None,self.sequence_length],name='reward')


    def build_pretrain_netword(self):
        """ Buid pretrained network

        Input:
            self.input_seqs_pre
            self.input_seqs_mask
        Output:
            self.pretrained_loss
            self.pretrained_loss_sum (optional)
        """
        self.build_input(name='pretrain')
        self.pretrained_loss = 0.0
        with tf.variable_scope('teller'):
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim,state_is_tuple=True)
            with tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W",[self.num_emb,self.emb_dim],"float32",self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W",[self.emb_dim,self.num_emb],"float32",self.initializer)


            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    if j==0:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,self.start_token)
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,self.input_seqs_pre[:,j-1])


                    if j == 0:
                        state = lstm1.zero_state(self.batch_size,tf.float32)

                    output,state = lstm1(lstm1_in,state,scope=tf.get_variable_scope())

                    logits = tf.matmul(output,output_W)
                    # 计算每一个lstm的损失
                    pretrained_loss_t = tf.nn.sparse_softmax_cross_entropy_with_logits(logits= logits,labels=self.input_seqs_pre[:,j])
                    pretrained_loss_t = tf.reduce_sum(tf.multiply(pretrained_loss_t,self.input_seqs_mask[:,j]))
                    self.pretrained_loss += pretrained_loss_t
                    word_predict = tf.to_int32(tf.argmax(logits,1))

            self.pretrained_loss /= tf.reduce_sum(self.input_seqs_mask)
            self.pretrained_loss_sum = tf.summary.scalar("pretrained_loss",self.pretrained_loss)


    def build_adversarial_network(self):
        """ Buid adversarial training network

        Input:
            self.input_seqs_adv
            self.rewards
        Output:
            self.gen_loss_adv
        """
        self.build_input(name='adversarial')
        self.softmax_list_reshape = []
        self.softmax_list = []

        with tf.variable_scope('teller'):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim, state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.emb_dim, self.num_emb], "float32", self.initializer)

            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    tf.get_variable_scope().reuse_variables()
                    if j==0:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,self.start_token)
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,self.input_seqs_adv[:,j-1])
                    if j == 0:
                        state = lstm1.zero_state(self.batch_size, tf.float32)

                    output, state = lstm1(lstm1_in, state, scope=tf.get_variable_scope())

                    logits = tf.matmul(output,output_W)
                    softmax = tf.nn.softmax(logits)
                    self.softmax_list.append(softmax) # seqs * batch * emb_size



            self.softmax_list_reshape = tf.transpose(self.softmax_list,perm=[1,0,2]) # batch * seqs * emb_size


            self.pgen_loss_adv = - tf.reduce_sum(
                tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(self.input_seqs_adv,[-1])),self.num_emb,on_value=1.0,off_value=0.0)
                    * tf.log(tf.clip_by_value(tf.reshape(self.softmax_list_reshape,[-1,self.num_emb]),1e-20,1.0)),1
                ) * tf.reshape(self.rewards,[-1]))


    def build_sample_network(self):
        """ Buid sampling network

        Output:
            self.sample_word_list_reshape
        """

        self.build_input(name='sample')
        self.sample_word_list = []
        with tf.variable_scope('teller'):
            tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("lstm"):
                lstm1 = tf.nn.rnn_cell.LSTMCell(self.hidden_dim,state_is_tuple=True)
            with tf.device("/cpu:0"), tf.variable_scope("embedding"):
                word_emb_W = tf.get_variable("word_emb_W", [self.num_emb, self.emb_dim], "float32", self.initializer)
            with tf.variable_scope("output"):
                output_W = tf.get_variable("output_W", [self.emb_dim, self.num_emb], "float32", self.initializer)

            with tf.variable_scope("lstm"):
                for j in range(self.sequence_length):
                    if j==0:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,self.start_token)
                    else:
                        lstm1_in = tf.nn.embedding_lookup(word_emb_W,sample_word)


                    if j==0:
                        state = lstm1.zero_state(self.batch_size,tf.float32)

                    output,state = lstm1(lstm1_in,state,scope=tf.get_variable_scope())

                    logits = tf.matmul(output,output_W)
                    logprob = tf.log(tf.nn.softmax(logits))
                    #Tensorflow 中，想要使用sequence to sequence 模型，在RNN的输出端采样(sampling)，
                    # 可以在softmax层之后，做简单的log p 再用tf.multinomial()来实现：
                    sample_word = tf.reshape(tf.to_int32(tf.multinomial(logprob,1)),shape=[self.batch_size])
                    self.sample_word_list.append(sample_word)

            self.sample_word_list_reshpae = tf.transpose(tf.squeeze(tf.stack(self.sample_word_list)),perm=[1,0])


    def build(self):
        self.build_pretrain_netword()
        self.build_adversarial_network()
        self.build_sample_network()


    def generate(self,sess):
        return sess.run(self.sample_word_list_reshpae)

