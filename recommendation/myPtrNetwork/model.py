import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.python.framework import tensor_util
from tensorflow.contrib import rnn
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes

LSTMCell = rnn.LSTMCell
MultiRNNCell = rnn.MultiRNNCell

def trainable_initial_state(batch_size, state_size,
                            initializer=None, name="initial_state"):
    flat_state_size = nest.flatten(state_size)

    if not initializer:
        flat_initializer = tuple(tf.zeros_initializer for _ in flat_state_size)
    else:
        flat_initializer = tuple(tf.zeros_initializer for initializer in flat_state_size)

    names = ["{}_{}".format(name, i) for i in range(len(flat_state_size))]
    tiled_states = []

    for name, size, init in zip(names, flat_state_size, flat_initializer):
        shape_with_batch_dim = [1, size]
        initial_state_variable = tf.get_variable(
            name, shape=shape_with_batch_dim, initializer=init())

        tiled_state = tf.tile(initial_state_variable,
                              [batch_size, 1], name=(name + "_tiled"))
        tiled_states.append(tiled_state)

    return nest.pack_sequence_as(structure=state_size,
                                 flat_sequence=tiled_states)


def index_matrix_to_pairs(index_matrix):
    # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
    #                        [[0, 2], [1, 3], [2, 1]]]
    replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
    rank = len(index_matrix.get_shape())
    if rank == 2:
        replicated_first_indices = tf.tile(
            tf.expand_dims(replicated_first_indices, dim=1),
            [1, tf.shape(index_matrix)[1]])
    return tf.stack([replicated_first_indices, index_matrix], axis=rank)




class Model(object):
    def __init__(self, config):

        self.task = config.task
        self.debug = config.debug
        self.config = config

        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.attention_dim = config.attention_dim
        self.num_layers = config.num_layers

        self.batch_size = config.batch_size

        self.max_enc_length = config.max_enc_length
        self.max_dec_length = config.max_dec_length
        self.num_glimpse = config.num_glimpse

        self.init_min_val = config.init_min_val
        self.init_max_val = config.init_max_val
        self.initializer = \
            tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

        self.lr_start = config.lr_start
        self.lr_decay_step = config.lr_decay_step
        self.lr_decay_rate = config.lr_decay_rate
        self.max_grad_norm = config.max_grad_norm

        ##############
        # inputs
        ##############

        self.is_training = tf.placeholder_with_default(
            tf.constant(False, dtype=tf.bool),
            shape=(), name='is_training'
        )


        self._build_model()



    def _build_model(self):

        # -----------------定义输入------------------
        self.enc_seq = tf.placeholder(dtype=tf.float32,shape=[self.batch_size,self.max_enc_length,2],name='enc_seq')
        self.target_seq = tf.placeholder(dtype=tf.int32,shape=[self.batch_size,self.max_dec_length],name='target_seq')
        self.enc_seq_length = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='enc_seq_length')
        self.target_seq_length = tf.placeholder(dtype=tf.int32,shape=[self.batch_size],name='target_seq_length')

        # ----------------输入处理-------------------
        # 将输入转换成embed
        # input_dim 是 2，hidden_dim 是 lstm的隐藏层的数量
        input_embed = tf.get_variable(
            "input_embed", [1, self.input_dim, self.hidden_dim],
            initializer=self.initializer)

        # 将 输入转换成embedding,一下是根据源码的转换过程：
        # enc_seq :[batch_size,seq_length,2] -> [batch_size,1,seq_length,2]，在第一维进行维数扩展
        # input_embed : [1,2,256] -> [1,1,2,256] # 在第0维进行维数扩展
        # tf.nn.conv1d首先将input和filter进行填充，然后进行二维卷积，因此卷积之后维度为batch * 1 * seq_length * 256
        # 卷积的步长是[1,1,第三个参数,1]，因此为[1,1,1,1]
        # 最后还有一步squeeze的操作，从tensor中删除所有大小是1的维度，所以最后的维数为batch * seq_length * 256
        self.embeded_enc_inputs = tf.nn.conv1d(
            self.enc_seq, input_embed, 1, "VALID")

        # -----------------encoder------------------

        tf.logging.info("Create a model..")
        with tf.variable_scope("encoder"):
            # 构建一个多层的LSTM
            self.enc_cell = LSTMCell(
                self.hidden_dim,
                initializer=self.initializer)

            if self.num_layers > 1:
                cells = [self.enc_cell] * self.num_layers
                self.enc_cell = MultiRNNCell(cells)

            self.enc_init_state = trainable_initial_state(
                self.batch_size, self.enc_cell.state_size)

            # self.encoder_outputs : [batch_size, max_sequence, hidden_dim]
            self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
                self.enc_cell, self.embeded_enc_inputs,
                self.enc_seq_length, self.enc_init_state)

            # 给最开头添加一个结束标记，同时这个标记也将作为decoder的初始输入
            # batch_size * 1 * hidden_dim
            self.first_decoder_input = tf.expand_dims(trainable_initial_state(
                self.batch_size, self.hidden_dim, name="first_decoder_input"), 1)
            # batch_size * max_sequence + 1 * hidden_dim
            self.enc_outputs = tf.concat(
                [self.first_decoder_input, self.enc_outputs], axis=1)

        # -----------------decoder 训练--------------------
        with tf.variable_scope("decoder"):
            # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
            #                        [[0, 2], [1, 3], [2, 1]]]
            self.idx_pairs = index_matrix_to_pairs(self.target_seq)
            self.embeded_dec_inputs = tf.stop_gradient(
                tf.gather_nd(self.enc_outputs, self.idx_pairs))

            # 给target最后一维增加结束标记,数据都是从1开始的，所以结束也是回到1，所以结束标记为1
            tiled_zero_idxs = tf.tile(tf.zeros(
                [1, 1], dtype=tf.int32), [self.batch_size, 1], name="tiled_zero_idxs")
            self.add_terminal_target_seq = tf.concat([self.target_seq, tiled_zero_idxs], axis=1)
            #如果使用了结束标记的话，要给encoder的输出拼上开始状态，同时给decoder的输入拼上开始状态
            self.embeded_dec_inputs = tf.concat(
                [self.first_decoder_input, self.embeded_dec_inputs], axis=1)

            # 建立一个多层的lstm网络
            self.dec_cell = LSTMCell(
                self.hidden_dim,
                initializer=self.initializer)

            if self.num_layers > 1:
                cells = [self.dec_cell] * self.num_layers
                self.dec_cell = MultiRNNCell(cells)

            # encoder的最后的状态作为decoder的初始状态
            dec_state = self.enc_final_states

            # 预测的序列
            self.predict_indexes = []
            # 预测的softmax序列，用于计算损失
            self.predict_indexes_distribution = []


            # 训练self.max_dec_length  + 1轮，每一轮输入batch * hiddennum
            for j in range(self.max_dec_length  + 1):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_input = tf.squeeze(self.embeded_dec_inputs[:, j, :])  # B * L
                output, dec_state = self.dec_cell(cell_input, dec_state)  # B * L
                # 使用pointer 机制 选择得到softmax的输出，batch * enc_seq + 1
                idx_softmax = self.choose_index(self.enc_outputs, output)
                # 选择每个batch 最大的id [batch]
                idx = tf.argmax(idx_softmax, 1, output_type=dtypes.int32)
                # decoder的每个输出的softmax序列
                self.predict_indexes_distribution.append(idx_softmax)  # D+1 * B * E + 1
                # decoder的每个输出的id
                self.predict_indexes.append(idx)

            self.predict_indexes = tf.convert_to_tensor(self.predict_indexes)
            self.predict_indexes_distribution = tf.convert_to_tensor(self.predict_indexes_distribution)

        # ----------------------decoder 预测----------------------
        # 预测输出的id序列
        self.infer_predict_indexes = []
        # 预测输出的softmax序列
        self.infer_predict_indexes_distribution = []
        with tf.variable_scope("decoder", reuse=True):

            dec_state = self.enc_final_states
            # 预测阶段最开始的输入是之前定义的初始输入
            self.predict_decoder_input = self.first_decoder_input
            for j in range(self.max_dec_length + 1):
                if j > 0:
                    tf.get_variable_scope().reuse_variables()

                self.predict_decoder_input = tf.squeeze(self.predict_decoder_input)  # B * L

                output, dec_state = self.dec_cell(self.predict_decoder_input, dec_state)  # B * L
                # 同样根据pointer机制得到softmax输出
                idx_softmax = self.choose_index(self.enc_outputs, output)  # B * E + 1
                # 选择 最大的那个id
                idx = tf.argmax(idx_softmax, 1, output_type=dtypes.int32)  # B * 1
                # 将选择的id转换为pair
                idx_pairs = index_matrix_to_pairs(idx)
                # 选择的下一个时刻的输入
                self.predict_decoder_input = tf.stop_gradient(
                    tf.gather_nd(self.enc_outputs, idx_pairs))  # B * 1 * L

                # decoder的每个输出的id
                self.infer_predict_indexes.append(idx)
                # decoder的每个输出的softmax序列
                self.infer_predict_indexes_distribution.append(idx_softmax)
            self.infer_predict_indexes = tf.convert_to_tensor(self.infer_predict_indexes,dtype=tf.int32)
            self.infer_predict_indexes_distribution = tf.convert_to_tensor(self.infer_predict_indexes_distribution,dtype=tf.float32)


        # ----------------loss------------------
        with tf.variable_scope("loss"):
            # # 我们计算交叉熵来作为我们的损失
            # # -sum(y * log y')
            # # 首先我们要对我们的输出进行一定的处理，首先我们的target的维度是batch * self.max_dec_length * 1，
            # # 而训练或预测得到的softmax序列是 self.max_dec_length +1 * batch * self.max_enc_length + 1
            # # 所以我们先去掉预测序列的最后一行，然后进行transpose，再转成一行
            # # 对实际的序列，我们先将其转换成one-hot，再转成一行，随后便可以计算损失
            #
            # self.dec_pred_logits = tf.reshape(
            #     tf.transpose(tf.squeeze(self.predict_indexes_distribution), [1, 0, 2]), [-1])  # B * D * E + 1
            # self.dec_inference_logits = tf.reshape(
            #     tf.transpose(tf.squeeze(self.infer_predict_indexes_distribution), [1, 0, 2]),
            #     [-1])  # B * D * E + 1
            # self.dec_target_labels = tf.reshape(tf.one_hot(self.add_terminal_target_seq, depth=self.max_enc_length+ 1), [-1])
            #
            # self.loss = -tf.reduce_sum(self.dec_target_labels * tf.log(self.dec_pred_logits))
            # self.inference_loss = -tf.reduce_mean(self.dec_target_labels * tf.log(self.dec_inference_logits))
            #

            training_logits = tf.identity(tf.transpose(self.predict_indexes_distribution[:-1],[1,0,2]))
            targets = tf.identity(self.target_seq)
            masks = tf.sequence_mask(self.target_seq_length,self.max_dec_length,dtype=tf.float32,name="masks")
            self.loss = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks
            )
            self.optimizer = tf.train.AdamOptimizer(self.lr_start)
            self.train_op = self.optimizer.minimize(self.loss)

    def train(self, sess, batch):
        #对于训练阶段，需要执行self.train_op, self.loss, self.summary_op三个op，并传入相应的数据
        feed_dict = {self.enc_seq: batch['enc_seq'],
                     self.enc_seq_length: batch['enc_seq_length'],
                     self.target_seq: batch['target_seq'],
                     self.target_seq_length: batch['target_seq_length']}
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def eval(self, sess, batch):
        # 对于eval阶段，不需要反向传播，所以只执行self.loss, self.summary_op两个op，并传入相应的数据
        feed_dict = {self.enc_seq: batch['enc_seq'],
                      self.enc_seq_length: batch['enc_seq_length'],
                      self.target_seq: batch['target_seq'],
                      self.target_seq_length: batch['target_seq_length']}
        loss= sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def infer(self, sess, batch):

        feed_dict = {self.enc_seq: batch['enc_seq'],
                     self.enc_seq_length: batch['enc_seq_length'],
                     self.target_seq: batch['target_seq'],
                     self.target_seq_length: batch['target_seq_length']}
        predict = sess.run([self.infer_predict_indexes], feed_dict=feed_dict)
        return predict


    def attention(self,ref, query, with_softmax, scope="attention"):
        """

        :param ref: encoder的输出
        :param query: decoder的输出
        :param with_softmax:
        :param scope:
        :return:
        """
        with tf.variable_scope(scope):
            W_1 = tf.get_variable("W_e", [self.hidden_dim, self.attention_dim], initializer=self.initializer)  # L x A
            W_2 = tf.get_variable("W_d", [self.hidden_dim, self.attention_dim], initializer=self.initializer) # L * A

            dec_portion = tf.matmul(query, W_2)

            scores = [] # S * B
            v_blend = tf.get_variable("v_blend", [self.attention_dim, 1], initializer=self.initializer)  # A x 1
            bais_blend = tf.get_variable("bais_v_blend", [1], initializer=self.initializer)  # 1 x 1
            for i in range(self.max_enc_length + 1):
                refi = tf.matmul(tf.squeeze(ref[:,i,:]),W_1)
                ui = tf.add(tf.matmul(tf.nn.tanh(dec_portion+refi),v_blend),bais_blend) # B * 1
                scores.append(tf.squeeze(ui))
            scores = tf.transpose(scores,[1,0]) # B * S
            if with_softmax:
                return tf.nn.softmax(scores,dim=1)
            else:
                return scores

    def glimpse_fn(self,ref, query, scope="glimpse"):
        p = self.attention(ref, query, with_softmax=True, scope=scope)
        alignments = tf.expand_dims(p, 2)
        return tf.reduce_sum(alignments * ref, [1])

    def choose_index(self,ref,query):
        if self.num_glimpse > 0:
            query = self.glimpse_fn(ref,query)
        return self.attention(ref, query, with_softmax=True, scope="attention")
