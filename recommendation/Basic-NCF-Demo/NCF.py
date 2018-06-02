
import numpy as np
import pandas as pd
import tensorflow as tf

import sys
import metrics


class NCF(object):
    def __init__(self, embed_size, user_size, item_size, lr,
                 optim, initializer, loss_func, activation_func,
                 regularizer_rate, iterator, topk, dropout, is_training):
        """
        Important Arguments.

        embed_size: The final embedding size for users and items.
        optim: The optimization method chosen in this model.
        initializer: The initialization method.
        loss_func: Loss function, we choose the cross entropy.
        regularizer_rate: L2 is chosen, this represents the L2 rate.
        iterator: Input dataset.
        topk: For evaluation, computing the topk items.
        """

        self.embed_size = embed_size
        self.user_size = user_size
        self.item_size = item_size
        self.lr = lr
        self.initializer = initializer
        self.loss_func = loss_func
        self.activation_func = activation_func
        self.regularizer_rate = regularizer_rate
        self.optim = optim
        self.topk = topk
        self.dropout = dropout
        self.is_training = is_training
        self.iterator = iterator


    def get_data(self):
        sample = self.iterator.get_next()
        self.user = sample['user']
        self.item = sample['item']
        self.label = tf.cast(sample['label'],tf.float32)


    def inference(self):
        """ Initialize important settings """
        self.regularizer = tf.contrib.layers.l2_regularizer(self.regularizer_rate)

        if self.initializer == 'Normal':
            self.initializer = tf.truncated_normal_initializer(stddev=0.01)
        elif self.initializer == 'Xavier_Normal':
            self.initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.initializer = tf.glorot_uniform_initializer()

        if self.activation_func == 'ReLU':
            self.activation_func = tf.nn.relu
        elif self.activation_func == 'Leaky_ReLU':
            self.activation_func = tf.nn.leaky_relu
        elif self.activation_func == 'ELU':
            self.activation_func = tf.nn.elu

        if self.loss_func == 'cross_entropy':
            # self.loss_func = lambda labels, logits: -tf.reduce_sum(
            # 		(labels * tf.log(logits) + (
            # 		tf.ones_like(labels, dtype=tf.float32) - labels) *
            # 		tf.log(tf.ones_like(logits, dtype=tf.float32) - logits)), 1)
            self.loss_func = tf.nn.sigmoid_cross_entropy_with_logits

        if self.optim == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(self.lr,
                                                           name='SGD')
        elif self.optim == 'RMSProp':
            self.optim = tf.train.RMSPropOptimizer(self.lr, decay=0.9,
                                                   momentum=0.0, name='RMSProp')
        elif self.optim == 'Adam':
            self.optim = tf.train.AdamOptimizer(self.lr, name='Adam')


    def create_model(self):
        with tf.name_scope('input'):
            self.user_onehot = tf.one_hot(self.user,self.user_size,name='user_onehot')
            self.item_onehot = tf.one_hot(self.item,self.item_size,name='item_onehot')

        with tf.name_scope('embed'):
            self.user_embed_GMF = tf.layers.dense(inputs = self.user_onehot,
                                                  units = self.embed_size,
                                                  activation = self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_GMF')

            self.item_embed_GMF = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_GMF')

            self.user_embed_MLP = tf.layers.dense(inputs=self.user_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='user_embed_MLP')
            self.item_embed_MLP = tf.layers.dense(inputs=self.item_onehot,
                                                  units=self.embed_size,
                                                  activation=self.activation_func,
                                                  kernel_initializer=self.initializer,
                                                  kernel_regularizer=self.regularizer,
                                                  name='item_embed_MLP')



        with tf.name_scope("GMF"):
            self.GMF = tf.multiply(self.user_embed_GMF,self.item_embed_GMF,name='GMF')

        with tf.name_scope("MLP"):
            self.interaction = tf.concat([self.user_embed_MLP, self.item_embed_MLP],
                                         axis=-1, name='interaction')

            self.layer1_MLP = tf.layers.dense(inputs=self.interaction,
                                              units=self.embed_size * 2,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer1_MLP')
            self.layer1_MLP = tf.layers.dropout(self.layer1_MLP, rate=self.dropout)

            self.layer2_MLP = tf.layers.dense(inputs=self.layer1_MLP,
                                              units=self.embed_size,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer2_MLP')
            self.layer2_MLP = tf.layers.dropout(self.layer2_MLP, rate=self.dropout)

            self.layer3_MLP = tf.layers.dense(inputs=self.layer2_MLP,
                                              units=self.embed_size // 2,
                                              activation=self.activation_func,
                                              kernel_initializer=self.initializer,
                                              kernel_regularizer=self.regularizer,
                                              name='layer3_MLP')
            self.layer3_MLP = tf.layers.dropout(self.layer3_MLP, rate=self.dropout)

        with tf.name_scope('concatenation'):
            self.concatenation = tf.concat([self.GMF,self.layer3_MLP],axis=-1,name='concatenation')


            self.logits = tf.layers.dense(inputs= self.concatenation,
                                          units = 1,
                                          activation=None,
                                          kernel_initializer=self.initializer,
                                          kernel_regularizer=self.regularizer,
                                          name='predict')

            self.logits_dense = tf.reshape(self.logits,[-1])

        with tf.name_scope("loss"):

            self.loss = tf.reduce_mean(self.loss_func(
                labels=self.label, logits=self.logits_dense, name='loss'))
            # self.loss = tf.reduce_mean(self.loss_func(self.label, self.logits),
            # 								name='loss')

        with tf.name_scope("optimzation"):
            self.optimzer = self.optim.minimize(self.loss)


    def eval(self):
        with tf.name_scope("evaluation"):
            self.item_replica = self.item
            _, self.indice = tf.nn.top_k(tf.sigmoid(self.logits_dense), self.topk)


    def summary(self):
        """ Create summaries to write on tensorboard. """
        self.writer = tf.summary.FileWriter('./graphs/NCF', tf.get_default_graph())
        with tf.name_scope("summaries"):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()




    def build(self):
        self.get_data()
        self.inference()
        self.create_model()
        self.eval()
        self.summary()
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, step):
        """ Train the model step by step. """
        if self.is_training:
            loss, optim, summaries = session.run(
                [self.loss, self.optimzer, self.summary_op])
            self.writer.add_summary(summaries, global_step=step)
        else:
            indice, item = session.run([self.indice, self.item_replica])
            prediction = np.take(item, indice)

            return prediction, item
