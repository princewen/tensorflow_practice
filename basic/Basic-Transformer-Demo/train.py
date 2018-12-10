from data_load import get_batch_data, load_de_vocab, load_en_vocab
from hyperparams import Hyperparams as hp

import tensorflow as tf
from modules import embedding,positional_encoding, multihead_attention,feedforward,label_smoothing
import os,codecs
from tqdm import tqdm


class Graph():
    def __init__(self,is_training=True):
        self.graph = tf.Graph()

        with self.graph.as_default():
            if is_training:
                self.x,self.y,self.num_batch = get_batch_data()
            else:
                self.x = tf.placeholder(tf.int32,shape=(None,hp.maxlen))
                self.y = tf.placeholder(tf.int32,shape=(None,hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat((tf.ones_like(self.y[:,:1]) * 2,self.y[:,:-1]) ,-1) # 2代表<S>，是decoder的初始输入

            de2idx,idx2de = load_de_vocab()
            en2idx,idx2en = load_en_vocab()

            with tf.variable_scope("encoder"):
                # Embedding
                self.enc = embedding(self.x,
                                     vocab_size=len(de2idx),
                                     num_units = hp.hidden_units,
                                     zero_pad=True, # 让padding一直是0
                                     scale=True,
                                     scope="enc_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units = hp.hidden_units,
                                                    zero_pad = False,
                                                    scale = False,
                                                    scope='enc_pe')

                else:
                    self.enc += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.x)[1]),0),[tf.shape(self.x)[0],1]),
                                          vocab_size = hp.maxlen,
                                          num_units = hp.hidden_units,
                                          zero_pad = False,
                                          scale = False,
                                          scope = "enc_pe")

                ##Drop out
                self.enc = tf.layers.dropout(self.enc,rate = hp.dropout_rate,
                                             training = tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ### MultiHead Attention
                        self.enc = multihead_attention(queries = self.enc,
                                                       keys = self.enc,
                                                       num_units = hp.hidden_units,
                                                       num_heads = hp.num_heads,
                                                       dropout_rate = hp.dropout_rate,
                                                       is_training = is_training,
                                                       causality = False
                                                       )
                        self.enc = feedforward(self.enc,num_units = [4 * hp.hidden_units,hp.hidden_units])



            with tf.variable_scope("decoder"):
                # Embedding
                self.dec = embedding(self.decoder_inputs,
                                     vocab_size=len(en2idx),
                                     num_units = hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed")

                ## Positional Encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                                    vocab_size = hp.maxlen,
                                                    num_units = hp.hidden_units,
                                                    zero_pad = False,
                                                    scale = False,
                                                    scope = "dec_pe")
                else:
                    self.dec += embedding(
                        tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                        vocab_size=hp.maxlen,
                        num_units=hp.hidden_units,
                        zero_pad=False,
                        scale=False,
                        scope="dec_pe")

                # Dropout
                self.dec = tf.layers.dropout(self.dec,
                                            rate = hp.dropout_rate,
                                            training = tf.convert_to_tensor(is_training))

                ## Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        ## Multihead Attention ( self-attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.dec,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=True,
                                                       scope="self_attention")

                        ## Multihead Attention ( vanilla attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False,
                                                       scope="vanilla_attention")

                        ## Feed Forward
                        self.dec = feedforward(self.dec, num_units=[4 * hp.hidden_units, hp.hidden_units])

            # Final linear projection
            self.logits = tf.layers.dense(self.dec,len(en2idx))
            self.preds = tf.to_int32(tf.argmax(self.logits,dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y,0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds,self.y)) * self.istarget / (tf.reduce_sum(self.istarget)))

            if is_training:
                # Loss
                # 将one_hot中的0改成了一个很小的数，1改成了一个比较接近于1的数。
                self.y_smoothed = label_smoothing(tf.one_hot(self.y,depth=len(en2idx)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

                self.global_step = tf.Variable(0,name='global_step',trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate = hp.lr,beta1 = 0.9,beta2 = 0.98,epsilon = 1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss,global_step = self.global_step)

                tf.summary.scalar('mean_loss',self.mean_loss)
                self.merged = tf.summary.merge_all()




if __name__ == '__main__':
    de2idx,idx2de = load_de_vocab()
    en2idx,idx2en = load_en_vocab()

    g = Graph(is_training = True)
    print("Graph loaded")
    sv = tf.train.Supervisor(graph=g.graph,
                             logdir=hp.logdir,
                             save_model_secs=0)
    with sv.managed_session() as sess:
        for epoch in range(1, hp.num_epochs + 1):
            if sv.should_stop():
                break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                loss,_ = sess.run([g.mean_loss,g.train_op])
                print(step , ":" ,loss)

            gs = sess.run(g.global_step)
            sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))

    print("Done")