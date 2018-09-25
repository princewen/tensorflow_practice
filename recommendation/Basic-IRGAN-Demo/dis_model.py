import tensorflow as tf
import pickle

class DIS():
    def __init__(self,itemNum,userNum,emb_dim,lamda,param=None,initdelta=0.05,learning_rate =0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []

        with tf.variable_scope("discriminator"):
            if self.param == None:
                self.user_embeddings = tf.Variable(tf.random_uniform([self.userNum,self.emb_dim],
                                                                    minval=-self.initdelta,maxval=self.initdelta,
                                                                    dtype=tf.float32))
                self.item_embeddings = tf.Variable(tf.random_uniform([self.itemNum,self.emb_dim],
                                                                    minval=-self.initdelta,maxval=self.initdelta,
                                                                    dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros(self.itemNum))

            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings,self.item_embeddings,self.item_bias]

        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings,self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings,self.i)
        self.i_bias = tf.gather(self.item_bias,self.i)

        self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.label,
                                                                logits = self.pre_logits) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss,var_list=self.d_params)

        self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding,self.i_embedding),1) + self.i_bias
        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        self.all_rating = tf.matmul(self.u_embedding,self.item_embeddings,transpose_a=False,transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding,self.item_embeddings),1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias


