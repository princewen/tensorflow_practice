import tensorflow as tf
from abc import abstractclassmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0

    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Layer(object):
    def __init__(self,name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.vars = []

    def __call__(self,inputs):
        outputs = self._call(inputs)
        return outputs


    @abstractclassmethod
    def _call(self,inputs):
        pass


class Dense(Layer):
    def __init__(self,input_dim,output_dim,dropout=0.0,act = tf.nn.relu,name=None):
        super(Dense,self).__init__(name)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act =act
        with tf.variable_scope(self.name):
            self.weight = tf.get_variable(name='weight',shape=(input_dim,output_dim),dtype=tf.float32)
            self.bias = tf.get_variable(name='bias',shape=output_dim,initializer=tf.zeros_initializer())
        self.vars = [self.weight]

    def _call(self,inputs):
        x = tf.nn.dropout(inputs,1-self.dropout)
        output = tf.matmul(x,self.weight) + self.bias
        return self.act(output)



class CrossCompressUnit(Layer):
    def __init__(self,dim,name=None):
        super(CrossCompressUnit,self).__init__(name)
        self.dim = dim
        with tf.variable_scope(self.name):
            self.weight_vv = tf.get_variable(name='weight_vv',shape=(dim,1),dtype=tf.float32)
            self.weight_ev = tf.get_variable(name='weight_ev',shape=(dim,1),dtype=tf.float32)
            self.weight_ve = tf.get_variable(name='weight_ve',shape=(dim,1),dtype=tf.float32)
            self.weight_ee = tf.get_variable(name='weight_ee',shape=(dim,1),dtype=tf.float32)

            self.bias_v = tf.get_variable(name='bias_v',shape=dim,initializer=tf.zeros_initializer())
            self.bias_e = tf.get_variable(name='bias_e',shape=dim,initializer=tf.zeros_initializer())

        self.vars = [self.weight_vv,self.weight_ev,self.weight_ve,self.weight_ee]

    def _call(self,inputs):
        # [batch_size, dim]
        v,e = inputs

        v = tf.expand_dims(v,dim=2)
        e = tf.expand_dims(e,dim=1)


        # [batch_size, dim, dim]
        c_matrix = tf.matmul(v, e)
        c_matrix_transpose = tf.transpose(c_matrix, perm=[0, 2, 1])

        # [batch_size * dim, dim]
        c_matrix = tf.reshape(c_matrix, [-1, self.dim])
        c_matrix_transpose = tf.reshape(c_matrix_transpose, [-1, self.dim])

        v_output = tf.reshape(tf.matmul(c_matrix,self.weight_vv) + tf.matmul(c_matrix_transpose,self.weight_ev),[-1,self.dim]) + self.bias_v

        e_output = tf.reshape(tf.matmul(c_matrix, self.weight_ve) + tf.matmul(c_matrix_transpose, self.weight_ee),
                              [-1, self.dim]) + self.bias_e

        return v_output,e_output


