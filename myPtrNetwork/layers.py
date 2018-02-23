import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers
from tensorflow.contrib import seq2seq
from tensorflow.python.util import nest

from tensorflow.contrib.layers.python.layers import utils

smart_cond = utils.smart_cond

LSTMCell = rnn.LSTMCell
MultiRNNCell = rnn.MultiRNNCell
dynamic_rnn_decoder = seq2seq.dynamic_rnn_decoder

