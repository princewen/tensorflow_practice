#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:   Travis A. Ebesu
@created:  2017-05-08
@summary:
'''
import tensorflow as tf
import sonnet as snt
from .helper import GraphKeys, OPTIMIZER


def _bpr_loss(positive, negative, name=None):
    r"""
    Pairwise Loss from Bayesian Personalized Ranking.

    \log \sigma(pos - neg)

    where \sigma is the sigmoid function, we try to set the ranking

    if pos > neg = + number
    if neg < pos = - number

    Then applying the sigmoid to obtain a monotonically increasing function. Any
    monotonically increasing function could be used, eg piecewise or probit.

    :param positive: Score of prefered example
    :param negative: Score of negative example
    :param name: str, name scope
    :returns: mean loss
    """

    with tf.name_scope(name, 'BPRLoss', [positive, negative]) as scope:
        difference = positive - negative
        # Numerical stability
        eps = 1e-12
        loss = -tf.log(tf.nn.sigmoid(difference) + eps)
        return tf.reduce_mean(loss, name=scope)



class LossLayer(snt.AbstractModule):
    """
    Loss Function Wrapper. Applies regularization from GraphKeys.REGULARIZATION_LOSSES
    """
    def __init__(self, name='Loss'):
        """
        Wrapper Function for loss with l1/l2 regularization

        :param loss_type: str, see rbase.utils.tfutils.Loss for Keys
        :param name: name of this module
        """
        super(LossLayer, self).__init__(name=name)

    def _build(self, X, y):
        """

        :param X: predicted value
        :param y: ground truth
        :returns: Loss with l1/l2 regularization added if in keys
        """
        graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self._loss = tf.squeeze(_bpr_loss(X, y))
        self._regularization = None
        self._loss_no_regularization = self._loss

        # Add regularization
        if graph_regularizers:
            self._regularization = tf.reduce_sum(graph_regularizers)
            tf.add_to_collection(GraphKeys.LOSS_REG, self._regularization)
            tf.add_to_collection(GraphKeys.LOSS_NO_REG, self._loss)
            self._loss = self._loss + self._regularization

        tf.add_to_collection(GraphKeys.LOSSES, self._loss)
        return self._loss

    @property
    def loss(self):
        """
        Total loss including regularization terms
        """
        return self._loss

    @property
    def regularization(self):
        """
        Value of the regularization/weight decay
        """
        return self._regularization

    @property
    def loss_no_regularization(self):
        """
        Obtain the loss without regularization added. This corresponds to
        no regularization
        """
        return self._loss_no_regularization


class DenseLayer(snt.AbstractModule):
    """
    Simple dense layer with an activation function
    """
    def __init__(self, output_size, add_bias=True, activation_fn=None,
                 initializers=None, partitioners=None, regularizers=None,
                 name="DenseLayer"):
        super(DenseLayer, self).__init__(name=name)
        self._output_size = output_size
        self._add_bias = add_bias
        self._initializers = initializers
        self._partitioners = partitioners
        self._regularizers = regularizers
        self._activation_fn = activation_fn
        self._layer = None

    def _build(self, inputs):
        """
        Perform dense/fully connected layer with a activation function
        """
        self._layer = snt.Linear(self._output_size, self._add_bias, self._initializers,
                                 self._partitioners, self._regularizers, name='LinearWx')
        output = self._layer(inputs)
        # Add GraphKeys
        if self._add_bias:
            tf.add_to_collection(GraphKeys.BIASES, self._layer.b)

        tf.add_to_collection(GraphKeys.WEIGHTS, self._layer.w)
        tf.add_to_collection(GraphKeys.PRE_ACTIVATIONS, output)

        if self._activation_fn is None or self._activation_fn == tf.identity:
            return output

        output = self._activation_fn(output)

        # Add to GraphKeys for activation output
        tf.add_to_collection(GraphKeys.ACTIVATIONS, output)
        return output

    # Below are just convenience to access properties from the underlying layer

    @property
    def output_size(self):
        """
        Return the output size of this layer
        """
        return self._layer.output_size

    @property
    def input_shape(self):
        """Returns shape of input `Tensor` passed at last call to `build`."""
        return self._layer.input_shape

    @property
    def w(self):
        """
        Get the weights matrix for this layer
        :returns: Variable of the weights
        """
        return self._layer.w

    @property
    def b(self):
        """Biases for this layer or raises an error if add_bias = False

        :returns: Variable of the biases
        """

        return self._layer.b

    @property
    def layer(self):
        return self._layer


class OptimizerLayer(snt.AbstractModule):

    def __init__(self, optimizer_name, clip=None, global_step=None,
                 params=None, name='Optimizer'):
        """
        Optimizer Wrapper

        :param optimizer_name: str, name of the optimizer to use
        :param clip: float, gradient clipping value to use else None
        :param global_step: tensor, global step to use, default gets default from graph
        :param params: dict for optimizer parameters to override defaults
        :param name: str, name of module name space
        """
        super(OptimizerLayer, self).__init__(name=name)
        self._params = params
        self._optimizer = OPTIMIZER[optimizer_name]
        self._clip = clip
        if global_step is None:
            self._global_step = tf.contrib.framework.get_or_create_global_step()
        else:
            self._global_step = global_step
        self._name = name

    @property
    def train_op(self):
        """
        Return the operation to minimize the loss function
        """
        return self._train

    def _build(self, loss, trainable_variables=None):
        """
        Pass a tensor for the loss to be optimized

        :param loss: tensor
        :returns: Operation to minimize the loss
        """

        # Init optimizer
        self._optimizer = self._optimizer(**self._params)
        tvars = trainable_variables

        # Obtain vars to train
        if trainable_variables is None:
            tvars = tf.trainable_variables()

        # Get gradients
        self._grads_vars = self._optimizer.compute_gradients(loss, tvars,
                                                             colocate_gradients_with_ops=True)

        for g, v in self._grads_vars:
            if g is None:
                print(v)
                print(g)
                print("Trainable Variables error, the graph is not connected")
                raise Exception('Variable may not be connected or set to be trained..')
            tf.add_to_collection(GraphKeys.GRADIENTS, g)

        # Clip gradients
        if self._clip is not None and self._clip > 0:
            self._grads_vars = [(tf.clip_by_norm(g, self._clip), v)
                                for g, v in self._grads_vars]

        self._train = self._optimizer.apply_gradients(self._grads_vars,
                                                      global_step=self._global_step,
                                                      name='ApplyGradients')
        tf.add_to_collection(GraphKeys.TRAIN, g)
        return self._train


class ModelBase(object):

    def __init__(self, config):
        self.config = config
        self._global_step = tf.contrib.framework.get_or_create_global_step()
        with tf.name_scope("LearningRateDecay"):
            self.learning_rate = tf.Variable(float(config.learning_rate),
                                             trainable=False, dtype=tf.float32)
            # Placeholder to decay learning rate by some amount
            self._learning_rate_decay_factor = tf.placeholder(tf.float32,
                                                              name='LearningRateDecayFactor')
            # Operation to decay learning rate
            self._learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * self._learning_rate_decay_factor)

        with tf.name_scope("Dropout"):
            self.dropout = tf.Variable(1.0,
                                       trainable=False, dtype=tf.float32,
                                       name='DropoutProbability')

            self._dropout_update = tf.placeholder(tf.float32,
                                                  name='SetDropoutRate')

            self._set_dropout_op = self.dropout.assign(self._dropout_update)

        # Set the learning rate for the optimizer parameters as our variable
        self.config.optimizer_params['learning_rate'] = self.learning_rate

    def decay_learning_rate(self, session, learning_rate_decay):
        """
        Decay the current learning rate by decay amount
        New Learning Rate = Current Learning Rate * Rate Decay
        """
        session.run(self._learning_rate_decay_op,
                    {self._learning_rate_decay_factor: learning_rate_decay})

    def turn_off_dropout(self, sess):
        """
        Sets keep probability to 1.0
        :param sess: Tf Session
        """
        sess.run(self._set_dropout_op,
                 {self._dropout_update: 1.0})

    def set_dropout(self, sess, rate):
        """Set the dropout rate

        :param sess: Tf Session
        :param rate: float, dropout keep probability
        """
        sess.run(self._set_dropout_op,
                 {self._dropout_update: float(rate)})