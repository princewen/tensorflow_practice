import tensorflow as tf
from tensorflow.contrib.framework import arg_scope

from layers import *

class Model(object):
  def __init__(self, config, 
               inputs, labels, enc_seq_length, dec_seq_length, mask,
               reuse=False, is_critic=False):
    self.task = config.task
    self.debug = config.debug
    self.config = config

    self.input_dim = config.input_dim
    self.hidden_dim = config.hidden_dim
    self.num_layers = config.num_layers

    self.max_enc_length = config.max_enc_length
    self.max_dec_length = config.max_dec_length
    self.num_glimpse = config.num_glimpse

    self.init_min_val = config.init_min_val
    self.init_max_val = config.init_max_val
    self.initializer = \
        tf.random_uniform_initializer(self.init_min_val, self.init_max_val)

    self.use_terminal_symbol = config.use_terminal_symbol

    self.lr_start = config.lr_start
    self.lr_decay_step = config.lr_decay_step
    self.lr_decay_rate = config.lr_decay_rate
    self.max_grad_norm = config.max_grad_norm

    self.layer_dict = {}

    ##############
    # inputs
    ##############

    self.is_training = tf.placeholder_with_default(
        tf.constant(False, dtype=tf.bool),
        shape=(), name='is_training'
    )

    self.enc_inputs, self.dec_targets, self.enc_seq_length, self.dec_seq_length, self.mask = \
        smart_cond(
            self.is_training,
            lambda: (inputs['train'], labels['train'], enc_seq_length['train'],
                     dec_seq_length['train'], mask['train']),
            lambda: (inputs['test'], labels['test'], enc_seq_length['test'],
                     dec_seq_length['test'], mask['test'])
        )

    if self.use_terminal_symbol:
      self.dec_seq_length += 1 # terminal symbol

    self._build_model()
    self._build_steps()

    if not reuse:
      self._build_optim()

    self.train_summary = tf.summary.merge([
        tf.summary.scalar("train/total_loss", self.total_loss),
        tf.summary.scalar("train/lr", self.lr),
    ])

    self.test_summary = tf.summary.merge([
        tf.summary.scalar("test/total_loss", self.total_loss),
    ])

  def _build_steps(self):
    def run(sess, fetch, feed_dict, summary_writer, summary):
      fetch['step'] = self.global_step
      if summary is not None:
        fetch['summary'] = summary

      result = sess.run(fetch)
      if summary_writer is not None:
        summary_writer.add_summary(result['summary'], result['step'])
        summary_writer.flush()
      return result

    def train(sess, fetch, summary_writer):
      return run(sess, fetch, feed_dict={},
                 summary_writer=summary_writer, summary=self.train_summary)

    def test(sess, fetch, summary_writer=None):
      return run(sess, fetch, feed_dict={self.is_training: False},
                 summary_writer=summary_writer, summary=self.test_summary)

    self.train = train
    self.test = test

  def _build_model(self):
    tf.logging.info("Create a model..")
    self.global_step = tf.Variable(0, trainable=False)

    input_embed = tf.get_variable(
        "input_embed", [1, self.input_dim, self.hidden_dim],
        initializer=self.initializer)

    with tf.variable_scope("encoder"):
      self.embeded_enc_inputs = tf.nn.conv1d(
          self.enc_inputs, input_embed, 1, "VALID")

    batch_size = tf.shape(self.enc_inputs)[0]
    with tf.variable_scope("encoder"):
      self.enc_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.enc_cell] * self.num_layers
        self.enc_cell = MultiRNNCell(cells)
      self.enc_init_state = trainable_initial_state(
          batch_size, self.enc_cell.state_size)

      # self.encoder_outputs : [None, max_time, output_size]
      self.enc_outputs, self.enc_final_states = tf.nn.dynamic_rnn(
          self.enc_cell, self.embeded_enc_inputs,
          self.enc_seq_length, self.enc_init_state)

      self.first_decoder_input = tf.expand_dims(trainable_initial_state(
          batch_size, self.hidden_dim, name="first_decoder_input"), 1)
      if self.use_terminal_symbol:
        # 0 index indicates terminal
        self.enc_outputs = concat_v2(
            [self.first_decoder_input, self.enc_outputs], axis=1)

    with tf.variable_scope("decoder"):
      self.idx_pairs = index_matrix_to_pairs(self.dec_targets)
      self.embeded_dec_inputs = tf.stop_gradient(
          tf.gather_nd(self.enc_outputs, self.idx_pairs))

      if self.use_terminal_symbol:
        tiled_zero_idxs = tf.tile(tf.zeros(
            [1, 1], dtype=tf.int32), [batch_size, 1], name="tiled_zero_idxs")
        self.dec_targets = concat_v2([self.dec_targets, tiled_zero_idxs], axis=1)

      self.embeded_dec_inputs = concat_v2(
          [self.first_decoder_input, self.embeded_dec_inputs], axis=1)

      self.dec_cell = LSTMCell(
          self.hidden_dim,
          initializer=self.initializer)

      if self.num_layers > 1:
        cells = [self.dec_cell] * self.num_layers
        self.dec_cell = MultiRNNCell(cells)

      self.dec_pred_logits, _, _ = decoder_rnn(
          self.dec_cell, self.embeded_dec_inputs, 
          self.enc_outputs, self.enc_final_states,
          self.dec_seq_length, self.hidden_dim,
          self.num_glimpse, batch_size, is_train=True,
          initializer=self.initializer)
      self.dec_pred_prob = tf.nn.softmax(
          self.dec_pred_logits, 2, name="dec_pred_prob")
      self.dec_pred = tf.argmax(
          self.dec_pred_logits, 2, name="dec_pred")

    with tf.variable_scope("decoder", reuse=True):
      self.dec_inference_logits, _, _ = decoder_rnn(
          self.dec_cell, self.first_decoder_input,
          self.enc_outputs, self.enc_final_states,
          self.dec_seq_length, self.hidden_dim,
          self.num_glimpse, batch_size, is_train=False,
          initializer=self.initializer,
          max_length=self.max_dec_length + int(self.use_terminal_symbol))
      self.dec_inference_prob = tf.nn.softmax(
          self.dec_inference_logits, 2, name="dec_inference_logits")
      self.dec_inference = tf.argmax(
          self.dec_inference_logits, 2, name="dec_inference")

  def _build_optim(self):
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.dec_targets, logits=self.dec_pred_logits)
    inference_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.dec_targets, logits=self.dec_inference_logits)

    def apply_mask(op):
      length = tf.cast(op[:1], tf.int32)
      loss = op[1:]
      return tf.multiply(loss, tf.ones(length, dtype=tf.float32))

    batch_loss = tf.div(
        tf.reduce_sum(tf.multiply(losses, self.mask)),
        tf.reduce_sum(self.mask), name="batch_loss")

    batch_inference_loss = tf.div(
        tf.reduce_sum(tf.multiply(losses, self.mask)),
        tf.reduce_sum(self.mask), name="batch_inference_loss")

    tf.losses.add_loss(batch_loss)
    total_loss = tf.losses.get_total_loss()

    self.total_loss = total_loss
    self.target_cross_entropy_losses = losses
    self.total_inference_loss = batch_inference_loss

    self.lr = tf.train.exponential_decay(
        self.lr_start, self.global_step, self.lr_decay_step,
        self.lr_decay_rate, staircase=True, name="learning_rate")

    optimizer = tf.train.AdamOptimizer(self.lr)

    if self.max_grad_norm != None:
      grads_and_vars = optimizer.compute_gradients(self.total_loss)
      for idx, (grad, var) in enumerate(grads_and_vars):
        if grad is not None:
          grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
      self.optim = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
    else:
      self.optim = optimizer.minimize(self.total_loss, global_step=self.global_step)
