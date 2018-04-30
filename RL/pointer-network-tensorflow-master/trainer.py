import os
import numpy as np
from tqdm import trange
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope

from model import Model
from utils import show_all_variables
from data_loader import TSPDataLoader

class Trainer(object):
  def __init__(self, config, rng):
    self.config = config
    self.rng = rng

    self.task = config.task
    self.model_dir = config.model_dir
    self.gpu_memory_fraction = config.gpu_memory_fraction

    self.log_step = config.log_step
    self.max_step = config.max_step
    self.num_log_samples = config.num_log_samples
    self.checkpoint_secs = config.checkpoint_secs

    if config.task.lower().startswith('tsp'):
      self.data_loader = TSPDataLoader(config, rng=self.rng)
    else:
      raise Exception("[!] Unknown task: {}".format(config.task))

    self.model = Model(
        config,
        inputs=self.data_loader.x,
        labels=self.data_loader.y,
        enc_seq_length=self.data_loader.seq_length,
        dec_seq_length=self.data_loader.seq_length,
        mask=self.data_loader.mask)

    self.build_session()
    show_all_variables()

  def build_session(self):
    self.saver = tf.train.Saver()
    self.summary_writer = tf.summary.FileWriter(self.model_dir)

    sv = tf.train.Supervisor(logdir=self.model_dir,
                             is_chief=True,
                             saver=self.saver,
                             summary_op=None,
                             summary_writer=self.summary_writer,
                             save_summaries_secs=300,
                             save_model_secs=self.checkpoint_secs,
                             global_step=self.model.global_step)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=self.gpu_memory_fraction,
        allow_growth=True) # seems to be not working
    sess_config = tf.ConfigProto(allow_soft_placement=True,
                                 gpu_options=gpu_options)

    self.sess = sv.prepare_or_wait_for_session(config=sess_config)

  def train(self):
    tf.logging.info("Training starts...")
    self.data_loader.run_input_queue(self.sess)

    summary_writer = None
    for k in trange(self.max_step, desc="train"):
      fetch = {
          'optim': self.model.optim,
      }
      result = self.model.train(self.sess, fetch, summary_writer)

      if result['step'] % self.log_step == 0:
        self._test(self.summary_writer)

      summary_writer = self._get_summary_writer(result)

    self.data_loader.stop_input_queue()

  def test(self):
    tf.logging.info("Testing starts...")
    self.data_loader.run_input_queue(self.sess)

    for idx in range(10):
      self._test(None)

    self.data_loader.stop_input_queue()

  def _test(self, summary_writer):
    fetch = {
        'loss': self.model.total_inference_loss,
        'pred': self.model.dec_inference,
        'true': self.model.dec_targets,
    }
    result = self.model.test(self.sess, fetch, summary_writer)

    tf.logging.info("")
    tf.logging.info("test loss: {}".format(result['loss']))
    for idx in range(self.num_log_samples):
      pred, true = result['pred'][idx], result['true'][idx]
      tf.logging.info("test pred: {}".format(pred))
      tf.logging.info("test true: {} ({})".format(true, np.array_equal(pred, true)))

    if summary_writer:
      summary_writer.add_summary(result['summary'], result['step'])

  def _get_summary_writer(self, result):
    if result['step'] % self.log_step == 0:
      return self.summary_writer
    else:
      return None
