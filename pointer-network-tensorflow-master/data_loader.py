# Most of the codes are from 
# https://github.com/vshallc/PtrNets/blob/master/pointer/misc/tsp.py
import os
import re
import zipfile
import itertools
import threading
import numpy as np
from tqdm import trange, tqdm
from collections import namedtuple

import tensorflow as tf
from download import download_file_from_google_drive

GOOGLE_DRIVE_IDS = {
    'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}

TSP = namedtuple('TSP', ['x', 'y', 'name'])

def length(x, y):
  return np.linalg.norm(np.asarray(x) - np.asarray(y))

# https://gist.github.com/mlalevic/6222750
def solve_tsp_dynamic(points):
  #calc all lengths
  all_distances = [[length(x,y) for y in points] for x in points]
  #initial value - just distance from 0 to every other point + keep the track of edges
  A = {(frozenset([0, idx+1]), idx+1): (dist, [0,idx+1]) for idx,dist in enumerate(all_distances[0][1:])}
  cnt = len(points)
  for m in range(2, cnt):
    B = {}
    for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
      for j in S - {0}:
        B[(S, j)] = min( [(A[(S-{j},k)][0] + all_distances[k][j], A[(S-{j},k)][1] + [j]) for k in S if k != 0 and k!=j])  #this will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
    A = B
  res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
  return np.asarray(res[1]) + 1 # 0 for padding

def generate_one_example(n_nodes, rng):
  nodes = rng.rand(n_nodes, 2).astype(np.float32)
  solutions = solve_tsp_dynamic(nodes)
  return nodes, solutions

def read_paper_dataset(paths, max_length):
  x, y = [], []
  for path in paths:
    tf.logging.info("Read dataset {} which is used in the paper..".format(path))
    length = max(re.findall('\d+', path))
    with open(path) as f:
      for l in tqdm(f):
        inputs, outputs = l.split(' output ')
        x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]))
        y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one
  return x, y

class TSPDataLoader(object):
  def __init__(self, config, rng=None):
    self.config = config
    self.rng = rng

    self.task = config.task.lower()
    self.batch_size = config.batch_size
    self.min_length = config.min_data_length
    self.max_length = config.max_data_length

    self.is_train = config.is_train
    self.use_terminal_symbol = config.use_terminal_symbol
    self.random_seed = config.random_seed

    self.data_num = {}
    self.data_num['train'] = config.train_num
    self.data_num['test'] = config.test_num

    self.data_dir = config.data_dir
    self.task_name = "{}_({},{})".format(
        self.task, self.min_length, self.max_length)

    self.data = None
    self.coord = None
    self.threads = None
    self.input_ops, self.target_ops = None, None
    self.queue_ops, self.enqueue_ops = None, None
    self.x, self.y, self.seq_length, self.mask = None, None, None, None

    paths = self.download_google_drive_file()
    if len(paths) != 0:
      self._maybe_generate_and_save(except_list=paths.keys())
      for name, path in paths.items():
        self.read_zip_and_update_data(path, name)
    else:
      self._maybe_generate_and_save()
    self._create_input_queue()

  def _create_input_queue(self, queue_capacity_factor=16):
    self.input_ops, self.target_ops = {}, {}
    self.queue_ops, self.enqueue_ops = {}, {}
    self.x, self.y, self.seq_length, self.mask = {}, {}, {}, {}

    for name in self.data_num.keys():
      self.input_ops[name] = tf.placeholder(tf.float32, shape=[None, None])
      self.target_ops[name] = tf.placeholder(tf.int32, shape=[None])

      min_after_dequeue = 1000
      capacity = min_after_dequeue + 3 * self.batch_size

      self.queue_ops[name] = tf.RandomShuffleQueue(
          capacity=capacity,
          min_after_dequeue=min_after_dequeue,
          dtypes=[tf.float32, tf.int32],
          shapes=[[self.max_length, 2,], [self.max_length]],
          seed=self.random_seed,
          name="random_queue_{}".format(name))
      self.enqueue_ops[name] = \
          self.queue_ops[name].enqueue([self.input_ops[name], self.target_ops[name]])

      inputs, labels = self.queue_ops[name].dequeue()

      seq_length = tf.shape(inputs)[0]
      if self.use_terminal_symbol:
        mask = tf.ones([seq_length + 1], dtype=tf.float32) # terminal symbol
      else:
        mask = tf.ones([seq_length], dtype=tf.float32)

      self.x[name], self.y[name], self.seq_length[name], self.mask[name] = \
          tf.train.batch(
              [inputs, labels, seq_length, mask],
              batch_size=self.batch_size,
              capacity=capacity,
              dynamic_pad=True,
              name="batch_and_pad")

  def run_input_queue(self, sess):
    self.threads = []
    self.coord = tf.train.Coordinator()

    for name in self.data_num.keys():
      def load_and_enqueue(sess, name, input_ops, target_ops, enqueue_ops, coord):
        idx = 0
        while not coord.should_stop():
          feed_dict = {
              input_ops[name]: self.data[name].x[idx],
              target_ops[name]: self.data[name].y[idx],
          }
          sess.run(self.enqueue_ops[name], feed_dict=feed_dict)
          idx = idx+1 if idx+1 <= len(self.data[name].x) - 1 else 0

      args = (sess, name, self.input_ops, self.target_ops, self.enqueue_ops, self.coord)
      t = threading.Thread(target=load_and_enqueue, args=args)
      t.start()
      self.threads.append(t)
      tf.logging.info("Thread for [{}] start".format(name))

  def stop_input_queue(self):
    self.coord.request_stop()
    self.coord.join(self.threads)
    tf.logging.info("All threads stopped")

  def _maybe_generate_and_save(self, except_list=[]):
    self.data = {}

    for name, num in self.data_num.items():
      if name in except_list:
        tf.logging.info("Skip creating {} because of given except_list {}".format(name, except_list))
        continue
      path = self.get_path(name)

      if not os.path.exists(path):
        tf.logging.info("Creating {} for [{}]".format(path, self.task))

        x = np.zeros([num, self.max_length, 2], dtype=np.float32)
        y = np.zeros([num, self.max_length], dtype=np.int32)

        for idx in trange(num, desc="Create {} data".format(name)):
          n_nodes = self.rng.randint(self.min_length, self.max_length+ 1)
          nodes, res = generate_one_example(n_nodes, self.rng)
          x[idx,:len(nodes)] = nodes
          y[idx,:len(res)] = res

        np.savez(path, x=x, y=y)
        self.data[name] = TSP(x=x, y=y, name=name)
      else:
        tf.logging.info("Skip creating {} for [{}]".format(path, self.task))
        tmp = np.load(path)
        self.data[name] = TSP(x=tmp['x'], y=tmp['y'], name=name)

  def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))

  def download_google_drive_file(self):
    paths = {}
    for mode in ['train', 'test']:
      candidates = []
      candidates.append(
          '{}{}_{}'.format(self.task, self.max_length, mode))
      candidates.append(
          '{}{}-{}_{}'.format(self.task, self.min_length, self.max_length, mode))

      for key in candidates:
        for search_key in GOOGLE_DRIVE_IDS.keys():
          if search_key.startswith(key):
            path = os.path.join(self.data_dir, search_key)
            tf.logging.info("Download dataset of the paper to {}".format(path))

            if not os.path.exists(path):
              download_file_from_google_drive(GOOGLE_DRIVE_IDS[search_key], path)
              if path.endswith('zip'):
                with zipfile.ZipFile(path, 'r') as z:
                  z.extractall(self.data_dir)
            paths[mode] = path

    tf.logging.info("Can't found dataset from the paper!")
    return paths

  def read_zip_and_update_data(self, path, name):
    if path.endswith('zip'):
      filenames = zipfile.ZipFile(path).namelist()
      paths = [os.path.join(self.data_dir, filename) for filename in filenames]
    else:
      paths = [path]

    x_list, y_list = read_paper_dataset(paths, self.max_length)

    x = np.zeros([len(x_list), self.max_length, 2], dtype=np.float32)
    y = np.zeros([len(y_list), self.max_length], dtype=np.int32)

    for idx, (nodes, res) in enumerate(tqdm(zip(x_list, y_list))):
      x[idx,:len(nodes)] = nodes
      y[idx,:len(res)] = res

    if self.data is None:
      self.data = {}

    tf.logging.info("Update [{}] data with {} used in the paper".format(name, path))
    self.data[name] = TSP(x=x, y=y, name=name)
