import argparse
import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm

from util.gmf import PairwiseGMF
from util.helper import BaseConfig
from util.data import Dataset


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, default=0)
parser.add_argument('--iters', help='Max iters', type=int, default=15)
parser.add_argument('-b', '--batch_size', help='Batch Size', type=int, default=128)
parser.add_argument('-e', '--embedding', help='Embedding Size', type=int, default=50)
parser.add_argument('--dataset', help='path to npz file', type=str, default='pretrain_data/citeulike-a.npz')
parser.add_argument('-n', '--neg', help='Negative Samples Count', type=int, default=4)
parser.add_argument('--l2', help='l2 Regularization', type=float, default=0.001)
parser.add_argument('-o', '--output', help='save filename for trained embeddings', type=str,
                    default='pretrain/citeulike-a_e50.npz')


FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

class Config(BaseConfig):
    filename = FLAGS.dataset
    embed_size = FLAGS.embedding
    batch_size = FLAGS.batch_size
    l2 = FLAGS.l2
    user_count = -1
    item_count = -1
    optimizer = 'adam'
    neg_count = FLAGS.neg
    learning_rate = 0.001

config = Config()
dataset = Dataset(config.filename)
config.item_count = dataset.item_count
config.user_count = dataset.user_count
tf.logging.info("\n\n%s\n\n" % config)

model = PairwiseGMF(config)
sv = tf.train.Supervisor(logdir=None, save_model_secs=0, save_summaries_secs=0)
sess = sv.prepare_or_wait_for_session(
    config=tf.ConfigProto(gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=0.1,
        allow_growth=True)))

for i in range(FLAGS.iters):
    if sv.should_stop():
        break
    progress = tqdm(enumerate(dataset.get_data(FLAGS.batch_size, False, FLAGS.neg)),
                    dynamic_ncols=True, total=(dataset.train_size * FLAGS.neg) // FLAGS.batch_size)
    loss = []
    for k, example in progress:
        feed = {
            model.input_users: example[:, 0],
            model.input_items: example[:, 1],
            model.input_items_negative: example[:, 2],
        }
        batch_loss, _ = sess.run([model.loss, model.train], feed)
        loss.append(batch_loss)
        progress.set_description(u"[{}] Loss: {:,.4f} » » » » ".format(i, batch_loss))

    print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(i, np.mean(loss)))

user_embed, item_embed, v = sess.run([model.user_memory.embeddings, model.item_memory.embeddings, model.v.w])
np.savez(FLAGS.output, user=user_embed, item=item_embed, v=v)
print('Saving to: %s' % FLAGS.output)
sv.request_stop()
