import sys
import numpy as np
import tensorflow as tf

from config import get_config
from utils import prepare_dirs_and_logger, save_config
from data_util import gen_data
from model import Model


config = None


def main(_):
    prepare_dirs_and_logger(config)

    if not config.task.lower().startswith('tsp'):
        raise Exception("[!] Task should starts with TSP")

    if config.max_enc_length is None:
        config.max_enc_length = config.max_data_length
    if config.max_dec_length is None:
        config.max_dec_length = config.max_data_length

    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    model = Model(config)

    batch_size = config.batch_size

    train_enc_seq, train_target_seq, train_enc_seq_length, train_target_seq_length = gen_data('data/tsp10.txt')

    eval_enc_seq,eval_target_seq,eval_enc_seq_length,eval_target_seq_length = train_enc_seq[-batch_size:], \
                                                                              train_target_seq[-batch_size:], \
                                                                              train_enc_seq_length[-batch_size:], \
                                                                              train_target_seq_length[-batch_size:]

    train_enc_seq, train_target_seq, train_enc_seq_length, train_target_seq_length= train_enc_seq[: -batch_size], \
                                                                                  train_target_seq[:-batch_size], \
                                                                                  train_enc_seq_length[:-batch_size], \
                                                                                  train_target_seq_length[:-batch_size]

    test_enc_seq, test_target_seq, test_enc_seq_length, test_target_seq_length = gen_data('data/tsp10_test.txt')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(min(config.max_step,len(train_enc_seq)//batch_size)):
            train_batch={
                'enc_seq': train_enc_seq[step * batch_size:(step + 1) * batch_size],
                'enc_seq_length': train_enc_seq_length[step * batch_size:(step + 1) * batch_size],
                'target_seq': train_target_seq[step * batch_size:(step + 1) * batch_size],
                'target_seq_length': train_target_seq_length[step * batch_size:(step + 1) * batch_size]
            }
            loss = model.train(sess,train_batch)
            print(str(step) + " train loss : " + str(loss))

            if step > 0 and step % config.eval_step == 0:
                eval_batch = {
                    'enc_seq': eval_enc_seq,
                    'enc_seq_length': eval_enc_seq_length,
                    'target_seq': eval_target_seq,
                    'target_seq_length': eval_target_seq_length
                }
                eval_loss = model.eval(sess,eval_batch)
                print(str(step) + " eval loss : " + str(eval_loss))



if __name__ == "__main__":
    config, unparsed = get_config()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
