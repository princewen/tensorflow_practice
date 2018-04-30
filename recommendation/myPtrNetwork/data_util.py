import os
import re
import zipfile
import itertools
import threading
import numpy as np
from tqdm import trange, tqdm
from collections import namedtuple
import tensorflow as tf




def read_paper_dataset(path):
    enc_seq , target_seq , enc_seq_length , target_seq_length = [],[],[],[]
    tf.logging.info("Read dataset {} which is used in the paper..".format(path))
    length = max(re.findall('\d+', path))
    with open(path,'r') as f:
        for l in tqdm(f):
            # 使用output分割数据
            inputs, outputs = l.split(' output ')
            inputs = np.array(inputs.split(), dtype=np.float32).reshape([-1, 2])
            outputs = np.array(outputs.split(), dtype=np.int32)[:-1]
            # x分割成横纵坐标两列
            enc_seq.append(inputs)
            # y要忽略最后一个
            target_seq.append(outputs)  # skip the last one
            enc_seq_length.append(inputs.shape[0])
            target_seq_length.append(outputs.shape[0])
    return enc_seq,target_seq,enc_seq_length,target_seq_length


def gen_data(path):
    x,y,enc_seq_length,target_seq_length = read_paper_dataset(path)
    # max_length 是 config的序列的最大长度
    enc_seq = np.zeros([len(x), 10, 2], dtype=np.float32)
    target_seq = np.zeros([len(y), 10], dtype=np.int32)

    # 这里的作用就是将所有的输入都变成同样长度，用0补齐
    for idx, (nodes, res) in enumerate(tqdm(zip(x, y))):
        enc_seq[idx, :len(nodes)] = nodes
        target_seq[idx, :len(res)] = res
    return enc_seq,target_seq,enc_seq_length,target_seq_length
