import numpy as np
import json
import pickle as pkl
import random
import gzip


def load_dict(filename):
    try:
        with open(filename,'r') as f:
            return json.load(f)
    except:
        with open(filename,'rb') as f:
            return pkl.load(f)

def fopen(filename,mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename,mode)
    return open(filename,mode)


class DataIterator:

    def __init__(self,source,
                 uid_voc,
                 mid_voc,
                 cat_voc,
                 batch_size = 128,
                 maxlen = 100,
                 skip_empty = False,
                 sort_by_length = True,
                 max_batch_size = 20,
                 minlen = None):
        self.source = fopen(source,'r')

        self.source_dicts = []
        for source_dict in [uid_voc,mid_voc,cat_voc]:
            self.source_dicts.append(load_dict(source_dict))

        item_info = open("data/item-info",'r')
        meta_map = {}
        for line in item_info:
            arr = line.strip().split("\t")
            if arr[0] not in meta_map:
                meta_map[arr[0]] = arr[1] # meta对应的类别map

        self.meta_id_map = {}
        for key in meta_map:
            val = meta_map[key]
            if key in self.source_dicts[1]:
                mid_idx = self.source_dicts[1][key]
            else:
                mid_idx = 0
            if val in self.source_dicts[2]:
                cat_idx = self.source_dicts[2][val]
            else:
                cat_idx = 0
            self.meta_id_map[mid_idx] = cat_idx # 得到meta id对应的cat id


        review_info = open("data/reviews-info",'r')
        self.mid_list_for_random = []
        for line in review_info:
            arr = line.strip().split("\t")
            tmp_idx = 0
            if arr[1] in self.source_dicts[1]:
                tmp_idx = self.source_dicts[1][arr[1]]
            self.mid_list_for_random.append(tmp_idx)

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_uid = len(self.source_dicts[0])
        self.n_mid = len(self.source_dicts[1])
        self.n_cat = len(self.source_dicts[2])

        self.sort_by_length = sort_by_length
        self.source_buffer = []
        self.k = batch_size * max_batch_size
        self.end_of_data = False

    def get_n(self):
        return self.n_uid, self.n_mid, self.n_cat

    def reset(self):
        self.source.seek(0)

    def __iter__(self):
        return self

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip("\n").split('\t'))

            if self.sort_by_length:
                his_length = np.array([len(s[4].split(" ")) for s in self.source_buffer])
                tidx = his_length.argsort()
                _sbuf = [self.source_buffer[i] for i in tidx]
                self.source_buffer = _sbuf
            else:
                self.source_buffer.reverse()


        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        try:
            while True:

                # 将数据都转换为id
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break

                uid = self.source_dicts[0][ss[1]] if ss[1] in self.source_dicts[0] else 0
                mid = self.source_dicts[1][ss[2]] if ss[2] in self.source_dicts[1] else 0
                cat = self.source_dicts[2][ss[3]] if ss[3] in self.source_dicts[2] else 0

                tmp = []
                for fea in ss[4].split(""):
                    m = self.source_dicts[1][fea] if fea in self.source_dicts[1] else 0
                    tmp.append(m)
                mid_list = tmp

                tmp1 = []
                for fea in ss[5].split(""):
                    c = self.source_dicts[2][fea] if fea in self.source_dicts[2] else 0
                    tmp1.append(c)
                cat_list = tmp1

                # 如果小于规定的最小历史记录长度，直接过滤
                if self.minlen != None:
                    if len(mid_list) >= self.minlen:
                        continue
                if self.skip_empty and (not mid_list):
                    continue

                noclk_mid_list = []
                noclk_cat_list = []

                for pos_mid in mid_list:
                    noclk_tmp_mid = []
                    noclk_tmp_cat = []
                    noclk_index = 0

                    while True:
                        noclk_mid_indx = random.randint(0, len(self.mid_list_for_random) - 1)
                        noclk_mid = self.mid_list_for_random[noclk_mid_indx]
                        if noclk_mid == pos_mid:
                            continue
                        noclk_tmp_mid.append(noclk_mid)
                        noclk_tmp_cat.append(self.meta_id_map[noclk_mid])
                        noclk_index += 1
                        if noclk_index >= 5:
                            break

                    noclk_mid_list.append(noclk_tmp_mid)
                    noclk_cat_list.append(noclk_tmp_cat)

                source.append([uid,mid,cat,mid_list,cat_list,noclk_mid_list,noclk_cat_list])
                target.append([float(ss[0]), 1 - float(ss[0])])

                if len(source) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.__next__()


        return source,target



















