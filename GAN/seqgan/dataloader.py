import numpy as np

class Gen_Data_loader():
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.token_stream = []


    def create_batches(self,data_file):
        self.token_stream = []
        with open(data_file,'r') as f:
            for line in f:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    self.token_stream.append(parse_line)


        self.num_batch = int(len(self.token_stream) / self.batch_size)
        # 截取刚刚好的batch
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]
        # 使用np的split函数切分batch
        self.sequence_batch = np.split(np.array(self.token_stream),self.num_batch,0)
        self.pointer= 0


    def next_batch(self):
        ret = self.sequence_batch[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0


class Dis_dataloader():
    def __init__(self,batch_size):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])

    def load_train_data(self,positive_file,negative_file):
        positive_examples = []
        negative_examples = []
        with open(positive_file) as fin:
            for line in fin:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                positive_examples.append(parse_line)


        with open(negative_file) as fin:
            for line in fin:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == 20:
                    negative_examples.append(parse_line)

        self.sentences = np.array(positive_examples + negative_examples)

        positive_labels = [[0,1] for _ in positive_examples]
        negative_labels = [[1,0] for _ in negative_examples]

        self.labels = np.concatenate([positive_labels,negative_labels],0)

        # shuffle the data
        # 如果传给permutation一个矩阵，它会返回一个洗牌后的矩阵副本；
        # 而shuffle只是对一个矩阵进行洗牌，无返回值。 如果传入一个整数，它会返回一个洗牌后的arange。
        shuffle_indices = np.random.permutation(np.arange(len(self.labels)))
        self.sentences = self.sentences[shuffle_indices]
        self.labels = self.labels[shuffle_indices]

        # split batches
        self.num_batch = int(len(self.labels)/self.batch_size)
        self.sentences = self.sentences[:self.batch_size * self.num_batch]
        self.labels = self.labels[:self.batch_size * self.num_batch]

        self.sentences_batches = np.split(self.sentences,self.num_batch,0)
        self.labels_batches = np.split(self.labels,self.num_batch,0)

        self.pointer = 0

    def next_batch(self):
        ret = self.sentences_batches[self.pointer],self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0
