import scipy.sparse as sp
import numpy as np

ITEM_CLIP = 300

class Dataset(object):

    def __init__(self,path):
        self.trainMatrix = self.load_training_file_as_matrix(path + ".train.rating")
        self.trainList = self.load_training_file_as_list(path + ".train.rating")
        self.testRatings = self.load_rating_file_as_list(path + '.test.rating')
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        self.num_users, self.num_items = self.trainMatrix.shape



    def load_negative_file(self,filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "": # 一行是一个用户所有的neg
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_list(self,filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList



    def load_training_file_as_list(self,filename):
        u_ = 0
        lists, items = [], [] # 训练数据是按用户id排序过的
        with open(filename, "r") as f:
            line = f.readline()
            index = 0
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                if u_ < u:
                    index = 0
                    lists.append(items) # 每次的items是一个用户所有打过分的item
                    items = []
                    u_ += 1
                index += 1
                if index < ITEM_CLIP:
                    items.append(i)
                line = f.readline()
        lists.append(items)
        print("already load the trainList...")
        return lists




    def load_training_file_as_matrix(self,filename):

        num_users,num_items = 0,0
        with open(filename,"r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")

                u,i = int(arr[0]),int(arr[1])

                num_users = max(num_users,u)
                num_items = max(num_items,i)
                line = f.readline()


        mat = sp.dok_matrix((num_users+1,num_items+1),dtype=np.float32)
        with open(filename,"r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user,item,rating = int(arr[0]),int(arr[1]),float(arr[2])

                if rating > 0:
                    mat[user,item] = 1.0
                line = f.readline()

        print("already load the trainMatrix...")
        return mat