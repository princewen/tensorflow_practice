import numpy

class SumTree():
    write = 0
    count = 0

    def __init__(self,capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2 * capacity - 1)
        self.data = numpy.zeros( capacity ,dtype = object)


    def _propagate(self,idx,change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent!=0:
            self._propagate(parent,change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total_and_count(self):
        return self.tree[0], self.count


    def update(self,idx,p):
        change = p - self.tree[idx] # 得到变化的数值，
        self.tree[idx] = p
        self._propagate(idx,change) # 根据叶子结点的变化，不断修改父节点的值

    def add(self,p,data):
        idx = self.write + self.capacity - 1 # 得到叶子结点在树中的位置
        self.data[self.write] = data
        self.update(idx,p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.count < self.capacity:
            self.count += 1

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])