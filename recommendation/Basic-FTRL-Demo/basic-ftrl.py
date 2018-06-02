from datetime import datetime
from csv import DictReader

from math import exp,log,sqrt

train = 'data/train.csv'
test = 'data/test.csv'


alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
D = 2 ** 20             # number of weights to use
interaction = False     # whether to enable poly2 feature interactions

# D, training/validation
epoch = 1       # learn training data for N passes
holdafter = 29   # data after date N (exclusive) are used as validation
holdout = None  # use every N training instance for holdout validation


class ftrl_proximal(object):
    """
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''
    """
    pass


def data(path,D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    for t,row in enumerate(DictReader(open(path))):
        ID = row['id']
        del row['id']

        y = 0.


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()
learner = ftrl_proximal(alpha,beta,L1,L2,D,interaction)

for e in range(epoch):
    loss = 0.
    count = 0

    for t,date,ID,x,y in data(train,D):


    
    
    