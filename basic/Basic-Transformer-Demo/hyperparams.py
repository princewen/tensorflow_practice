class Hyperparams:
    '''Hyperparameters'''
    # data
    source_train = 'data/train.tags.de-en.de'
    target_train = 'data/train.tags.de-en.en'
    source_test = 'data/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'data/IWSLT16.TED.tst2014.de-en.en.xml'

    # training
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir'  # log directory

    # model
    maxlen = 10  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    min_cnt = 20  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.

