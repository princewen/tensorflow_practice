import tensorflow as tf
import argparse
from itertools import chain
import json
import pickle
import os
import logging


def add_to_collection(names, values):
    """
    Adds multiple elements to a given collection(s)

    :param names: str or list of collections
    :param values: tensor or list of tensors to add to collection
    """
    if isinstance(names, str):
        names = [names]
    if isinstance(values, str):
        values = [values]
    for name in names:
        for value in values:
            tf.add_to_collection(name, value)


class GraphKeys(object):
    """
    Custom GraphKeys; primarily to be backwards compatable incase tensorflow
    changes it. Also to add my own names

    https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/python/framework/ops.py#L3921
    """
    TRAINABLE_VARIABLES = "trainable_variables"
    PLACEHOLDER = 'placeholder'
    PREDICTION = 'prediction'
    ATTTENTION = 'attention'
    TRAIN_OP = 'train_op'
    EVAL_STEP = 'eval_step'
    LOSSES = 'losses'
    WEIGHTS = 'weights'
    BIASES = 'biases'
    REG_WEIGHTS = 'reg_weights'
    USER_WEIGHTS = 'user_weights'
    ITEM_WEIGHTS = 'item_weights'
    GRADIENTS = 'gradients'

    # Regularization l1/l2 Penalty that would be added
    LOSS_REG = 'regularization_losses'

    # Loss Value without Penalty
    LOSS_NO_REG = 'loss'

    # Keys for the activation of a layer
    ACTIVATIONS = 'activations'

    # Keys for prior to applying the activation function of a layer
    PRE_ACTIVATIONS = 'pre_activations'

    SUMMARIES = 'summaries'
    METRIC_UPDATE = 'metric_update'
    METRIC = 'metric'
    TRAIN = 'train_op'


# List of optimizer classes mappings
OPTIMIZER = {
    # learning_rate=0.001, beta1=0.9, beta2=0.999
    'adam': tf.train.AdamOptimizer,

    # Lazy Adam only updates momentum estimators on values used; it may cause
    # different results than adam
    'lazyadam': tf.contrib.opt.LazyAdamOptimizer,

    # learning_rate, initial_accumulator_value=0.1
    'adagrad': tf.train.AdagradOptimizer,

    # learning_rate, decay=0.9, momentum=0.0
    'rmsprop': tf.train.RMSPropOptimizer,

    # learning_rate, momentum,  use_nesterov=False
    'momentum': tf.train.MomentumOptimizer,

    # learning_rate=0.001, rho=0.95, epsilon=1e-08
    'adadelta': tf.train.AdadeltaOptimizer,

    'sgd': tf.train.GradientDescentOptimizer,
}

# Hyperparameters for various optimizers
# learning_rate is for all
_optimizer_args = {
    'adam': ['beta1', 'beta2', 'epsilon'],
    'lazyadam': ['beta1', 'beta2', 'epsilon'],
    'momentum': ['momentum', 'use_nesterov'],
    'rmsprop': ['momentum', 'decay'],
    'adadelta': ['rho']
}


def get_optimizer_argparse():
    """
    Get arguments for our blocks optimizer
    """
    parser = argparse.ArgumentParser(add_help=False)

    optimizer_group = parser.add_argument_group('OPTIMIZATION',
                                                description='Hyperparameters')

    optimizer_group.add_argument('--optimizer', default='adam', help='SGD optimizer',
                                 choices=OPTIMIZER.keys())

    optimizer_group.add_argument('--learning_rate', default=0.001, type=float,
                                 help='learning rate [All]')

    optimizer_group.add_argument('--momentum', default=0.9, type=float,
                                 help='Momentum value [Momentum/RMSProp]')

    optimizer_group.add_argument('--use_nesterov', default=False, action='store_true',
                                 help='Use nesterov momentum [Momentum]')

    optimizer_group.add_argument('--beta1', default=0.9, type=float,
                                 help='beta 1 hyperparameter [Adam]')

    optimizer_group.add_argument('--beta2', default=0.999, type=float,
                                 help='beta 1 hyperparameter [Adam]')

    optimizer_group.add_argument('--epsilon', default=1e-08, type=float,
                                 help='Epsilon for numerical stability [Adam]')

    optimizer_group.add_argument('--decay', default=0.9, type=float,
                                 help='decay rate hyperparameter [RMSProp]')

    optimizer_group.add_argument('--rho', default=0.95, type=float,
                                 help='rho hyperparameter [Adadelta]')
    return parser


def _preprocess_args(parsed_obj, remove_attrs, keep_attrs, keyname):
    """
    Note modifies inplace. Removes the attributes from a given class object and
    consolidates list of keep_attrs to a single dictionary and sets the
    attribute in the object with keyname.

    :param parsed_obj: object to access via attributes
    :param remove_attrs: iterable of keys of attributes to remove
    :param keep_attrs: iterable of keys to add to a dict and add keyname in
                       namespace
    :param keyname: str, name of key to add keep_attrs to as a dict
    """
    args = {attr: getattr(parsed_obj, attr) for attr in keep_attrs}
    setattr(parsed_obj, keyname, args)

    for attr in remove_attrs:
        delattr(parsed_obj, attr)

def preprocess_args(FLAGS):
    _preprocess_args(FLAGS, set(list(chain.from_iterable(_optimizer_args.values()))),
                     _optimizer_args[FLAGS.optimizer], 'optimizer_params')


class BaseConfig(object):

    save_directory = None
    _IGNORE = ['fields', 'save', 'load']

    # Set Custom Parameters by name with init
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def fields(self):
        """
        Get all fields/properties stored in this config class
        """
        return [m for m in dir(self)
                if not m.startswith('_') and m not in self._IGNORE]

    def save(self):
        """
        Config is dumped as a json file
        """
        json.dump(self._get_dict(),
                  open('%s/config.json' % self.save_directory, 'w'),
                  sort_keys=True, indent=2)
        pickle.dump({key: self.__getattribute__(key) for key in self.fields},
                    open('%s/config.pkl' % self.save_directory, 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    def load(self):
        """
        Load config, equivalent to loading json and updating this classes' dict
        """
        try:
            d = pickle.load(open('%s/config.pkl' % self.save_directory))
            self.__dict__.update(d)
        except Exception:
            d = json.load(open('%s/config.json' % self.save_directory))
            self.__dict__.update(d)

    def _get_dict(self):
        return {key: self.__getattribute__(key) if isinstance(self.__getattribute__(key), (int, float))
                else str(self.__getattribute__(key)) for key in self.fields}

    def __repr__(self):
        return json.dumps(self._get_dict(), sort_keys=True, indent=2)

    def __str__(self):
        return json.dumps(self._get_dict(), sort_keys=True, indent=2)


def create_exp_directory(cwd=''):
    '''
    Creates a new directory to store experiment to save data

    Folders: XXX, creates directory sequentially

    Returns
    -------
    exp_dir : str
        The newly created experiment directory

    '''
    created = False
    for i in range(1, 10000):
        exp_dir = str(i).zfill(3)
        path = os.path.join(cwd, exp_dir)
        if not os.path.exists(path):
            # Create directory
            os.mkdir(path)
            created = True
            break
    if not created:
        print('Could not create directory for experiments')
        exit(-1)
    return path + '/'

def get_logging_config(save_directory):
    # Setup Logging
    return dict(
        version=1,
        formatters={
            # For files
            'detailed': {
                'format': "[%(asctime)s - %(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
            },
            # For the console
            'console': {
                'format':"[%(levelname)s:%(name)s]<%(funcName)s>:%(lineno)d: %(message)s",
            }
        },
        handlers={
            'console': {
                'class': 'logging.StreamHandler',
                'level': logging.INFO,
                'formatter': 'console',
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': logging.DEBUG,
                'formatter': 'detailed',
                'filename': "{}/log".format(save_directory),
                'mode': 'a',
                'maxBytes': 10485760,  # 10 MB
                'backupCount': 5
            }
        },
        loggers={
            'tensorflow': {
                'level': logging.INFO,
                'handlers': ['console', 'file'],
            }
        },
        disable_existing_loggers=False,
    )