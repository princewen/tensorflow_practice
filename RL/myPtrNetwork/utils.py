import os
import json
import logging
import numpy as np
from datetime import datetime

import tensorflow as tf
import tensorflow.contrib.slim as slim


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger('tensorflow')

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(tf.logging.INFO)

    if config.load_path:
        if config.load_path.startswith(config.task):
            config.model_name = config.load_path
        else:
            config.model_name = "{}_{}".format(config.task, config.load_path)
    else:
        config.model_name = "{}_{}".format(config.task, get_time())

    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.log_dir, config.data_dir, config.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def save_config(model_dir, config):
    param_path = os.path.join(model_dir, "params.json")

    tf.logging.info("MODEL dir: %s" % model_dir)
    tf.logging.info("PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)
