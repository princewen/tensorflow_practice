# Neural Combinatorial Optimization in Tensorflow

TensorFlow implementation of [Neural Combinatorial Optimization with Reinforcement Learning](http://arxiv.org/abs/1611.09940).

![model](./assets/model.png)

(in progress)


## Requirements

- Python 2.7
- [tqdm](httsp://github.com/tqdm/tqdm)
- [TensorFlow 0.12.1](httsp://github.com/tensorflow/tensorflow/tree/r0.12)


## Usage

To train a model:

    $ python main.py --task=tsp20 --lr_start=0.001 --min_data_length=5 --max_data_length=20
    $ python main.py --task=tsp50 --lr_start=0.001 --min_data_length=5 --max_data_length=50
    $ python main.py --task=tsp100 --lr_start=0.0001 --min_data_length=5 --max_data_length=100


To train a model:

    $ python main.py
    $ tensorboard --logdir=logs --host=0.0.0.0

To test a model:

    $ python main.py --is_train=False

## Results

(in progress)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io)
