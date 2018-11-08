import argparse

from data_loader import load_data
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default='news/train.txt', help='path to the training file')
parser.add_argument('--test_file', type=str, default='news/test.txt', help='path to the test file')
parser.add_argument('--transform', type=bool, default=True, help='whether to transform entity embeddings')
parser.add_argument('--use_context', type=bool, default=False, help='whether to use context embeddings')
parser.add_argument('--max_click_history', type=int, default=30, help='number of sampled click history for each user')
parser.add_argument('--n_filters', type=int, default=128, help='number of filters for each size in KCNN')
parser.add_argument('--filter_sizes', type=int, default=[1, 2], nargs='+',
                    help='list of filter sizes, e.g., --filter_sizes 2 3')
parser.add_argument('--l2_weight', type=float, default=0.01, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=128, help='number of samples in one batch')
parser.add_argument('--n_epochs', type=int, default=10, help='number of training epochs')
parser.add_argument('--KGE', type=str, default='TransE',
                    help='knowledge graph embedding method, please ensure that the specified input file exists')
parser.add_argument('--entity_dim', type=int, default=50,
                    help='dimension of entity embeddings, please ensure that the specified input file exists')
parser.add_argument('--word_dim', type=int, default=50,
                    help='dimension of word embeddings, please ensure that the specified input file exists')
parser.add_argument('--max_title_length', type=int, default=10,
                    help='maximum length of news titles, should be in accordance with the input datasets')
args = parser.parse_args()


train_data, test_data = load_data(args)
train(args, train_data, test_data)

