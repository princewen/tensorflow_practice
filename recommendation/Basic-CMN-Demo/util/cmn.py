import sonnet as snt
import tensorflow as tf
from util.helper import GraphKeys, add_to_collection
from util.layers import LossLayer, OptimizerLayer, ModelBase, DenseLayer
from util.attention import VariableLengthMemoryLayer


class CollaborativeMemoryNetwork(ModelBase):

    def __init__(self, config):
        """

        :param config:
        """
        super(CollaborativeMemoryNetwork, self).__init__(config)

        self._embedding_initializers = {'embeddings': tf.truncated_normal_initializer(stddev=0.01)}
        self._initializers = {
            'w': tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                mode='FAN_IN',
                                                                uniform=False),
        }

        self._hops_init = {
            'w': tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                                mode='FAN_IN',
                                                                uniform=False),
            # Ensure ReLU fires
            'b': tf.constant_initializer(1.0)
        }

        self._output_initializers = {
            'w': tf.contrib.layers.xavier_initializer()
        }

        self._regularizers = {
            'w': tf.contrib.layers.l2_regularizer(config.l2)
        }

        self._construct_placeholders()
        self._construct_weights()
        self._construct()
        # Add summaries
        tf.summary.scalar('Model/Loss', tf.get_collection(GraphKeys.LOSSES)[0])
        tf.summary.scalar('Model/LearningRate', self.learning_rate)

        self.summary = tf.summary.merge_all()

    def _construct(self):
        """
        Construct the model; main part of it goes here
        """
        # our query = m_u + e_i
        query = (self._cur_user, self._cur_item)
        neg_query = (self._cur_user, self._cur_item_negative)

        # Positive
        neighbor = self._mem_layer(query,
                                   self.user_memory(self.input_neighborhoods),
                                   self.user_output(self.input_neighborhoods),
                                   self.input_neighborhood_lengths,
                                   self.config.max_neighbors)[-1].output
        self.score = self._output_module(tf.concat([self._cur_user * self._cur_item,
                                                    neighbor], axis=1))

        # Negative
        neighbor_negative = self._mem_layer(neg_query,
                                            self.user_memory(self.input_neighborhoods_negative),
                                            self.user_output(self.input_neighborhoods_negative),
                                            self.input_neighborhood_lengths_negative,
                                            self.config.max_neighbors)[-1].output
        negative_output = self._output_module(tf.concat(
            [self._cur_user * self._cur_item_negative, neighbor_negative], axis=1))

        # Loss and Optimizer
        self.loss = LossLayer()(self.score, negative_output)
        self._optimizer = OptimizerLayer(self.config.optimizer, clip=self.config.grad_clip,
                                         params=self.config.optimizer_params)
        self.train = self._optimizer(self.loss)

        tf.add_to_collection(GraphKeys.PREDICTION, self.score)

    def _construct_placeholders(self):
        """Create placeholders for our model"""
        self.input_users = tf.placeholder(tf.int32, [None], 'UserID')
        self.input_items = tf.placeholder(tf.int32, [None], 'ItemID')
        self.input_items_negative = tf.placeholder(tf.int32, [None],
                                                   'NegativeItemID')
        self.input_neighborhoods = tf.placeholder(tf.int32, [None, None],
                                                  'Neighborhood')

        self.input_neighborhood_lengths = tf.placeholder(tf.int32, [None],
                                                         'NeighborhoodLengthID')

        self.input_neighborhoods_negative = tf.placeholder(tf.int32,
                                                           [None, None],
                                                           'NeighborhoodNeg')

        self.input_neighborhood_lengths_negative = tf.placeholder(tf.int32,
                                                                  [None],
                                                                  'NeighborhoodLengthIDNeg')
        # Add our placeholders
        add_to_collection(GraphKeys.PLACEHOLDER, [self.input_users,
                                                  self.input_items,
                                                  self.input_items_negative,
                                                  self.input_neighborhoods,
                                                  self.input_neighborhood_lengths,
                                                  self.input_neighborhoods_negative,
                                                  self.input_neighborhood_lengths_negative,
                                                  self.dropout])

    def _construct_weights(self):
        """
        Constructs the user/item memories and user/item external memory/outputs

        Also add the embedding lookups
        """
        self.user_memory = snt.Embed(self.config.user_count, self.config.embed_size,
                                     initializers=self._embedding_initializers,
                                     name='MemoryEmbed')

        self.user_output = snt.Embed(self.config.user_count, self.config.embed_size,
                                     initializers=self._embedding_initializers,
                                     name='MemoryOutput')

        self.item_memory = snt.Embed(self.config.item_count,
                                     self.config.embed_size,
                                     initializers=self._embedding_initializers,
                                     name="ItemMemory")
        self._mem_layer = VariableLengthMemoryLayer(self.config.hops,
                                                    self.config.embed_size,
                                                    tf.nn.relu,
                                                    initializers=self._hops_init,
                                                    regularizers=self._regularizers,
                                                    name='UserMemoryLayer')

        self._output_module = snt.Sequential([
            DenseLayer(self.config.embed_size, True, tf.nn.relu,
                       initializers=self._initializers,
                       regularizers=self._regularizers,
                       name='Layer'),
            snt.Linear(1, False,
                       initializers=self._output_initializers,
                       regularizers=self._regularizers,
                       name='OutputVector'),
            tf.squeeze])

        # [batch, embedding size]
        self._cur_user = self.user_memory(self.input_users)
        self._cur_user_output = self.user_output(self.input_users)

        # Item memories a query
        self._cur_item = self.item_memory(self.input_items)
        self._cur_item_negative = self.item_memory(self.input_items_negative)

        # Share Embeddings
        self._cur_item_output = self._cur_item
        self._cur_item_output_negative = self._cur_item_negative


