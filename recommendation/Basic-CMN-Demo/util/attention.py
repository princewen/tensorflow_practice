
import tensorflow as tf
import sonnet as snt
from .helper import GraphKeys
from collections import namedtuple
import numpy as np

AttentionOutput = namedtuple("AttentionOutput", ['weight', 'output'])


class MemoryMask(snt.AbstractModule):
    """
    Helper Module to apply a simple memory mask for attention based reads. The
    values beyond the sequence length are set to the smallest possible value we
    can represent with a float32.
    """

    def __init__(self, name='MemoryMask'):
        super(MemoryMask, self).__init__(name=name)

    def _build(self, inputs, mask_length, maxlen=None,):
        """
        Apply a memory mask such that the values we mask result in being the
        minimum possible value we can represent with a float32.

        Taken from Sonnet Attention Module

        :param inputs: [batch size, length], dtype=tf.float32
        :param memory_mask: [batch_size] shape Tensor of ints indicating the
            length of inputs
        :param maxlen: Sets the maximum length of the sequence; if None infered
            from inputs
        :returns: [batch size, length] dim Tensor with the mask applied
        """
        if len(mask_length.shape) != 1:
            raise ValueError('Mask Length must be a 1-d Tensor, got %s' % mask_length.shape)

        # [batch_size, length]
        memory_mask = tf.sequence_mask(mask_length, maxlen=maxlen, name='SequenceMask')
        inputs.shape.assert_is_compatible_with(memory_mask.shape)


        num_remaining_memory_slots = tf.reduce_sum(
            tf.cast(memory_mask, dtype=tf.int32), axis=[1])

        with tf.control_dependencies([tf.assert_positive(
            num_remaining_memory_slots)]):
            # Get the numerical limits of a float
            finfo = np.finfo(np.float32)

            # If True = 1 = Keep that memory slot
            kept_indices = tf.cast(memory_mask, dtype=tf.float32)

            # Inverse
            ignored_indices = tf.cast(tf.logical_not(memory_mask), dtype=tf.float32)

            # If we keep the indices its the max float value else its the
            # minimum float value. Then we can take the minimum
            lower_bound = finfo.max * kept_indices + finfo.min * ignored_indices
            slice_length = tf.reduce_max(mask_length)

            # Return the elementwise
            return tf.minimum(inputs[:, :slice_length],
                              lower_bound[:, :slice_length])


class ApplyAttentionMemory(snt.AbstractModule):

    def __init__(self, name="AttentionMemory"):
        super(ApplyAttentionMemory, self).__init__(name=name)

    def _build(self, memory, output_memory, query, memory_mask=None, maxlen=None):
        """

        :param memory: [batch size, max length, embedding size],
            typically Matrix M
        :param output_memory: [batch size, max length, embedding size],
            typically Matrix C
        :param query: [batch size, embed size], typically u
        :param memory_mask: [batch size] dim Tensor, the length of each
            sequence if variable length
        :param maxlen: int/Tensor, the maximum sequence padding length; if None it
            infers based on the max of memory_mask
        :returns: AttentionOutput
             output: [batch size, embedding size]
             weight: [batch size, max length], the attention weights applied to
                     the output representation.
        """
        memory.shape.assert_has_rank(3)
        output_memory.shape.assert_has_rank(3)

        # query = [batch size, embeddings] => Expand => [batch size, embeddings, 1]
        #         Transpose => [batch size, 1, embeddings]
        query_expanded = tf.transpose(tf.expand_dims(query, -1), [0, 2, 1])

        # Apply batched dot product
        # memory = [batch size, <Max Length>, Embeddings]
        # Broadcast the same memory across each dimension of max length
        # We obtain an attention value for each memory,
        # ie a_0 p_0, a_1 p_1, .. a_n p_n, which equates to the max length
        #    because our query is only 1 dim, we only get attention over memory
        #    for that query. If our query was 2-d then we would obtain a matrix.
        # Return: [batch size, max length]
        scores = tf.reduce_sum(query_expanded * memory, axis=2)

        if memory_mask is not None:
            mask_mod = MemoryMask()
            scores = mask_mod(scores, memory_mask, maxlen)

        # Attention over memories: [Batch Size, <Max Length>]
        attention = tf.nn.softmax(scores, name='Attention')
        tf.add_to_collection(GraphKeys.ATTTENTION, attention)

        # [Batch Size, <Max Length>] => [Batch Size, 1, <Max Length>]
        probs_temp = tf.expand_dims(attention, 1, name='TransformAttention')

        # Output_Memories = [batch size, <Max Length>, Embeddings]
        #       Transpose = [Batch Size, Embedding Size, <Max Length>]
        c_temp = tf.transpose(output_memory, [0, 2, 1],
                              name='TransformOutputMemory')

        # Apply a weighted scalar or attention to the external memory
        # [batch size, 1, <max length>] * [batch size, embedding size, <max length>]
        neighborhood = tf.multiply(c_temp, probs_temp, name='WeightedNeighborhood')

        # Sum the weighted memories together
        # Input:  [batch Size, embedding size, <max length>]
        # Output: [Batch Size, Embedding Size]
        # Weighted output vector
        weighted_output = tf.reduce_sum(neighborhood, axis=2,
                                        name='OutputNeighborhood')

        return AttentionOutput(weight=attention, output=weighted_output)


class VariableLengthMemoryLayer(snt.AbstractModule):

    def __init__(self, hops, embed_size, activation_fn, initializers=None,
                 regularizers=None, name='MemoryLayer', ):
        super(VariableLengthMemoryLayer, self).__init__(name=name)
        self._hops = hops
        self._initializers = initializers
        self._regularizers = regularizers
        self._activation_fn = activation_fn
        self._embed_size = embed_size

    def _build(self, query, memory, output_memory, seq_length, maxlen=32):
        """

        :param query: initial query
        :param memory: internal memory to query
        :param output_memory: external memory to query
        :param seq_length: length of the sequences
        :param maxlen: int, the maximum length over the entire dataset
        :return:
        """
        memory.shape.assert_has_rank(3)
        output_memory.shape.assert_has_rank(3)
        max_length = tf.reduce_max(seq_length)
        # Slice to maximum length
        memory = memory[:, :max_length]
        output_memory = output_memory[:, :max_length]

        user_query, item_query = query
        hop_outputs = []
        query = tf.add(user_query, item_query, name='InitialQuery')

        for hop_k in range(self._hops):  # For each hop
            if hop_k > 0:
                # Apply Mapping
                hop_mapping = snt.Linear(self._embed_size, True,
                                         regularizers=self._regularizers,
                                         initializers=self._initializers,
                                         name='HopMap%s' % hop_k)
                with tf.name_scope('Map'):
                    # z = m_u + e_i
                    # f(Wz + o + b)
                    query = self._activation_fn(hop_mapping(query) + memory_hop.output)
                    tf.add_to_collection(GraphKeys.ACTIVATIONS, query)
                    tf.logging.info('Creating Hop Mapping {} with {}'.format(hop_k+1,
                                                                             self._activation_fn))
            # Apply attention
            hop = ApplyAttentionMemory('AttentionHop%s' % hop_k)

            # [batch size, embedding size]
            memory_hop = hop(memory, output_memory, query, seq_length, maxlen=maxlen)
            hop_outputs.append(memory_hop)

        return hop_outputs
