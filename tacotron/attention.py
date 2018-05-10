import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import LuongAttention


class LocalLuongAttention(LuongAttention):
    def __init__(self, num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="LocalLuongAttention"):
        # TODO: What about the query_layer in _BaseAttentionMechanism?
        super().__init__(num_units=num_units,
                         memory=memory,
                         memory_sequence_length=memory_sequence_length,
                         scale=scale,
                         probability_fn=probability_fn,
                         score_mask_value=score_mask_value,
                         dtype=dtype,
                         name=name)

        self._intermediate_layer = tf.layers.Dense(num_units, use_bias=False)

    def __call__(self, query, state):
        with tf.variable_scope(None, "local_luong_attention", [query]) as test:
            # Get the memory dtype.
            dtype = self._keys.dtype

            # Get the depth of the memory values.
            num_units = self._keys.get_shape()[-1]

            # Get the source sequence length from memory.
            seq_length = tf.shape(self._keys)[1]
            f_seq_length = tf.cast(seq_length, tf.float32)

            vp = tf.get_variable(name="local_v_p", shape=[num_units, 1], dtype=tf.float32)
            wp = tf.get_variable(name="local_w_p", shape=[num_units, num_units], dtype=tf.float32)

            # shape => (B, num_units)
            _intermediate_result = tf.transpose(tf.tensordot(wp, query, [0, 1]))

            # shape => (B, )
            # _tmp = tf.transpose(vp) * tf.tanh(_intermediate_result)
            _tmp = tf.transpose(tf.tensordot(vp, tf.tanh(_intermediate_result), [0, 1]))

            _intermediate_prob = tf.sigmoid(_tmp)

            # p_t as described by Luong for the predictive local-p case.
            p = f_seq_length * _intermediate_prob

            # TODO: Refactor this variables into separate hyper-parameters.
            # Window size is 0.5s backwards from p_t and 0.5s forward from p_t.
            window_size = 2 * 40
            d = window_size // 2

            window_start = tf.maximum(0, tf.cast(p - d, dtype=tf.int32))
            window_end = tf.minimum(seq_length, tf.cast(p + d, dtype=tf.int32))

            # TODO: Should we pad this so the window size is always consistent?
            # TODO: What changes have to be made in order for the alignment history to work?
            # window = self._keys[:, window_start:window_end, :]

            # TODO: Slice a window [p_t - D, p_t + D] from self._keys.
            # NOTE: It would be less computationally expensive if this would be done in the
            # BaseAttentionMechnaism since we could slice values before applying the dense
            # memory_layer. Therefore the memory_layer would only have to be calculated on the
            # sliced window and not on all encoder outputs just to throw most of them away.
            score = _local_luong_score(query, self._keys, self._scale)

        score = tf.Print(score, [tf.shape(window_start)], 'LocalAttention window_start:')
        score = tf.Print(score, [tf.shape(window_end)], 'LocalAttention window_end:')

        alignments = self._probability_fn(score, state)
        next_state = alignments

        return alignments, next_state


def _local_luong_score(query, keys, scale):
    # TODO: Implement the "location" version ("dot": current, "general" and "concat" are also possible).
    # TODO: In the local version the tensor no longer contains max_time states but only 2D+1 ones.
    """Implements Luong-style (multiplicative) scoring function.

    This attention has two forms.  The first is standard Luong attention,
    as described in:

    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, call this function with `scale=True`.

    Args:
      query: Tensor, shape `[batch_size, num_units]` to compare to keys.
      keys: Processed memory, shape `[batch_size, max_time, num_units]`.
      scale: Whether to apply a scale to the score function.

    Returns:
      A `[batch_size, max_time]` tensor of unnormalized score values.

    Raises:
      ValueError: If `key` and `query` depths do not match.
    """
    depth = query.get_shape()[-1]
    key_units = keys.get_shape()[-1]
    if depth != key_units:
        raise ValueError(
            "Incompatible or unknown inner dimensions between query and keys.  "
            "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
            "Perhaps you need to set num_units to the keys' dimension (%s)?"
            % (query, depth, keys, key_units, key_units))
    dtype = query.dtype

    # Reshape from [batch_size, depth] to [batch_size, 1, depth]
    # for matmul.
    query = tf.expand_dims(query, 1)

    # Inner product along the query units dimension.
    # matmul shapes: query is [batch_size, 1, depth] and
    #                keys is [batch_size, max_time, depth].
    # the inner product is asked to **transpose keys' inner shape** to get a
    # batched matmul on:
    #   [batch_size, 1, depth] . [batch_size, depth, max_time]
    # resulting in an output shape of:
    #   [batch_size, 1, max_time].
    # we then squeeze out the center singleton dimension.
    score = tf.matmul(query, keys, transpose_b=True)
    score = tf.squeeze(score, [1])

    if scale:
        # Scalar used in weight scaling
        g = tf.get_variable(
            "attention_g", dtype=dtype,
            initializer=tf.ones_initializer, shape=())
        score = g * score

    return score
