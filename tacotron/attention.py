import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import LuongAttention
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):

    print('THIS IS A SUUUUPER HACK FOR OVERRIDING SOMEONE OTHERS SHIT!')

    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)
    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.

    value_windows = []
    window_start = attention_mechanism.w_start
    window_stop = attention_mechanism.w_stop

    pre_padding = attention_mechanism.pre_padding
    post_padding = attention_mechanism.post_padding

    full_pre_padding = attention_mechanism.full_seq_pre_padding
    full_post_padding = attention_mechanism.full_seq_post_padding

    # TODO: Super ugly hack, but it works for now.
    for i in range(0, 4):
        tmp_window = attention_mechanism.values[i, window_start[i][0]:window_stop[i][0], :]
        paddings = [
            [pre_padding[i][0], post_padding[i][0]],
            [0, 0]
        ]

        tmp_window = tf.pad(tmp_window, paddings, 'CONSTANT')
        value_windows.append(tmp_window)

    values = tf.stack(value_windows)
    print('values', values)
    values = tf.Print(values, [tf.shape(values)], '_compute_attention values:')

    # TODO: Calculate this inside the loop and add padding afterwards.
    context = tf.matmul(expanded_alignments, values)
    context = tf.Print(context, [tf.shape(context)], '_compute_attention context.matmul:')

    context = tf.squeeze(context, [1])
    context = tf.Print(context, [tf.shape(context)], '_compute_attention context.squeeze:')

    # TODO: Why does the context no longer know it's shape?
    context.set_shape((4, 256))

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    # TODO: This dies not work outside the loop (get this inside the loop)!
    paddings = [
        [0, 0],
        [full_pre_padding[i][0], full_post_padding[i][0]]
    ]

    alignments = tf.pad(alignments, paddings, 'CONSTANT')
    alignments = tf.Print(alignments, [tf.shape(alignments)], '_compute_attention return '
                                                              'alignments:')

    return attention, alignments, next_attention_state


attention_wrapper._compute_attention = _compute_attention


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

    def __call__(self, query, state):
        with tf.variable_scope(None, "local_luong_attention", [query]) as test:
            # Get the memory dtype.
            dtype = self._keys.dtype

            # Get the depth of the memory values.
            num_units = self._keys.get_shape()[-1]

            # Get the current batch size.
            batch_size = self._keys.get_shape()[0]

            # Get the source sequence length from memory.
            seq_length = tf.shape(self._keys)[1]
            f_seq_length = tf.cast(seq_length, tf.float32)

            vp = tf.get_variable(name="local_v_p", shape=[num_units, 1], dtype=tf.float32)
            wp = tf.get_variable(name="local_w_p", shape=[num_units, num_units], dtype=tf.float32)

            # shape => (B, num_units)
            _intermediate_result = tf.transpose(tf.tensordot(wp, query, [0, 1]))

            # shape => (B, 1)
            _tmp = tf.transpose(tf.tensordot(vp, tf.tanh(_intermediate_result), [0, 1]))

            _intermediate_prob = tf.sigmoid(_tmp)

            # p_t as described by Luong for the predictive local-p case.
            p = f_seq_length * _intermediate_prob

            # TODO: Refactor this variables into separate hyper-parameters.
            # Window size is 0.5s backwards from p_t and 0.5s forward from p_t.
            window_size = 2 * 10
            d = window_size // 2

            start_index = tf.cast(p - d, dtype=tf.int32)
            window_start = tf.maximum(0, start_index)

            self.w_start = window_start

            stop_index = tf.cast(p + d + 1, dtype=tf.int32)
            window_stop = tf.minimum(seq_length, stop_index)

            self.w_stop = window_stop

            self.full_seq_pre_padding = tf.abs(start_index)
            self.full_seq_post_padding = tf.abs(stop_index - seq_length)

            self.pre_padding = tf.abs(window_start - start_index)
            self.post_padding = tf.abs(window_stop - stop_index)

            # TODO: Should we pad this so the window size is always consistent?
            # TODO: What changes have to be made in order for the alignment history to work?
            windows = []
            # TODO: Super ugly hack, but it works for now.
            for i in range(0, 4):
                # self._keys = tf.Print(self._keys, [window_start[i][0]],
                #                       '=========================================================\n'
                #                       'LocalAttention window_start[i][0]:')
                # self._keys = tf.Print(self._keys, [window_stop[i][0]],
                #                       'LocalAttention window_stop[i][0]:')

                tmp_window = self._keys[i, window_start[i][0]:window_stop[i][0], :]

                # self._keys = tf.Print(self._keys, [tf.shape(tmp_window)],
                #                       'LocalAttention pre_padding tmp_window[{}]:'.format(i))

                paddings = [
                    [self.pre_padding[i][0], self.post_padding[i][0]],
                    [0, 0]
                ]
                tmp_window = tf.pad(tmp_window, paddings, 'CONSTANT')

                # self._keys = tf.Print(self._keys, [tf.shape(tmp_window)],
                #                       'LocalAttention post_padding tmp_window[{}]:'.format(i))

                windows.append(tmp_window)

            window = tf.stack(windows)

            # window = self._keys[:, window_start[0][0]:window_stop[0][0], :]

            # window = tf.get_variable('attention_window', [batch_size, window_size, num_units])

            # Clear the window.
            # window = tf.assign(window, tf.zeros_like(window))

            # NOTE: It would be less computationally expensive if this would be done in the
            # BaseAttentionMechnaism since we could slice values before applying the dense
            # memory_layer. Therefore the memory_layer would only have to be calculated on the
            # sliced window and not on all encoder outputs just to throw most of them away.
            # score = _local_luong_score(query, self._keys, self._scale)
            score = _local_luong_score(query, window, self._scale)

        # score = tf.Print(score, [tf.shape(window_start)], 'LocalAttention window_start:')
        # score = tf.Print(score, [tf.shape(self._keys)], 'LocalAttention _keys:')
        score = tf.Print(score, [tf.shape(window)], 'LocalAttention window:', summarize=99)
        score = tf.Print(score, [tf.shape(self._keys)], 'LocalAttention _keys:')
        # score = tf.Print(score, [start_index[0][0]], 'LocalAttention start_index:')
        # score = tf.Print(score, [window_start[0][0]], 'LocalAttention window_start:')
        # score = tf.Print(score, [self.pre_padding[0][0]], 'LocalAttention pre_padding:')
        # score = tf.Print(score, [self.post_padding[0][0]], 'LocalAttention post_padding:')

        alignments = self._probability_fn(score, state)
        next_state = alignments

        alignments = tf.Print(alignments, [tf.shape(alignments)], 'LocalAttention alignments:')

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

# http://cnyah.com/2017/08/01/attention-variants/
