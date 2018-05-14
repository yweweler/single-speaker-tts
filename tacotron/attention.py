import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import LuongAttention, \
    AttentionWrapper, AttentionWrapperState


class AttentionMode:
    """
    Enumerator for the Luong style local attention modes.

    - See [1]: Effective Approaches to Attention-based Neural Machine Translation,
        http://arxiv.org/abs/1508.04025
    """
    # local-m mode.
    MONOTONIC = 'monotonic'

    # local-p mode.
    PREDICTIVE = 'predictive'


class AttentionScore:
    """
    Enumerator for the three different content-based scoring functions for Luong style attention.

    - See [1]: Effective Approaches to Attention-based Neural Machine Translation,
        http://arxiv.org/abs/1508.04025
    """
    DOT = 'dot'
    GENERAL = 'general'
    CONCAT = 'concat'


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
    print('Overwritten `tensorflow.contrib.seq2seq.python.ops.attention_wrapper'
          '._compute_attention`')

    """Computes the attention and alignments for a given attention_mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)

    window_start = attention_mechanism.window_start
    window_stop = attention_mechanism.window_stop

    pre_padding = attention_mechanism.window_pre_padding
    post_padding = attention_mechanism.window_post_padding

    full_pre_padding = attention_mechanism.full_seq_pre_padding
    full_post_padding = attention_mechanism.full_seq_post_padding

    def __process_entry(i):
        value_window = attention_mechanism.values[i, window_start[i][0]:window_stop[i][0], :]
        value_window_paddings = [
            [pre_padding[i][0], post_padding[i][0]],
            [0, 0]
        ]
        value_window = tf.pad(value_window, value_window_paddings, 'CONSTANT')
        value_window.set_shape((attention_mechanism.window_size, 256))

        context_window = tf.matmul(expanded_alignments[i], value_window)

        alignment_seq_paddings = [
            [full_pre_padding[i][0], full_post_padding[i][0]],
        ]

        # point_dist = tf.cast(tf.range(start=window_start[i][0],
        #                               limit=window_stop[i][0],
        #                               delta=1), dtype=tf.float32) - p[i][0]

        # gaussian_weights = tf.exp(-(point_dist ** 2) / 2 * (d / 2) ** 2)

        __alignments = tf.pad(alignments[i], alignment_seq_paddings, 'CONSTANT')

        return context_window, __alignments

    tmp_data = tf.map_fn(
        __process_entry,
        tf.range(start=0, limit=attention_mechanism.batch_size, delta=1, dtype=tf.int32),
        dtype=(tf.float32, tf.float32),
        parallel_iterations=32)

    context = tmp_data[0]
    context = tf.Print(context, [tf.shape(context)], '_compute_attention context.matmul:')

    context = tf.squeeze(context, [1])
    context = tf.Print(context, [tf.shape(context)], '_compute_attention context.squeeze:')

    padded_alignment = tmp_data[1]
    padded_alignment = tf.Print(padded_alignment, [tf.shape(padded_alignment)],
                                '_compute_attention padded_alignments:')

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, padded_alignment, padded_alignment


# TODO: Dirty hack to override tensorflow's internal _compute_attention implementation.
attention_wrapper._compute_attention = _compute_attention


class LocalLuongAttention(LuongAttention):
    """
     Implements a Luong-style local attention mechanism.

     This implementation supports both monotonic attention as well as predictive attention.

    - See [1]: Effective Approaches to Attention-based Neural Machine Translation,
        http://arxiv.org/abs/1508.04025
    """

    def __init__(self, num_units,
                 memory,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="LocalLuongAttention",
                 d=10,
                 attention_mode=AttentionMode.MONOTONIC,
                 score_mode=AttentionScore.DOT):
        super().__init__(num_units=num_units,
                         memory=memory,
                         memory_sequence_length=memory_sequence_length,
                         scale=scale,
                         probability_fn=probability_fn,
                         score_mask_value=score_mask_value,
                         dtype=dtype,
                         name=name)

        # Initialize the decoding time counter.
        # This variable is updated by the `ÀdvancedAttentionWrapper`.
        self.time = 0

        # Calculate the attention window size.
        self.d = d
        self.window_size = 2 * self.d + 1

        # Store the attention mode.
        self.attention_mode = attention_mode

        # Store the scoring function style to be used.
        self.score_mode = score_mode

    def __call__(self, query, state):
        """
        TODO: Update docstring.
        Arguments:
            query:
            state:

        Returns:

        """
        with tf.variable_scope(None, "local_luong_attention", [query]) as test:
            # Get the depth of the memory values.
            num_units = self._keys.get_shape()[-1]

            # Get the source sequence length from memory.
            source_seq_length = tf.shape(self._keys)[1]

            # Predict p_t ==========================================================================
            vp = tf.get_variable(name="local_v_p", shape=[num_units, 1], dtype=tf.float32)
            wp = tf.get_variable(name="local_w_p", shape=[num_units, num_units], dtype=tf.float32)

            # shape => (B, num_units)
            _intermediate_result = tf.transpose(tf.tensordot(wp, query, [0, 1]))

            # shape => (B, 1)
            _tmp = tf.transpose(tf.tensordot(vp, tf.tanh(_intermediate_result), [0, 1]))

            _intermediate_prob = tf.sigmoid(_tmp)

            # p_t as described by Luong for the predictive local-p case.
            # self.p = tf.cast(source_seq_length, tf.float32) * _intermediate_prob
            # self.p = tf.Print(self.p, [self.p], 'LocalAttention self.p:', summarize=99)

            # p_t as described by Luong for the predictive local-m case.
            self.p = tf.tile(
                [[self.time]],
                tf.convert_to_tensor([self.batch_size, 1])
            )

            self.p = tf.maximum(self.p, self.d)
            self.p = tf.minimum(self.p, source_seq_length - (self.d + 1))

            self.p = tf.cast(self.p, dtype=tf.float32)
            # ======================================================================================

            start_index = tf.cast(self.p - self.d, dtype=tf.int32)
            self.window_start = tf.maximum(0, start_index)

            stop_index = tf.cast(self.p + self.d + 1, dtype=tf.int32)
            self.window_stop = tf.minimum(source_seq_length, stop_index)

            self.full_seq_pre_padding = tf.abs(start_index)
            self.full_seq_post_padding = tf.abs(stop_index - source_seq_length)

            self.window_pre_padding = tf.abs(self.window_start - start_index)
            self.window_post_padding = tf.abs(self.window_stop - stop_index)

            def __process_entry(i):
                __window = self._keys[i, self.window_start[i][0]:self.window_stop[i][0], :]

                paddings = [
                    [self.window_pre_padding[i][0], self.window_post_padding[i][0]],
                    [0, 0]
                ]
                return tf.pad(__window, paddings, 'CONSTANT')

            window = tf.map_fn(
                __process_entry,
                tf.range(start=0, limit=self.batch_size, delta=1, dtype=tf.int32),
                dtype=(tf.float32),
                parallel_iterations=32)

            score = _luong_dot_score(query, window, self._scale)

        score = tf.Print(score, [tf.shape(window)], 'LocalAttention window:', summarize=99)
        score = tf.Print(score, [tf.shape(self._keys)], 'LocalAttention _keys:')

        alignments = self._probability_fn(score, state)
        next_state = alignments

        alignments = tf.Print(alignments, [tf.shape(alignments)], 'LocalAttention alignments:')

        return alignments, next_state


def _luong_dot_score(query, keys, scale):
    """
    Implements the Luong-style dot scoring function.

    This attention has two forms. The first is standard Luong attention, as described in:

    Minh-Thang Luong, Hieu Pham, Christopher D. Manning.
    "Effective Approaches to Attention-based Neural Machine Translation."
    EMNLP 2015.  https://arxiv.org/abs/1508.04025

    The second is the scaled form inspired partly by the normalized form of
    Bahdanau attention.

    To enable the second form, call this function with `scale=True`.

    This implementation is derived from: `tensorflow.contrib.seq2seq.python.ops.attention_wrapper`

    Arguments:
        query (tf.Tensor):
            Decoder cell outputs to compare to the keys (memory).
            The shape is expected to be shape=(B, num_units) with B being the batch size
            and `num_units` being the output size of the decoder_cell.

        keys (tf.Tensor):
            Processed memory (usually the encoder states processed by the memory_layer).
            The shape is expected to be shape=(B, X, num_units) with B being the batch size
            and `num_units` being the output size of the memory_layer. X may be the
            maximal length of the encoder time domain or in the case of local attention the
            window size.

        scale (boolean):
            Whether to apply a scale to the score function.

    Returns:
        score (tf.Tensor):
            A tensor with shape=(B, X) containing the non-normalized score values.

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

    query = tf.expand_dims(query, 1)
    score = tf.matmul(query, keys, transpose_b=True)
    score = tf.squeeze(score, [1])

    if scale:
        # Scalar used in weight scaling
        g = tf.get_variable(
            "attention_g", dtype=dtype,
            initializer=tf.ones_initializer, shape=())
        score = g * score

    return score


def _luong_general_score(query, keys):
    """
    Implements the Luong-style general scoring function.

    - See [1]: Effective Approaches to Attention-based Neural Machine Translation,
        http://arxiv.org/abs/1508.04025

    Arguments:
        query (tf.Tensor):
            Decoder cell outputs to compare to the keys (memory).
            The shape is expected to be shape=(B, num_units) with B being the batch size
            and `num_units` being the output size of the decoder_cell.

        keys (tf.Tensor):
            Processed memory (usually the encoder states processed by the memory_layer).
            The shape is expected to be shape=(B, X, num_units) with B being the batch size
            and `num_units` being the output size of the memory_layer. X may be the
            maximal length of the encoder time domain or in the case of local attention the
            window size.

    Returns:
        score (tf.Tensor):
            A tensor with shape=(B, X) containing the non-normalized score values.
    """
    raise NotImplementedError('Luong style general mode attention scoring is not implemented yet!')


def _luong_concat_score(query, keys):
    """
    Implements the Luong-style concat scoring function.

    - See [1]: Effective Approaches to Attention-based Neural Machine Translation,
        http://arxiv.org/abs/1508.04025

    Arguments:
        query (tf.Tensor):
            Decoder cell outputs to compare to the keys (memory).
            The shape is expected to be shape=(B, num_units) with B being the batch size
            and `num_units` being the output size of the decoder_cell.

        keys (tf.Tensor):
            Processed memory (usually the encoder states processed by the memory_layer).
            The shape is expected to be shape=(B, X, num_units) with B being the batch size
            and `num_units` being the output size of the memory_layer. X may be the
            maximal length of the encoder time domain or in the case of local attention the
            window size.

    Returns:
        score (tf.Tensor):
            A tensor with shape=(B, X) containing the non-normalized score values.

    """
    raise NotImplementedError('Luong style concat mode attention scoring is not implemented yet!')


class AdvancedAttentionWrapper(AttentionWrapper):
    """
    Wraps the standard AttentionWrapper class so that during decoding steps the decoding time
    index is updated in the attention mechanism.

    This is a hack to enable us using Luong style monotonic attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):

        super().__init__(cell=cell,
                         attention_mechanism=attention_mechanism,
                         attention_layer_size=attention_layer_size,
                         alignment_history=alignment_history,
                         cell_input_fn=cell_input_fn,
                         output_attention=output_attention,
                         initial_cell_state=initial_cell_state,
                         name=name)

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        - Step 1: Mix the `inputs` and previous step's `attention` output via
          `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
          `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
                cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []
        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            # Note: This is the only modification hacked into the attention wrapper to support
            # monotonic Luong attention.
            attention_mechanism.time = state.time

            attention, alignments, next_attention_state = _compute_attention(
                attention_mechanism, cell_output, previous_attention_state[i],
                self._attention_layers[i] if self._attention_layers else None)
            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state
