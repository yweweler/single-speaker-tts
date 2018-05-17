import tensorflow as tf
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


def _luong_local_compute_attention(attention_mechanism, cell_output, attention_state,
                                   attention_layer):
    """Computes the attention and alignments for the Luong style local attention mechanism."""
    alignments, next_attention_state = attention_mechanism(
        cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)

    context_windows = []
    padded_alignment_windows = []

    window_start = attention_mechanism.window_start
    window_stop = attention_mechanism.window_stop

    pre_padding = attention_mechanism.window_pre_padding
    post_padding = attention_mechanism.window_post_padding

    full_pre_padding = attention_mechanism.full_seq_pre_padding
    full_post_padding = attention_mechanism.full_seq_post_padding

    for i in range(0, attention_mechanism.const_batch_size):
        # Slice out the window from the memory.
        value_window = attention_mechanism.values[i, window_start[i][0]:window_stop[i][0], :]

        # Add zero padding to the slice in order to ensure the window size is (2D+1).
        value_window_paddings = [
            [pre_padding[i][0], post_padding[i][0]],
            [0, 0]
        ]
        value_window = tf.pad(value_window, value_window_paddings, 'CONSTANT')

        # Shape information is lost after padding ;(.
        value_window.set_shape((attention_mechanism.window_size,
                                attention_mechanism._num_units))

        # Calculate the context vector for the current batch entry using only information from
        # teh window.
        context_window = tf.matmul(expanded_alignments[i], value_window)
        context_windows.append(context_window)

        if attention_mechanism.force_gaussian is True:
            # Apply gaussian weighting of the window contents.
            point_dist = tf.cast(tf.range(start=window_start[i][0],
                                          limit=window_stop[i][0],
                                          delta=1), dtype=tf.float32) - attention_mechanism.p[i][0]

            gaussian_weights = tf.exp(-(point_dist ** 2) / 2 * (attention_mechanism.d / 2) ** 2)

            __alignments = alignments[i] * gaussian_weights
        else:
            # Use the raw window contents.
            __alignments = alignments[i]

        # Add padding to the alignments to get from the window size 2D+1 up to the original
        # memory length.
        alignment_seq_paddings = [
            [full_pre_padding[i][0], full_post_padding[i][0]],
        ]
        __alignments = tf.pad(__alignments, alignment_seq_paddings, 'CONSTANT')

        padded_alignment_windows.append(__alignments)

    # Stack all context vectors into one tensor.
    context = tf.stack(context_windows)
    # Squeeze out the helper dimension used for calculating the context.
    context = tf.squeeze(context, [1])

    # Stack all alignment vectors into one tensor. This tensor gives alignments for each encoder
    # step.
    padded_alignment = tf.stack(padded_alignment_windows)

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, padded_alignment, padded_alignment


class LocalLuongAttention(LuongAttention):
    """
     Implements a Luong-style local attention mechanism.

     This implementation supports both monotonic attention as well as predictive attention.

    - See [1]: Effective Approaches to Attention-based Neural Machine Translation,
        http://arxiv.org/abs/1508.04025
    """

    def __init__(self, num_units,
                 memory,
                 const_batch_size,
                 memory_sequence_length=None,
                 scale=False,
                 probability_fn=None,
                 score_mask_value=None,
                 dtype=None,
                 name="LocalLuongAttention",
                 d=10,
                 attention_mode=AttentionMode.MONOTONIC,
                 score_mode=AttentionScore.DOT,
                 force_gaussian=False
                 ):
        """
        Arguments:
            num_units (int):
                The depth of the attention mechanism. This controls the number of units in the
                memory layer that processes the encoder states into the `keys`.

            memory (tf.Tensor):
                The memory to query; usually the output of an RNN encoder.
                The shape is expected to be shape=(batch_size, encoder_max_time, ...)

            const_batch_size (int):
                The constant batch size to expect from every batch. Every batch is expected to
                contain exactly `const_batch_size` samples.

            memory_sequence_length:
                (optional) Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths.

            scale (boolean):
                Whether to scale the energy term.

            probability_fn:
                (optional) A `callable`.  Converts the score to
                probabilities.  The default is @{tf.nn.softmax}. Other options include
                @{tf.contrib.seq2seq.hardmax} and @{tf.contrib.sparsemax.sparsemax}.
                Its signature should be: `probabilities = probability_fn(score)`.

            score_mask_value:
                (optional) The mask value for score before passing into
                `probability_fn`. The default is -inf. Only used if
                `memory_sequence_length` is not None.

            dtype (tf.DType):
                The data type for the memory layer of the attention mechanism.

            name (string):
                Name to use when creating ops.

            d (int):
                D parameter controlling the window size and gaussian distribution.
                The window size is set to be `2D + 1`.

            attention_mode (AttentionMode):
                The attention mode to use. Can be either `MONOTONIC` or `PREDICTIVE`.

            score_mode (AttentionScore):
                The attention scoring function to use. Can either be `DOT`, `GENERAL` or `CONCAT`.

            force_gaussian (boolean):
                Force a gaussian distribution onto the scores in the attention window.
                Defaults to False.
        """
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

        # The constant batch size to expect.
        self.const_batch_size = const_batch_size

        self.force_gaussian = force_gaussian

    def __call__(self, query, state):
        """
        Calculate the alignments and next_state for the current decoder output.

        Arguments:
            query (tf.Tensor):
                Decoder cell outputs to compare to the keys (memory).
                The shape is expected to be shape=(B, num_units) with B being the batch size
                and `num_units` being the output size of the decoder_cell.

            state (tf.Tensor):
                In Luong attention the state is equal to the alignments. Therefore this will
                contain the alignments from the previous decoding step.

        Returns:
            (alignments, next_state):
                alignments (tf.Tensor):
                    The normalized attention scores for the attention window. The shape is
                    shape=(B, 2D+1), with B being the batch size and `2D+1` being the window size.
                next_state (tf.Tensor):
                    In Luong attention this is equal to `alignments`.
        """
        with tf.variable_scope(None, "local_luong_attention", [query]):
            # Get the depth of the memory values.
            num_units = self._keys.get_shape()[-1]

            # Get the source sequence length from memory.
            source_seq_length = tf.shape(self._keys)[1]

            if self.attention_mode == AttentionMode.PREDICTIVE:
                # Predictive selection fo the attention window position.
                vp = tf.get_variable(name="local_v_p", shape=[num_units, 1], dtype=tf.float32)
                wp = tf.get_variable(name="local_w_p", shape=[num_units, num_units],
                                     dtype=tf.float32)

                # shape => (B, num_units)
                _intermediate_result = tf.transpose(tf.tensordot(wp, query, [0, 1]))

                # shape => (B, 1)
                _tmp = tf.transpose(tf.tensordot(vp, tf.tanh(_intermediate_result), [0, 1]))

                # Derive p_t as described by Luong for the predictive local-p case.
                self.p = tf.cast(source_seq_length, tf.float32) * tf.sigmoid(_tmp)

            elif self.attention_mode == AttentionMode.MONOTONIC:
                # Derive p_t as described by Luong for the predictive local-m case.
                self.p = tf.tile(
                    [[self.time]],
                    tf.convert_to_tensor([self.batch_size, 1])
                )

                # Prevent the window from leaving the memory.
                self.p = tf.maximum(self.p, self.d)
                self.p = tf.minimum(self.p, source_seq_length - (self.d + 1))
                self.p = tf.cast(self.p, dtype=tf.float32)

            # Calculate the memory sequence index at which the window should start.
            start_index = tf.floor(self.p) - self.d
            start_index = tf.Print(start_index, [tf.reshape(start_index, [-1])], 'start_index FLOAT', summarize=99)

            # Prevent the window from leaving the memory.
            self.window_start = tf.maximum(0, start_index)

            # Calculate the memory sequence index at which the window should stop.
            stop_index = tf.floor(self.p) + self.d + 1
            stop_index = tf.Print(stop_index, [tf.reshape(stop_index, [-1])], 'stop_index FLOAT', summarize=99)

            # Prevent the window from leaving the memory.
            self.window_stop = tf.minimum(source_seq_length, stop_index)

            # Calculate how many padding frames should be added to the start of the window.
            # This is used to get up to the total memory length again.
            self.full_seq_pre_padding = tf.abs(start_index)

            # Calculate how many padding frames should be added to the end of the window.
            # This is used to get up to the total memory length again.
            self.full_seq_post_padding = tf.abs(stop_index - source_seq_length)

            # Calculate how many padding frames should be added to the start of the window.
            # This is used to get the window up to 2D+1 frames.
            self.window_pre_padding = tf.abs(self.window_start - start_index)

            # Calculate how many padding frames should be added to the end of the window.
            # This is used to get the window up to 2D+1 frames.
            self.window_post_padding = tf.abs(self.window_stop - stop_index)

            # Slice the windows for every batch entry.
            with tf.variable_scope(None, "window_extraction", [query]):
                windows = []
                # Iterate the batch entries.
                for i in range(0, self.const_batch_size):
                    # Slice out the window from the processed memory.
                    __window = self._keys[i, self.window_start[i][0]:self.window_stop[i][0], :]

                    # Add zero padding to the slice in order to ensure the window size is (2D+1).
                    paddings = [
                        [self.window_pre_padding[i][0], self.window_post_padding[i][0]],
                        [0, 0]
                    ]
                    __window = tf.pad(__window, paddings, 'CONSTANT')

                    # Collect the extracted windows for each batch entry.
                    windows.append(__window)

                # Merge all extracted windows into one tensor.
                window = tf.stack(windows)

            # Calculate the not not normalized attention score as described by Luong as dot.
            if self.score_mode == AttentionScore.DOT:
                score = _luong_dot_score(query, window, self._scale)
            # Calculate the not not normalized attention score as described by Luong as general.
            elif self.score_mode == AttentionScore.GENERAL:
                score = _luong_general_score(query, window)
            # Calculate the not not normalized attention score as described by Luong as general.
            elif self.score_mode == AttentionScore.CONCAT:
                score = _luong_concat_score(query, window)
            else:
                score = None
                raise Exception("An invalid attention scoring mode was supplied.")

        # Normalize the scores.
        alignments = self._probability_fn(score, state)

        next_state = alignments

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

            attention, alignments, next_attention_state = _luong_local_compute_attention(
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
