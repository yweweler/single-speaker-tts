import tensorflow as tf
import tensorflow.contrib as tfc

from tacotron.layers import pre_net


class PrenetWrapper(tfc.rnn.RNNCell):
    """
    RNNCell wrapper that applies a Pre-Net to the RNNs inputs as described in
    "Tacotron: Towards End-to-End Speech Synthesis".

    See: "Tacotron: Towards End-to-End Speech Synthesis"
      * Source: [1] https://arxiv.org/abs/1703.10135
    """

    def __init__(self, cell, pre_net_layers, training):
        """
        Wraps a RNNCell instance and applies a Pre-Net to the inputs.

        Arguments:
            cell (tensorflow.contrib.rnn.RNNCell):
                RNN cell to be wrapped.

            pre_net_layers (:obj:`list` of :obj:`tuple`):
                A list of length L containing tuples of the form (units, dropout, activation )
                defining the number of units the dropout rate and the activation function for L
                layers.

            training (boolean):
                Boolean defining whether to apply the dropout or not.
        """
        super(PrenetWrapper, self).__init__()
        self._cell = cell
        self._layers = pre_net_layers
        self._training = training

    @property
    def state_size(self):
        """
        Get the size(s) of state(s) used by the wrapped cell.

        Returns:
            object:
                It can be represented by an Integer, a TensorShape or a tuple of Integers
                or TensorShapes.
        """
        return self._cell.state_size

    @property
    def output_size(self):
        """
        Get the size of the outputs produced by the wrapped cell.

        Returns:
            object:
                Integer or TensorShape.
        """
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer given the input shape.

        Note:
            - This is function is implemented since it is abstract in the super classes.
            - This function is unused however.

        Arguments:
            input_shape: Unused
        """
        raise NotImplementedError

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Arguments:
          batch_size (object):
            int, float, or unit Tensor representing the batch size.

          dtype (tf.DType):
            The data type to use for the state.

        Returns:
            tf.Tensor:
                - If `state_size` is an int or TensorShape, then the return value is a `N-D` tensor
                of shape `[batch_size, state_size]` filled with zeros.
                - If `state_size` is a nested list or tuple, then the return value is a nested
                list or tuple (of the same structure) of `2-D` tensors with the shapes
                `[batch_size, s]` for each s in `state_size`.

        """
        return self._cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        """
        Run this RNN cell on inputs, starting from the given state.

        A Tacotron pre-net is applied before the inputs are fed into the cell.
        The state is fed unmodified.

        Arguments:
            inputs (tf.Tensor):
                TODO: (B, 336)
                `2-D` tensor with shape `[batch_size, input_size]`.

            state:
                If `self.state_size` is an integer, this should be a `2-D Tensor` with shape
                `[batch_size, self.state_size]`.  Otherwise, if `self.state_size` is a tuple of
                integers, this should be a tuple with shapes `[batch_size, s] for s in
                self.state_size`.

            scope:
                Unused.

        Returns:
            (output, new_state):
                output (tf.Tensor):
                    A `2-D` tensor with shape `[batch_size, self.output_size]`.
                new_state (tf.Tensor):
                    Either a single `2-D` tensor, or a tuple of tensors matching the arity and
                    shapes of `state`.
        """
        projected_inputs = pre_net(inputs, self._layers, scope='pre_net', training=self._training)

        return self._cell(projected_inputs, state)


class ConcatOutputAndAttentionWrapper(tfc.rnn.RNNCell):
    # TODO: This is actually part of Tacotron 2 and only used for experimental reasons.
    """
    Concatenates the output on a wrapped RNNCell with the attention context vector produced by
    the cell.

    `concat([rnn_output, rnn_state.attention], axis=-1)`

    The wrapped cell is expected to fulfill the following criteria:
       - The cell is expected to be wrapped with an AttentionWrapper.
       - attention_layer_size = None
       - output_attention = False
    """

    def __init__(self, cell):
        """
        Wraps a RNNCell instance derived from an AttentionWrapper and concatenates the RNNCells
        output with the attention context vector.

        Arguments:
            cell (tensorflow.contrib.rnn.RNNCell):
                RNN cell to be wrapped.
        """
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        """
        Get the size(s) of state(s) used by the wrapped cell.

        Returns:
            # TODO: Returns an AttentionWrapperState instance also containing normal tensors (why?).
            tf.contrib.seq2seq.AttentionWrapperState:
                A tuple of Integers or TensorShapes.
        """
        # print('self._cell.state_size', self._cell.state_size)

        # AttentionWrapperState(
        #   cell_state=256,
        #   attention=256,
        #   time=TensorShape([]),
        #   alignments=<tf.Tensor 'decoder/BahdanauAttention/strided_slice_2:0' shape=() dtype=int32>,
        #   alignment_history=(), attention_state=<tf.Tensor 'decoder/BahdanauAttention/strided_slice_2:0' shape=() dtype=int32>
        # )
        return self._cell.state_size

    @property
    def output_size(self):
        """
        Get the size of the outputs produced by the wrapped cell.

        Returns:
            object:
                Integer or TensorShape.
        """
        return self._cell.output_size + self._cell.state_size.attention

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer given the input shape.

        Note:
            - This is function is implemented since it is abstract in the super classes.
            - This function is unused however.

        Arguments:
            input_shape: Unused
        """
        raise NotImplementedError

    def __call__(self, inputs, state, scope=None):
        """
        Run this RNN cell on inputs, starting from the given state.

        The outputs are concatenated with the states attention to form the new outputs.
        The inputs and state are fed unmodified.

        Arguments:
            inputs (tf.Tensor):
                `2-D` tensor with shape `[batch_size, input_size]`.

            state:
                TODO: Actually is an AttentionWrapperState like self.state_size
                If `self.state_size` is an integer, this should be a `2-D Tensor` with shape
                `[batch_size, self.state_size]`.  Otherwise, if `self.state_size` is a tuple of
                integers, this should be a tuple with shapes `[batch_size, s] for s in
                self.state_size`.

            scope:
                Unused.

        Returns:
            (concat_output, new_state):
                concat_output (tf.Tensor):
                    A `2-D` tensor with shape `[batch_size, self.output_size]`.
                new_state (tf.Tensor):
                    Either a single `2-D` tensor, or a tuple of tensors matching the arity and
                    shapes of `state`.
        """
        output, new_state = self._cell(inputs, state)
        concat_output = tf.concat([output, new_state.attention], axis=-1)

        return concat_output, new_state

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).

        Arguments:
          batch_size (object):
            int, float, or unit Tensor representing the batch size.

          dtype (tf.DType):
            The data type to use for the state.

        Returns:
            tf.Tensor:
                - If `state_size` is an int or TensorShape, then the return value is a `N-D` tensor
                of shape `[batch_size, state_size]` filled with zeros.
                - If `state_size` is a nested list or tuple, then the return value is a nested
                list or tuple (of the same structure) of `2-D` tensors with the shapes
                `[batch_size, s]` for each s in `state_size`.

        """

        return self._cell.zero_state(batch_size, dtype)
