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
        Wraps and RNNCell instance and applies a Pre-Net to the inputs.

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
        Get the size(s) of state(s) used by teh wrapped cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
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

        Arguments:
            inputs (tf.Tensor):
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
        projected = pre_net(inputs, self._layers, scope='pre_net', training=self._training)

        return self._cell(projected, state)


class ConcatOutputAndAttentionWrapper(tfc.rnn.RNNCell):
    # TODO: This is actually part of Tacotron 2 and only used for experimental reasons.
    """
    Concatenates RNN cell output with the attention context vector.
    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    """

    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state, **kwargs):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
