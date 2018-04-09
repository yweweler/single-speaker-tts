import tensorflow as tf
import tensorflow.contrib as tfc

from tacotron.layers import pre_net


class PrenetWrapper(tfc.rnn.RNNCell):
    def __init__(self, cell, pre_net_layers, training):
        super(PrenetWrapper, self).__init__()
        self._cell = cell
        self._layers = pre_net_layers
        self._training = training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def compute_output_shape(self, input_shape):
        raise NotImplementedError()

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)

    # TODO: Do we have to override the RNNLayer.call func. or the RNNCell.__call__ func. ?
    def call(self, inputs, state, **kwargs):
        projected = pre_net(inputs, self._layers, scope='pre_net', training=self._training)

        return self._cell(projected, state)


class ConcatOutputAndAttentionWrapper(tfc.rnn.RNNCell):
    '''Concatenates RNN cell output with the attention context vector.
    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''

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
