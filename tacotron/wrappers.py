# from tensorflow import contrib as tfc
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
