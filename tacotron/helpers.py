import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.framework import tensor_shape


class CustomTacotronInferenceHelper(seq2seq.CustomHelper):
    pass


class TacotronInferenceHelper(seq2seq.Helper):
    # See: https://github.com/tensorflow/tensorflow/issues/12065
    def __init__(self, batch_size, input_size):
        self._batch_size = batch_size
        self.input_size = input_size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        # Copied from the abstract seq2seq.CustomHelper class.
        return tensor_shape.TensorShape([])
        # raise NotImplementedError('Not implemented, since the decoder does not output embeddings')

    @property
    def sample_ids_dtype(self):
        # Copied from the abstract seq2seq.CustomHelper class.
        return tf.int32
        # raise NotImplementedError('Not implemented, since the decoder does not output embeddings')

    def initialize(self, name=None):
        # When the decoder starts, there is no sequence in the batch that is finished.
        initial_finished = tf.tile([False], [self._batch_size])

        # The initial input for the decoder is considered to be a <GO> frame.
        # We will input an zero vector as the <GO> frame.
        initial_inputs = tf.zeros([self._batch_size, self.input_size], dtype=tf.float32)

        return initial_finished, initial_inputs

    def sample(self, time, outputs, state, name=None):
        # A callable that takes outputs and emits tensor sample_ids.
        # Not sure why this is called when it is not actually needed ;(.

        # return None => ValueError: x and y must both be non-None or both be None

        # It seems to work when just returning some tensor of dtype=tf.int32 and random shape.
        return tf.zeros(1, dtype=tf.int32)
        # raise NotImplementedError('Not implemented, since the decoder does not output embeddings')

    def __is_decoding_finished(self, batch):
        # TODO: Since I am not sure when to stop I will let the decoder stop run into max_steps.
        finished = tf.tile([False], [self._batch_size])

        return finished

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time, sample_ids  # unused by next_inputs

        # TODO: Make sure that the outputs that are passed to this function are the last steps outp.
        # Use the last steps outputs as the next steps inputs.
        next_inputs = outputs

        # Use the resulting state from the last step as the next state.
        next_state = state

        # Check if decoding is finished.
        finished = self.__is_decoding_finished(outputs)

        return finished, next_inputs, next_state


class InferenceHelper(seq2seq.Helper):
    """A helper to use during inference with a custom sampling function."""

    def __init__(self, sample_fn, sample_shape, sample_dtype,
                 start_inputs, end_fn, next_inputs_fn=None, is_time_major=False):
        """Initializer.

        Args:
          sample_fn: A callable that takes `outputs` and emits tensor `sample_ids`.
          sample_shape: Either a list of integers, or a 1-D Tensor of type `int32`,
            the shape of the each sample in the batch returned by `sample_fn`.
          sample_dtype: the dtype of the sample returned by `sample_fn`.
          start_inputs: The initial batch of inputs.
          end_fn: A callable that takes `sample_ids` and emits a `bool` vector
            shaped `[batch_size]` indicating whether each sample is an end token.
          next_inputs_fn: (Optional) A callable that takes `sample_ids` and returns
            the next batch of inputs. If not provided, `sample_ids` is used as the
            next batch of inputs.
        """
        self._sample_fn = sample_fn
        self._end_fn = end_fn
        self._sample_shape = tensor_shape.TensorShape(sample_shape)
        self._sample_dtype = sample_dtype
        self._next_inputs_fn = next_inputs_fn

        if is_time_major:
            self._batch_size = tf.shape(start_inputs)[1]
        else:
            self._batch_size = tf.shape(start_inputs)[0]

        self._start_inputs = ops.convert_to_tensor(
            start_inputs, name="start_inputs")

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sample_ids_shape(self):
        return self._sample_shape

    @property
    def sample_ids_dtype(self):
        return self._sample_dtype

    def initialize(self, name=None):
        finished = array_ops.tile([False], [self._batch_size])
        return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        del time, state  # unused by sample
        return self._sample_fn(outputs)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        del time, outputs  # unused by next_inputs
        if self._next_inputs_fn is None:
            next_inputs = sample_ids
        else:
            next_inputs = self._next_inputs_fn(sample_ids)
        finished = self._end_fn(sample_ids)
        return (finished, next_inputs, state)
