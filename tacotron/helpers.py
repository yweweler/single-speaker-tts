import tensorflow as tf
from tensorflow.contrib import seq2seq


class TacotronInferenceHelper(seq2seq.Helper):
    def __init__(self):
        pass

    @property
    def batch_size(self):
        pass

    @property
    def sample_ids_shape(self):
        pass

    @property
    def sample_ids_dtype(self):
        pass

    def initialize(self, name=None):
        pass

    def sample(self, time, outputs, state, name=None):
        pass

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        pass


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
