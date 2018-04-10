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


class TrainingHelper(seq2seq.Helper):
  """A helper for use during training.  Only reads inputs.

  Returned sample_ids are the argmax of the RNN output logits.
  """

  def __init__(self, inputs, sequence_length, time_major=False, name=None):
    """Initializer.

    Args:
      inputs: A (structure of) input tensors.
      sequence_length: An int32 vector tensor.
      time_major: Python bool.  Whether the tensors in `inputs` are time major.
        If `False` (default), they are assumed to be batch major.
      name: Name scope for any created operations.

    Raises:
      ValueError: if `sequence_length` is not a 1D tensor.
    """
    with ops.name_scope(name, "TrainingHelper", [inputs, sequence_length]):
      inputs = ops.convert_to_tensor(inputs, name="inputs")
      self._inputs = inputs
      if not time_major:
        inputs = nest.map_structure(_transpose_batch_time, inputs)

      self._input_tas = nest.map_structure(_unstack_ta, inputs)
      self._sequence_length = ops.convert_to_tensor(
          sequence_length, name="sequence_length")
      if self._sequence_length.get_shape().ndims != 1:
        raise ValueError(
            "Expected sequence_length to be a vector, but received shape: %s" %
            self._sequence_length.get_shape())

      self._zero_inputs = nest.map_structure(
          lambda inp: array_ops.zeros_like(inp[0, :]), inputs)

      self._batch_size = array_ops.size(sequence_length)

  @property
  def inputs(self):
    return self._inputs

  @property
  def sequence_length(self):
    return self._sequence_length

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def sample_ids_shape(self):
    return tensor_shape.TensorShape([])

  @property
  def sample_ids_dtype(self):
    return dtypes.int32

  def initialize(self, name=None):
    with ops.name_scope(name, "TrainingHelperInitialize"):
      finished = math_ops.equal(0, self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: nest.map_structure(lambda inp: inp.read(0), self._input_tas))
      return (finished, next_inputs)

  def sample(self, time, outputs, name=None, **unused_kwargs):
    with ops.name_scope(name, "TrainingHelperSample", [time, outputs]):
      sample_ids = math_ops.cast(
          math_ops.argmax(outputs, axis=-1), dtypes.int32)
      return sample_ids

  def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
    """next_inputs_fn for TrainingHelper."""
    with ops.name_scope(name, "TrainingHelperNextInputs",
                        [time, outputs, state]):
      next_time = time + 1
      finished = (next_time >= self._sequence_length)
      all_finished = math_ops.reduce_all(finished)
      def read_from_ta(inp):
        return inp.read(next_time)
      next_inputs = control_flow_ops.cond(
          all_finished, lambda: self._zero_inputs,
          lambda: nest.map_structure(read_from_ta, self._input_tas))
      return (finished, next_inputs, state)