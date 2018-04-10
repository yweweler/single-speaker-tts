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


class TacotronTrainingHelper(seq2seq.Helper):
    def __init__(self, batch_size, inputs, outputs, output_size):
        with tf.name_scope("TacotronTrainingHelper"):
            self._batch_size = batch_size
            self._inputs = inputs
            self._outputs = outputs
            self._output_size = output_size

            print('batch_size', batch_size)
            print('_inputs', self._inputs)
            print('_outputs', self._outputs)
            print('_output_size', self._output_size)

            # Get the number of time frames the decoder has to produce.
            n_decoder_steps = tf.shape(self._outputs)[1]
            self._sequence_length = tf.tile([n_decoder_steps], [self._batch_size])

            self._zero_inputs = tf.zeros([self._batch_size, 256], dtype=tf.float32)

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
        # Copied from the seq2seq.TrainingHelper class.
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        # Copied from the seq2seq.TrainingHelper class.
        return tf.int32

    def initialize(self, name=None):
        with tf.name_scope(name, "TacotronTrainingHelperInitialize"):
            # When the decoder starts, there is no sequence in the batch that is finished.
            initial_finished = tf.tile([False], [self._batch_size])

            # The initial input for the decoder is considered to be a <GO> frame.
            # We will input an zero vector as the <GO> frame.
            initial_output = tf.zeros([self._batch_size, self._output_size], dtype=tf.float32)

        return initial_finished, initial_output

    def sample(self, time, outputs, name=None, **unused_kwargs):
        # It seems to work when just returning some tensor of dtype=tf.int32 and random shape.
        return tf.zeros(1, dtype=tf.int32)
        # raise NotImplementedError('Not implemented, since the decoder does not output embeddings')

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """next_inputs_fn for TacotronTrainingHelper."""
        with tf.name_scope("TacotronTrainingHelperNextInputs"):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = tf.reduce_all(finished)

            # next_inputs = tf.cond(all_finished,
            #                       lambda: self._zero_inputs,
            #                       lambda: self._outputs[:, time, :])

            next_inputs = self._outputs[:, time, :]  # self._zero_inputs

            # test = tf.Print(next_inputs, [tf.shape(next_inputs)], 'next_inputs.shape')
            # tf.summary.tensor_summary('test', test)

            return finished, next_inputs, state
