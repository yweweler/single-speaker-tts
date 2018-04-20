import tensorflow as tf
from tensorflow.contrib import seq2seq
from tensorflow.python.framework import tensor_shape


class TacotronInferenceHelper(seq2seq.Helper):
    """
    Custom sequence to sequence inference helper for the Tacotron model.
    This helper handles proper initialization of the decoder RNNs initial states and is
    responsible for feeding the last frame of the decoder output as input to the next step.

    See: https://github.com/tensorflow/tensorflow/issues/12065
    """

    def __init__(self, batch_size, input_size):
        """
        Creates an TacotronInferenceHelper instance.

        Arguments:
            batch_size (tf.Dimension):
                Size of the current batch.

            input_size (int):
                RNN input size.
        """
        self._batch_size = batch_size
        self._input_size = input_size

    @property
    def batch_size(self):
        """
        Get the batch size of the current batch.

        Returns:
            tf.Dimension:
                Batch size.
        """
        return self._batch_size

    @property
    def sample_ids_shape(self):
        """
        Shape of tensor returned by `sample`, excluding the batch dimension.

        Note:
            - Since the decoder does not output embeddings this function is basically irrelevant.
            - However it has to be implemented since it is called for some reason.

        Returns:
            tf.TensorShape
        """
        # Copied from the abstract seq2seq.CustomHelper class.
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        """
        DType of tensor returned by `sample`.

        Note:
            - Since the decoder does not output embeddings this function is basically irrelevant.
            - However it has to be implemented since it is called for some reason.

        Returns:
            tf.DType
        """
        # Copied from the abstract seq2seq.CustomHelper class.
        return tf.int32

    def initialize(self, name=None):
        """
        Query information used to initialize the decoder RNN.
        This information includes information about whether the decoding process is finished yet
        as well as the initial inputs to the RNN.

        The initial state of the decoding process has to be that decoding is not finished.
        As for the initial input we use a zero vector aka. <GO> frame.

        Arguments:
            name: Unused.

        Returns:
            (initial_finished, initial_inputs):
                initial_finished (tf.Tensor):
                    A tensor indicating for each sequence in the batch that decoding is not
                    finished. The shape is shape=(B), with B being the batch size.
                initial_inputs:
                    A all zero tensor resembling the <GO> frame used as the first decoder input.
        """
        # When the decoder starts, there is no sequence in the batch that is finished.
        initial_finished = tf.tile([False], [self._batch_size])

        # The initial input for the decoder is considered to be a <GO> frame.
        # We will input an zero vector as the <GO> frame.
        initial_inputs = tf.zeros([self._batch_size, self._input_size], dtype=tf.float32)

        return initial_finished, initial_inputs

    def sample(self, time, outputs, state, name=None):
        """
        Takes outputs and emits sample id's.

        Note:
            - Since the decoder does not use embeddings this function is basically irrelevant.
            - However it has to be implemented since it is called for some reason.

        Arguments:
            time: Unused.
            outputs: Unused.
            state: Unused.
            name: Unused.

        Returns:
            tf.Tensor
        """
        # return None => ValueError: x and y must both be non-None or both be None

        # Returning some tensor of dtype=tf.int32 and random shape seems to be enough.
        return tf.zeros(1, dtype=tf.int32)

    def __is_decoding_finished(self, outputs):
        """
        Determine for each sequence in a batch if decoding is finished or not.

        Arguments:
            outputs (tf.Tensor):
                Outputs of the last decoder step. The shape is expected to be shape=(B, O),
                with B being the batch size and O being the RNNs output size.

        Returns:
            tf.Tensor:
                A tensor indicating for each sequence in the batch whether decoding is
                finished. The shape is shape=(B), with B being the batch size.

        """
        # TODO: Since I am not sure when to stop I will let the decoder run into max_steps.
        finished = tf.tile([False], [self._batch_size])

        return finished

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """
        Query the next RNN inputs and RNN state as well as whether decoding is finished or not.

        Arguments:
            time: Unused.

            outputs (tf.Tensor):
                RNN outputs from the last decoding step. The shape is expected to be shape=(B, O),
                with B being the batch size and O being the RNNs output size.

            state:
                RNN state from the last decoding step.

            sample_ids: Unused.
            name: Unused.

        Returns:
            (finished, next_inputs, next_state):
                finished (tf.Tensor):
                    A tensor indicating for each sequence in the batch if decoding is finished.
                    The shape is shape=(B), with B being the batch size.
                next_inputs (tf.Tensor):
                    Tensor containing the inputs for the next step. The shape is
                    shape=(B, input_size), with B being the batch size.
                next_state:
                    RNN state.
        """
        del time, sample_ids  # unused by next_inputs

        # Check if decoding is finished.
        finished = self.__is_decoding_finished(outputs)

        # Use the last steps outputs as the next steps inputs.
        # When using the Tacotron reduction factor r the RNN produces an output of size
        # r * `input_size`. But it only takes input of size `input_size`.
        # We will therefore only pass every r'th frame to the next decoding step.
        next_inputs = outputs[:, -self._input_size:]

        # Use the resulting state from the last step as the next state.
        next_state = state

        return finished, next_inputs, next_state


class TacotronTrainingHelper(seq2seq.Helper):
    """
    Custom sequence to sequence training helper for the Tacotron model.
    This helper handles proper initialization of the decoder RNNs initial states and is
    responsible for feeding the last frame of the decoder output as input to the next step.

    This helper feeds every r'th frame of the RNNs output as input to the next decoding
    step as described in [1], section "3.3 Decoder".

    See: "Tacotron: Towards End-to-End Speech Synthesis"
      * Source: [1] https://arxiv.org/abs/1703.10135
    """

    def __init__(self, inputs, targets, output_dim, reduction_factor):
        """
        Creates an TacotronTrainingHelper instance.

        Arguments:
            batch_size (tf.Dimension):
                Size of the current batch.

            inputs (tf.Tensor):
                Tensor containing the encoder outputs. The shape is expected to be shape=(B,
                T_sent, embedding_size), with B being the batch size and T_sent being the number
                of symbols in the input sentence.

            outputs (tf.Tensor):
                Ground truth Mel. spectrogram data used for feeding ground truth frames during
                training. The shape is expected to be shape=(B, T_spec, n_mels), with B being the
                batch size and T_spec being the number of frames in the spectrogram.
                
            input_size (int):
                RNN input size.

            reduction_factor (int):
                The Tacotron reduction factor to use. Used to feed every r'th ground truth frame.
        """
        with tf.name_scope("TacotronTrainingHelper"):
            self._inputs = inputs
            self._targets = targets[:, reduction_factor - 1::reduction_factor, :]
            self._output_dim = output_dim
            self._reduction_factor = reduction_factor

            self._batch_size = tf.shape(inputs)[0]

            # Copy every r'th frame from the ground truth spectrogram.
            # => shape=(B, T_spec // reduction_factor, n_mels)
            # self._outputs = outputs[:, self._reduction_factor - 1::self._reduction_factor, :]

            # outputs = tf.Print(outputs, [tf.shape(outputs)], 'outputs.shape')
            # self._outputs = self._inputs[:, :, :]

            # Get the number of time frames the decoder has to produce.
            # Note that we will produce sequences over the entire length of the batch. Maybe this
            # way the network will learn to generate silence after producing the actual sentence.
            n_target_steps = tf.shape(self._targets)[1]
            self._sequence_length = tf.tile([n_target_steps], [self._batch_size])
            self._sequence_length = tf.Print(self._sequence_length, [self._sequence_length], '_sequence_length')

    # @property
    # def inputs(self):
    #     """
    #     Get the RNN inputs.
    #
    #     Returns (tf.Tensor):
    #         The RNNs inputs. The shape is expected to be shape=(B, T_sent, embedding_size),
    #         with B being the batch size and T_sent being the number of symbols in the input
    #         sentence.
    #     """
    #     return self._inputs

    @property
    def sequence_length(self):
        """
        Get the sequence lengths.

        Returns:
            tf.Tensor:
                Tensor containing the the sequence lengths of each entry in the batch. The shape
                is shape=(B), with B being the batch size.
        """
        return self._sequence_length

    @property
    def batch_size(self):
        """
        Get the batch size of the current batch.

        Returns:
            tf.Dimension:
                Batch size.
        """
        return self._batch_size

    @property
    def sample_ids_shape(self):
        """
        Shape of tensor returned by `sample`, excluding the batch dimension.

        Note:
            - Since the decoder does not output embeddings this function is basically irrelevant.
            - However it has to be implemented since it is called for some reason.

        Returns:
            tf.TensorShape
        """
        # Copied from the seq2seq.TrainingHelper class.
        return tensor_shape.TensorShape([])

    @property
    def sample_ids_dtype(self):
        """
        DType of tensor returned by `sample`.

        Note:
            - Since the decoder does not output embeddings this function is basically irrelevant.
            - However it has to be implemented since it is called for some reason.

        Returns:
            tf.DType
        """
        # Copied from the seq2seq.TrainingHelper class.
        return tf.int32

    def initialize(self, name=None):
        """
        Query information used to initialize the decoder RNN.
        This information includes information about whether the decoding process is finished yet
        as well as the initial inputs to the RNN.

        The initial state of the decoding process has to be that decoding is not finished.
        As for the initial input we use a zero vector aka. <GO> frame.

        Arguments:
            name: Unused.

        Returns:
            (initial_finished, initial_inputs):
                initial_finished (tf.Tensor):
                    A tensor indicating for each sequence in the batch that decoding is not
                    finished. The shape is shape=(B), with B being the batch size.
                initial_inputs:
                    A all zero tensor resembling the <GO> frame used as the first decoder input.
        """
        with tf.name_scope(name, "TacotronTrainingHelperInitialize"):
            # When the decoder starts, there is no sequence in the batch that is finished.
            initial_finished = tf.tile([False], [self._batch_size])

            # The initial input for the decoder is considered to be a <GO> frame.
            # We will input an zero vector as the <GO> frame.
            initial_inputs = tf.zeros([self._batch_size, self._output_dim], dtype=tf.float32)

        return initial_finished, initial_inputs

    def sample(self, time, outputs, name=None, **unused_kwargs):
        """
        Takes outputs and emits sample id's

        Note:
            - Since the decoder does not use embeddings this function is basically irrelevant.
            - However it has to be implemented since it is called for some reason.

        Arguments:
            time: Unused.
            outputs: Unused.
            name: Unused.
            **unused_kwargs: Unused.

        Returns:
            tf.Tensor
        """
        # Returning some tensor of dtype=tf.int32 and random shape seems to be enough.
        return tf.zeros(1, dtype=tf.int32)

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        """
        Query the next RNN inputs and RNN state as well as whether decoding is finished or not.

        Arguments:
            time:
                Index in the time axis from the last decoding step.

            outputs (tf.Tensor):
                RNN outputs from the last decoding step. The shape is expected to be shape=(B, O),
                with B being the batch size and O being the RNNs output size.

            state:
                RNN state from the last decoding step.

            name: Unused.
            **unused_kwargs: Unused.

        Returns:
            (finished, next_inputs, next_state):
                finished (tf.Tensor):
                    A tensor indicating for each sequence in the batch if decoding is finished.
                    The shape is shape=(B), with B being the batch size.
                next_inputs (tf.Tensor):
                    Tensor containing the inputs for the next step. The shape is
                    shape=(B, input_size), with B being the batch size.
                next_state:
                    RNN state.
        """
        with tf.name_scope("TacotronTrainingHelperNextInputs"):
            # Increment the time index.
            next_time = time + 1

            # Query finished state for each sequence in the batch.
            finished = (next_time >= self._sequence_length)

            # During training we do not use the last steps outputs as the next steps inputs.
            # We will feed the r'th ground truth frame from the Mel. spectrogram we prepared
            # earlier.

            next_inputs = self._targets[:, time, :]
            print('pre_fetch_next_inputs', next_inputs)
            # next_inputs.set_shape(shape=(4, self._input_size))
            print('pre_fetch_next_inputs.set_shape', next_inputs)

            # Use the resulting state from the last step as the next state.
            next_state = state

            return finished, next_inputs, next_state
