import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.contrib import seq2seq

from audio.conversion import inv_normalize_decibel, decibel_to_magnitude, ms_to_samples
from audio.synthesis import spectrogram_to_wav
from tacotron.helpers import TacotronInferenceHelper, TacotronTrainingHelper
from tacotron.layers import cbhg, pre_net
from tacotron.params.model import model_params
from tacotron.wrappers import PrenetWrapper


class Mode:
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'predict'


class Tacotron:
    """
    Implementation of the Tacotron architecture as described in
    "Tacotron: Towards End-to-End Speech Synthesis".

    See: "Tacotron: Towards End-to-End Speech Synthesis"
      * Source: [1] https://arxiv.org/abs/1703.10135
    """

    def __init__(self, inputs, mode):
        """
        Creates an instance of the Tacotron model.

        Arguments:
            inputs (:obj:`dict`):
                Input data placeholders. All data that is used for training or inference is
                consumed from this placeholders.
                The placeholder dictionary contains the following fields with keys of the same name:
                    - ph_sentences (tf.Tensor):
                        Batched integer sentence sequence with appended <EOS> token padded to same
                        length using the <PAD> token. The characters were converted
                        converted to their vocabulary id's. The shape is shape=(B, T_sent, ?),
                        with B being the batch size and T_sent being the sentence length
                        including the <EOS> token.
                    - ph_sentence_length (tf.Tensor):
                        Batched sequence lengths including the <EOS> token, excluding the padding.
                        The shape is shape=(B), with B being the batch size.
                    - ph_mel_specs (tf.Tensor):
                        Batched Mel. spectrogram's that were padded to the same length in the
                        time axis using zero frames. The shape is shape=(B, T_spec, n_mels),
                        with B being the batch size and T_spec being the number of frames in the
                        spectrogram.
                    - ph_lin_specs (tf.Tensor):
                        Batched linear spectrogram's that were padded to the same length in the
                        time axis using zero frames. The shape is shape=(B, T_spec, 1 + n_fft // 2),
                        with B being the batch size and T_spec being the number of frames in the
                        spectrogram.
                    - ph_time_frames (tf.Tensor):
                        Batched number of frames in the spectrogram's excluding the padding
                        frames. The shape is shape=(B), with B being the batch size.


            mode (Mode):
                Flag that controls the application of special architecture behaviour that only
                has to be applied during training or evaluation.
                Permitted modes are `TRAIN`, `EVAL` and `PREDICT`.
        """
        self.hparams = model_params

        # Get the placeholders for the input data.
        self.inp_sentences = inputs['ph_sentences']
        self.seq_lengths = inputs['ph_sentence_length']
        self.inp_mel_spec = inputs['ph_mel_specs']
        self.inp_linear_spec = inputs['ph_lin_specs']
        self.inp_time_steps = inputs['ph_time_frames']

        # Merged loss function.
        self.loss_op = None
        # Mel. spectrogram loss measured after the decoder.
        self.loss_op_decoder = None
        # Linear spectrogram loss measured at the and of the network.
        self.loss_op_post_processing = None

        # Decoded Mel. spectrogram, shape=(B, T_spec, n_mels).
        self.output_mel_spec = None

        # Decoded linear spectrogram, shape => (B, T_spec, (1 + n_fft // 2)).
        self.output_linear_spec = None

        self._mode = mode

        # Construct the network.
        self.model()

    def is_training(self):
        """
        Returns if the model is in training mode or not.

        Returns:
            boolean:
                True if Mode == TRAIN, False otherwise.
        """
        if self._mode == Mode.TRAIN:
            return True
        else:
            return False

    def encoder(self, inputs):
        """
        Implementation of the CBHG based Tacotron encoder network.

        Arguments:
            inputs (tf.Tensor):
            The shape is expected to be shape=(B, T_sent, ) with B being the batch size, T_sent
            being the number of tokens in the sentence including the EOS token.

        Returns:
         (outputs, output_states):
            outputs (tf.Tensor): The output states (output_fw, output_bw) of the RNN concatenated
                over time. Its shape is expected to be shape=(B, T_sent, 2 * n_gru_units) with B
                being the batch size, T_sent being the number of tokens in the sentence including
                the EOS token.

            output_states (tf.Tensor): A tensor containing the forward and the backward final states
                (output_state_fw, output_state_bw) of the bidirectional rnn.
                Its shape is expected to be shape=(B, 2, n_gru_units) with B being the batch size.
        """
        with tf.variable_scope('encoder'):
            char_embeddings = tf.get_variable("embedding",
                                              [
                                                  self.hparams.vocabulary_size,
                                                  self.hparams.encoder.embedding_size
                                              ],
                                              dtype=tf.float32,
                                              initializer=tf.glorot_uniform_initializer())

            # shape => (B, T_sent, 256)
            embedded_char_ids = tf.nn.embedding_lookup(char_embeddings, inputs)

            # shape => (B, T_sent, 128)
            network = pre_net(inputs=embedded_char_ids,
                              layers=self.hparams.encoder.pre_net_layers,
                              training=self.is_training())

            # network.shape => (B, T_sent, 2 * n_gru_units)
            # state.shape   => (2, n_gru_units)
            network, state = cbhg(inputs=network,
                                  n_banks=self.hparams.encoder.n_banks,
                                  n_filters=self.hparams.encoder.n_filters,
                                  n_highway_layers=self.hparams.encoder.n_highway_layers,
                                  n_highway_units=self.hparams.encoder.n_highway_units,
                                  projections=self.hparams.encoder.projections,
                                  n_gru_units=self.hparams.encoder.n_gru_units,
                                  training=self.is_training())

        return network, state

    def attention_decoder(self, inputs, memory, num_units=None, scope="attention_decoder",
                          reuse=None):

        # with tf.variable_scope(scope, reuse=reuse):
        #     attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units,
        #                                                                memory)
        #     decoder_cell = tf.contrib.rnn.GRUCell(num_units)
        #     cell_with_attention = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
        #                                                               attention_mechanism,
        #                                                               num_units,
        #                                                               alignment_history=True)
        #     outputs, state = tf.nn.dynamic_rnn(cell_with_attention, inputs,
        #                                        dtype=tf.float32)  # ( N, T', 16)
        #
        # return outputs, state

        with tf.variable_scope(scope, reuse=reuse):
            # Create the attention mechanism.
            attention_mechanism = tfc.seq2seq.BahdanauAttention(
                num_units=num_units,
                memory=memory,
            )

            # Create the attention RNN cell.
            decoder_cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)

            # Connect the attention cell with the attention mechanism.
            cell_with_attention = tfc.seq2seq.AttentionWrapper(
                cell=decoder_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=num_units,
                alignment_history=True,
                # TODO: Why does this even work with BahdanauAttention?
                output_attention=True,  # True for Luong-style att., False for Bhadanau-style.
            )  # => (B, T_sent, n_attention_units) = (B, T_sent, 256)

            # ==========================================================================================

            # Determine the current batch size.
            batch_size = tf.shape(inputs)[0]

            decoder_initial_state = cell_with_attention.zero_state(
                batch_size=batch_size,
                dtype=tf.float32
            )

            if self.is_training():
                # Create a custom training helper for feeding ground truth frames during training.
                helper = TacotronTrainingHelper(
                    batch_size=batch_size,
                    inputs=inputs,
                    outputs=None,
                    input_size=128,
                    reduction_factor=self.hparams.reduction
                )
            # else:
            #     # Create a custom inference helper that handles proper data feeding.
            #     helper = TacotronInferenceHelper(batch_size=batch_size,
            #                                      input_size=256)

            decoder = seq2seq.BasicDecoder(cell=cell_with_attention,
                                           helper=helper,
                                           initial_state=decoder_initial_state)

            if self.is_training():
                # During training we do not stop decoding manually. The decoder automatically
                # decodes as many time steps as are contained in the ground truth data.
                maximum_iterations = None
            else:
                # During inference we stop decoding after `maximum_iterations`. Note that when
                # using the reduction factor the RNN actually outputs
                # `maximum_iterations` * `reduction_factor` frames.
                maximum_iterations = self.hparams.decoder.maximum_iterations // self.hparams.reduction

            # Start decoding.
            decoder_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=maximum_iterations)

            decoder_outputs = decoder_outputs.rnn_output

            print('attention_decoder.decoder_outputs', decoder_outputs)
            print('attention_decoder.decoder_state', final_state)

        return decoder_outputs, final_state

    def decoder1(self, _inputs, memory, is_training=True, scope="decoder1", reuse=None):
        '''
        Args:
          _inputs: A 3d tensor with shape of [N, T_y/r, n_mels(*r)]. Shifted log melspectrogram of sound files.
          memory: A 3d tensor with shape of [N, T_x, E].
          is_training: Whether or not the layer is in training mode.
          scope: Optional scope for `variable_scope`
          reuse: Boolean, whether to reuse the weights of a previous layer
            by the same name.

        Returns
          Predicted log melspectrogram tensor with shape of [N, T_y/r, n_mels*r].
        '''
        with tf.variable_scope(scope, reuse=reuse):
            # Decoder pre-net
            _inputs = pre_net(inputs=_inputs,
                              layers=(
                                  # (units, dropout, activation).
                                  (256, 0.5, tf.nn.relu),
                                  (128, 0.5, tf.nn.relu)
                              ),
                              training=self.is_training())  # (N, T_y/r, E/2)

            # Attention RNN
            dec, state = self.attention_decoder(_inputs, memory, num_units=256)  # (N, T_y/r, E)

            ## for attention monitoring
            alignments = tf.transpose(state.alignment_history.stack(), [1, 2, 0])

            tf.summary.image("stacked_alignments", tf.expand_dims(alignments, -1))

            # Stack several GRU cells and apply a residual connection after each cell.
            # Before the input reaches the decoder RNN it passes through the attention cell.
            cells = []
            for i in range(2):
                # => (B, T_spec, n_decoder_units) = (B, T_spec, 256)
                cell = tf.nn.rnn_cell.GRUCell(num_units=256, name='gru_cell_{}'.format(i))
                # => (B, T_spec, n_decoder_units) = (B, T_spec, 256)
                cell = tf.nn.rnn_cell.ResidualWrapper(cell)
                cells.append(cell)

            # => (B, T_spec, n_decoder_units) = (B, T_spec, 256)
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

            # Determine the current batch size.
            batch_size = tf.shape(_inputs)[0]

            decoder_initial_state = decoder_cell.zero_state(
                batch_size=batch_size,
                dtype=tf.float32
            )

            if self.is_training():
                # Create a custom training helper for feeding ground truth frames during training.
                helper = TacotronTrainingHelper(
                    batch_size=batch_size,
                    inputs=dec,
                    outputs=self.inp_mel_spec,
                    input_size=256,
                    reduction_factor=self.hparams.reduction
                )
            else:
                # Create a custom inference helper that handles proper data feeding.
                helper = TacotronInferenceHelper(batch_size=batch_size,
                                                 input_size=256)

            decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                           helper=helper,
                                           initial_state=decoder_initial_state)

            if self.is_training():
                # During training we do not stop decoding manually. The decoder automatically
                # decodes as many time steps as are contained in the ground truth data.
                maximum_iterations = None
            else:
                # During inference we stop decoding after `maximum_iterations`. Note that when
                # using the reduction factor the RNN actually outputs
                # `maximum_iterations` * `reduction_factor` frames.
                maximum_iterations = self.hparams.decoder.maximum_iterations // self.hparams.reduction

            # Start decoding.
            decoder_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=maximum_iterations)

            decoder_outputs = decoder_outputs.rnn_output

            # Decoder RNNs
            # dec += self.gru(dec, 256, bidirection=False, scope="decoder_gru1")  # (N, T_y/r, E)
            # dec += self.gru(dec, 256, bidirection=False, scope="decoder_gru2")  # (N, T_y/r, E)

            # Outputs => (N, T_y/r, n_mels*r)
            mel_hats = tf.layers.dense(decoder_outputs, 80 * self.hparams.reduction)

        return mel_hats, alignments

    def decoder(self, encoder_outputs, encoder_state):
        """
        Implementation of the Tacotron decoder network.

        Arguments:
            encoder_outputs (tf.Tensor):
                The output states of the encoder RNN concatenated over time. Its shape is
                expected to be shape=(B, T_sent, 2 * encoder.n_gru_units) with B being the batch
                size, T_sent being the number of tokens in the sentence including the EOS token.

            encoder_state (tf.Tensor):
                A tensor containing the forward and the backward final states
                (output_state_fw, output_state_bw) of the bidirectional RNN. Its shape is
                expected to be shape=(B, 2, encoder.n_gru_units) with B being the batch size.

        Returns:
            tf.tensor:
                Generated reduced Mel. spectrogram. The shape is
                shape=(B, T_spec // r, n_mels * r), with B being the batch size, T_spec being
                the number of frames in the spectrogram and r being the reduction factor.
        """
        with tf.variable_scope('decoder'):
            # Query the number of layers for the decoder RNN.
            n_decoder_layers = self.hparams.decoder.n_gru_layers

            # Query the number of units for the decoder cells.
            n_decoder_units = self.hparams.decoder.n_decoder_gru_units

            # Query the number of units for the attention cell.
            n_attention_units = self.hparams.decoder.n_attention_units

            # Determine the current batch size.
            batch_size = tf.shape(encoder_outputs)[0]

            # Create the attention mechanism.
            attention_mechanism = tfc.seq2seq.BahdanauAttention(
                num_units=n_attention_units,
                memory=encoder_outputs,
                # memory_sequence_length=None,
                # dtype=tf.float32
            )

            # Create the attention RNN cell.
            attention_cell = tf.nn.rnn_cell.GRUCell(num_units=n_attention_units,
                                                    name='attention_gru_cell')

            # Apply the pre-net to each decoder input as show in [1], figure 1.
            attention_cell = PrenetWrapper(attention_cell,
                                           self.hparams.decoder.pre_net_layers,
                                           self.is_training())

            # Connect the attention cell with the attention mechanism.
            wrapped_attention_cell = tfc.seq2seq.AttentionWrapper(
                cell=attention_cell,
                attention_mechanism=attention_mechanism,
                attention_layer_size=None,
                alignment_history=True,
                output_attention=False,  # True for Luong-style att., False for Bhadanau-style.
                initial_cell_state=None
            )  # => (B, T_sent, n_attention_units) = (B, T_sent, 256)

            # ======================================================================================
            # NOTE: This is actually derived from the Tacotron 2 paper and only an experiment.
            # ======================================================================================
            # => (B, T_sent, n_attention_units * 2) = (B, T_sent, 512)
            # concat_cell = ConcatOutputAndAttentionWrapper(wrapped_attention_cell)

            # => (B, T_sent, n_decoder_units) = (B, T_sent, 256)
            # concat_cell = tfc.rnn.OutputProjectionWrapper(concat_cell, n_decoder_units)
            # ======================================================================================

            # Stack several GRU cells and apply a residual connection after each cell.
            # Before the input reaches the decoder RNN it passes through the attention cell.
            cells = [wrapped_attention_cell]
            for i in range(n_decoder_layers):
                # => (B, T_spec, n_decoder_units) = (B, T_spec, 256)
                cell = tf.nn.rnn_cell.GRUCell(num_units=n_decoder_units, name='gru_cell')
                # => (B, T_spec, n_decoder_units) = (B, T_spec, 256)
                cell = tf.nn.rnn_cell.ResidualWrapper(cell)
                cells.append(cell)

            # => (B, T_spec, n_decoder_units) = (B, T_spec, 256)
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

            # Project the final cells output to the decoder target size.
            # => (B, T_spec, target_size * reduction) = (B, T_spec, 80 * reduction)
            output_cell = tfc.rnn.OutputProjectionWrapper(
                cell=decoder_cell,
                output_size=self.hparams.decoder.target_size * self.hparams.reduction,
                # activation=tf.nn.sigmoid
            )

            # TODO: Experiment with initialising using the encoder state:
            # ".clone(cell_state=encoder_state)"
            # Derived from: https://github.com/tensorflow/nmt/blob/365e7386e6659526f00fa4ad17eefb13d52e3706/nmt/attention_model.py#L131
            decoder_initial_state = output_cell.zero_state(
                batch_size=batch_size,
                dtype=tf.float32
            )

            if self.is_training():
                # Create a custom training helper for feeding ground truth frames during training.
                helper = TacotronTrainingHelper(
                    batch_size=batch_size,
                    inputs=encoder_outputs,
                    outputs=self.inp_mel_spec,
                    input_size=self.hparams.decoder.target_size,
                    reduction_factor=self.hparams.reduction
                )
            else:
                # Create a custom inference helper that handles proper data feeding.
                helper = TacotronInferenceHelper(batch_size=batch_size,
                                                 input_size=self.hparams.decoder.target_size)

            decoder = seq2seq.BasicDecoder(cell=output_cell,
                                           helper=helper,
                                           initial_state=decoder_initial_state)

            if self.is_training():
                # During training we do not stop decoding manually. The decoder automatically
                # decodes as many time steps as are contained in the ground truth data.
                maximum_iterations = None
            else:
                # During inference we stop decoding after `maximum_iterations`. Note that when
                # using the reduction factor the RNN actually outputs
                # `maximum_iterations` * `reduction_factor` frames.
                maximum_iterations = self.hparams.decoder.maximum_iterations // self.hparams.reduction

            # Start decoding.
            decoder_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=maximum_iterations)

            # decoder_outputs => type=BasicDecoderOutput, (rnn_output, _)
            # final_state => type=AttentionWrapperState, (attention_wrapper_state, _, _)
            # final_sequence_lengths.shape = (B)

            # Create an attention alignment summary image.
            Tacotron._create_attention_summary(final_state)

        # shape => (B, T_spec // r, n_mels * r)
        return decoder_outputs.rnn_output

    def decoder2(self, inputs, memory):
        with tf.variable_scope('decoder2'):
            # Determine the current batch size.
            batch_size = tf.shape(inputs)[0]

            # Attention
            attention_cell = seq2seq.AttentionWrapper(
                PrenetWrapper(tf.nn.rnn_cell.GRUCell(256),
                              self.hparams.decoder.pre_net_layers,
                              self.is_training()),
                seq2seq.BahdanauAttention(256, memory),
                attention_layer_size=256,
                alignment_history=True,
                output_attention=True)  # [N, T_in, attention_depth=256]

            # Decoder (layers specified bottom to top):
            decoder_cell = tf.nn.rnn_cell.MultiRNNCell([
                attention_cell,
                tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.GRUCell(256)),
                tf.nn.rnn_cell.ResidualWrapper(tf.nn.rnn_cell.GRUCell(256))
            ], state_is_tuple=True)  # [N, T_in, decoder_depth=256]

            # Project onto r mel spectrograms (predict r outputs at each RNN step):
            output_cell = tfc.rnn.OutputProjectionWrapper(decoder_cell,
                                                          self.hparams.n_mels *
                                                          self.hparams.reduction)

            decoder_init_state = output_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            if self.is_training():
                max_iters = None
                mel_targets = tf.reshape(self.inp_mel_spec, [batch_size, -1, self.hparams.n_mels])
                helper = TacotronTrainingHelper(inputs,
                                                mel_targets,
                                                self.hparams.n_mels,
                                                self.hparams.reduction)
            else:
                max_iters = 1000 // 5
                helper = TacotronInferenceHelper(batch_size,
                                                 self.hparams.n_mels)

            (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                seq2seq.BasicDecoder(output_cell, helper, decoder_init_state),
                maximum_iterations=max_iters)  # [N, T_out/r, M*r]

            alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])
            tf.summary.image("stacked_alignments", tf.expand_dims(alignments, -1))

        return decoder_outputs, final_decoder_state

    def post_process(self, inputs):
        """
        Apply the CBHG based post-processing network to the spectrogram.

        Arguments:
            inputs (tf.Tensor):
                The shape is expected to be shape=(B, T, n_mels) with B being the
                batch size and T being the number of time frames.

        Returns:
            tf.Tensor:
                A tensor which shape is expected to be shape=(B, T_spec, 2 * n_gru_units) with B
                being the batch size and T being the number of time frames.
        """
        with tf.variable_scope('post_process'):
            # network.shape => (B, T_spec, 2 * n_gru_units)
            # state.shape   => (2, n_gru_units)
            network, state = cbhg(inputs=inputs,
                                  n_banks=self.hparams.post.n_banks,
                                  n_filters=self.hparams.post.n_filters,
                                  n_highway_layers=self.hparams.post.n_highway_layers,
                                  n_highway_units=self.hparams.post.n_highway_units,
                                  projections=self.hparams.post.projections,
                                  n_gru_units=self.hparams.post.n_gru_units,
                                  training=self.is_training())

        return network

    def model(self):
        """
        Builds the Tacotron model.
        """
        # inp_sentences.shape = (B, T_sent, ?)
        batch_size = tf.shape(self.inp_sentences)[0]

        # network.shape => (B, T_sent, 256)
        # encoder_state.shape => (B, 2, 256)
        encoder_outputs, encoder_state = self.encoder(self.inp_sentences)

        decoder_inputs = tf.concat((
            tf.zeros_like(self.inp_mel_spec[:, :1, :]),
            self.inp_mel_spec[:, :-1, :]
        ), 1)  # (N, Ty/r, n_mels*r)

        decoder_inputs = decoder_inputs[:, :, -self.hparams.n_mels:]  # (N, Ty/r, n_mels)

        # # shape => (B, T_spec // r, n_mels * r)
        # decoder_outputs = self.decoder(encoder_outputs, encoder_state)
        decoder_outputs, _ = self.decoder2(inputs=decoder_inputs,
                                           memory=encoder_outputs)

        print('decoder_outputs', decoder_outputs)
        decoder_outputs = tf.Print(decoder_outputs, [tf.shape(decoder_outputs)],
                                   'decoder_outputs.shape')

        self.inp_mel_spec = tf.Print(self.inp_mel_spec, [tf.shape(self.inp_mel_spec)], 'inp_mel_spec.shape')

        tf.summary.image("reduced_decoder_outputs", tf.expand_dims(decoder_outputs, -1))

        # shape => (B, T_spec, n_mels)
        decoder_outputs = tf.reshape(decoder_outputs, [batch_size, -1, self.hparams.n_mels])

        # shape => (B, T_spec, n_mels)
        self.output_mel_spec = decoder_outputs

        outputs = decoder_outputs
        if self.hparams.apply_post_processing:
            # shape => (B, T_spec, 256)
            outputs = self.post_process(outputs)

        # shape => (B, T_spec, (1 + n_fft // 2))
        outputs = tf.layers.dense(inputs=outputs,
                                  units=(1 + self.hparams.n_fft // 2),
                                  # activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  bias_initializer=tf.glorot_normal_initializer())

        # shape => (B, T_spec, (1 + n_fft // 2))
        self.output_linear_spec = outputs

        inp_mel_spec = self.inp_mel_spec
        inp_linear_spec = self.inp_linear_spec

        inp_mel_spec = tf.reshape(inp_mel_spec, [batch_size, -1, self.hparams.n_mels])
        inp_linear_spec = tf.reshape(inp_linear_spec,
                                     [batch_size, -1, (1 + self.hparams.n_fft // 2)])

        output_mel_spec = self.output_mel_spec
        output_linear_spec = self.output_linear_spec

        if self.is_training():
            # ======================================================================================
            mel_spec_img = tf.expand_dims(
                tf.reshape(inp_mel_spec[0], (1, -1, self.hparams.n_mels)), -1)

            mel_spec_img = tf.transpose(mel_spec_img, perm=[0, 2, 1, 3])
            mel_spec_img = tf.reverse(mel_spec_img, axis=tf.convert_to_tensor([1]))
            tf.summary.image('mel_spec_gt_loss', mel_spec_img, max_outputs=1)

            # Convert thew linear spectrogram into an image that can be displayed.
            # => shape=(1, T_spec, (1 + n_fft // 2), 1)
            linear_spec_image = tf.expand_dims(
                tf.reshape(inp_linear_spec[0], (1, -1, (1 + self.hparams.n_fft // 2))), -1)

            # => shape=(1, (1 + n_fft // 2), T_spec, 1)
            linear_spec_image = tf.transpose(linear_spec_image, perm=[0, 2, 1, 3])
            linear_spec_image = tf.reverse(linear_spec_image, axis=tf.convert_to_tensor([1]))
            tf.summary.image('linear_spec_gt_loss', linear_spec_image, max_outputs=1)
            # ======================================================================================

        # TODO: This should be part of an Inference / Evaluation helper.
        if self._mode == Mode.EVAL:
            # Get the number of ground truth frames in the spectrogram's.
            n_frames = tf.shape(inp_mel_spec)[1]

            # Limit the number of produced frames to that in the input data.
            output_mel_spec = output_mel_spec[:, : n_frames, :]
            output_linear_spec = output_linear_spec[:, : n_frames, :]

        # Calculate decoder Mel. spectrogram loss.
        self.loss_op_decoder = tf.reduce_mean(
            tf.abs(inp_mel_spec - output_mel_spec))

        # Calculate post-processing linear spectrogram loss.
        self.loss_op_post_processing = tf.reduce_mean(
            tf.abs(inp_linear_spec - output_linear_spec))

        # Combine the decoder and the post-processing losses.
        self.loss_op = self.loss_op_decoder + self.loss_op_post_processing

    def get_loss_op(self):
        """
        Get the models loss function.

        Returns:
            tf.Tensor
        """
        return self.loss_op

    def summary(self):
        """
        Create all summary operations for the model.

        Returns:
            tf.Tensor:
                A tensor of type `string` containing the serialized `Summary` protocol
                buffer containing all merged model summaries.
        """
        if self._mode == Mode.TRAIN:
            with tf.name_scope('loss'):
                tf.summary.scalar('loss', self.loss_op)
                tf.summary.scalar('loss_decoder', self.loss_op_decoder)
                tf.summary.scalar('loss_post_processing', self.loss_op_post_processing)

            with tf.name_scope('normalized_inputs'):
                # Convert the mel spectrogram into an image that can be displayed.
                # => shape=(1, T_spec, n_mels, 1)
                mel_spec_img = tf.expand_dims(
                    tf.reshape(self.inp_mel_spec[0],
                               (1, -1, self.hparams.n_mels)), -1)

                # => shape=(1, n_mels, T_spec, 1)
                mel_spec_img = tf.transpose(mel_spec_img, perm=[0, 2, 1, 3])
                mel_spec_img = tf.reverse(mel_spec_img, axis=tf.convert_to_tensor([1]))
                tf.summary.image('mel_spec', mel_spec_img, max_outputs=1)

                # Convert thew linear spectrogram into an image that can be displayed.
                # => shape=(1, T_spec, (1 + n_fft // 2), 1)
                linear_spec_image = tf.expand_dims(
                    tf.reshape(self.inp_linear_spec[0],
                               (1, -1, (1 + self.hparams.n_fft // 2))), -1)

                # => shape=(1, (1 + n_fft // 2), T_spec, 1)
                linear_spec_image = tf.transpose(linear_spec_image, perm=[0, 2, 1, 3])
                linear_spec_image = tf.reverse(linear_spec_image, axis=tf.convert_to_tensor([1]))
                tf.summary.image('linear_spec', linear_spec_image, max_outputs=1)

        with tf.name_scope('normalized_outputs'):
            # Convert the mel spectrogram into an image that can be displayed.
            # => shape=(1, T_spec, n_mels, 1)
            mel_spec_img = tf.expand_dims(
                tf.reshape(self.output_mel_spec[0],
                           (1, -1, self.hparams.n_mels)), -1)

            # => shape=(1, n_mels, T_spec, 1)
            mel_spec_img = tf.transpose(mel_spec_img, perm=[0, 2, 1, 3])
            mel_spec_img = tf.reverse(mel_spec_img, axis=tf.convert_to_tensor([1]))
            tf.summary.image('decoder_mel_spec', mel_spec_img, max_outputs=1)

            # Convert thew linear spectrogram into an image that can be displayed.
            # => shape=(1, T_spec, (1 + n_fft // 2), 1)
            linear_spec_image = tf.expand_dims(
                tf.reshape(self.output_linear_spec[0],
                           (1, -1, (1 + self.hparams.n_fft // 2))), -1)

            # => shape=(1, (1 + n_fft // 2), T_spec, 1)
            linear_spec_image = tf.transpose(linear_spec_image, perm=[0, 2, 1, 3])
            linear_spec_image = tf.reverse(linear_spec_image, axis=tf.convert_to_tensor([1]))
            tf.summary.image('linear_spec', linear_spec_image, max_outputs=1)

        # TODO: Turned off since it is only of used for debugging.
        if self._mode == Mode.EVAL:
            with tf.name_scope('inference_reconstruction'):
                win_len = ms_to_samples(self.hparams.win_len, self.hparams.sampling_rate)
                win_hop = ms_to_samples(self.hparams.win_hop, self.hparams.sampling_rate)
                n_fft = self.hparams.n_fft

                def __synthesis(spec):
                    print('synthesis ....', spec.shape)
                    linear_mag_db = inv_normalize_decibel(spec.T, 20, 100)
                    linear_mag = decibel_to_magnitude(linear_mag_db)

                    _wav = spectrogram_to_wav(linear_mag,
                                              win_len,
                                              win_hop,
                                              n_fft,
                                              50)

                    # save_wav('/tmp/reconstr.wav', _wav, model_params.sampling_rate, True)
                    return _wav

                reconstruction = tf.py_func(__synthesis, [self.output_linear_spec[0]], [tf.float32])

                tf.summary.audio('synthesized', reconstruction, self.hparams.sampling_rate)

        return tf.summary.merge_all()

    @staticmethod
    def model_placeholders(max_sent_len):
        """
        Create placeholders for feeding data into the Tacotron model.

        Arguments:
            max_sent_len (int):
                Maximal sentence length.

        Returns:
            inputs (:obj:`dict`):
                Input data placeholders. All data that is used for training or inference is
                consumed from this placeholders.
                The placeholder dictionary contains the following fields with keys of the same name:
                    - ph_sentences (tf.Tensor):
                        Batched integer sentence sequence with appended <EOS> token padded to same
                        length using the <PAD> token.
                        including the <EOS> token.
                    - ph_sentence_length (tf.Tensor):
                        Batched sequence lengths including the <EOS> token, excluding the padding.
                    - ph_mel_specs (tf.Tensor):
                        Batched Mel. spectrogram's that were padded to the same length in the
                        time axis using zero frames.
                    - ph_lin_specs (tf.Tensor):
                        Batched linear spectrogram's that were padded to the same length in the
                        time axis using zero frames.
                    - ph_time_frames (tf.Tensor):
                        Batched number of frames in the spectrogram's excluding the padding
                        frames.
        """
        # TODO: Make the batch size dimension, shape = None.
        ph_sentences = tf.placeholder(dtype=tf.int32, shape=(1, max_sent_len),
                                      name='ph_inp_sentences')

        ph_mel_specs = tf.placeholder(dtype=tf.float32,
                                      shape=(1,
                                             model_params.decoder.maximum_iterations // model_params.reduction,
                                             model_params.n_mels * model_params.reduction)
                                      , name='ph_mel_specs')

        ph_lin_specs = tf.placeholder(dtype=tf.float32,
                                      name='ph_lin_specs')

        ph_sentence_length = tf.placeholder(dtype=tf.int32,
                                            name='ph_sentence_length')

        ph_time_frames = tf.placeholder(dtype=tf.int32,
                                        name='ph_time_frames')

        # Collect all created placeholder in a dictionary.
        placeholder_dict = {
            'ph_sentences': ph_sentences,
            'ph_sentence_length': ph_sentence_length,
            'ph_mel_specs': ph_mel_specs,
            'ph_lin_specs': ph_lin_specs,
            'ph_time_frames': ph_time_frames
        }

        return placeholder_dict

    @staticmethod
    def _create_attention_summary(final_context_state):
        """
        Add an attention alignment plot to the models summaries.

        Arguments:
            final_context_state (tf.contrib.seq2seq.AttentionWrapperState):
                Final state and attention information of the decoder network.
        """
        attention_wrapper_state, unkn1, unkn2 = final_context_state

        cell_state, attention, _, alignments, alignment_history, attention_state = \
            attention_wrapper_state

        stacked_alignment_hist = alignment_history.stack()
        stacked_alignments = tf.transpose(stacked_alignment_hist, [1, 2, 0])
        tf.summary.image("stacked_alignments", tf.expand_dims(stacked_alignments, -1))
