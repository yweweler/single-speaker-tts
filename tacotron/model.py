import tensorflow as tf
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib import seq2seq

from audio.conversion import inv_normalize_decibel, decibel_to_magnitude, ms_to_samples
from audio.io import save_wav
from audio.synthesis import spectrogram_to_wav
from tacotron.helpers import TacotronInferenceHelper
from tacotron.layers import cbhg, pre_net


# TODO: Clean up document and comments.

class Tacotron:
    def __init__(self, hparams, inputs, training=True):
        self.hparams = hparams

        # Create placeholders for the input data.
        self.inp_sentences, self.inp_mel_spec, self.inp_linear_spec, self.seq_lengths, self.inp_time_steps = inputs
        self.pred_linear_spec = None

        self.loss_op = None
        self.loss_op_decoder = None
        self.loss_op_post_processing = None

        self.debug_decoder_output = None

        self.training = training

        # Construct the network.
        self.model()

    def get_inputs_placeholders(self):
        # inp_mel_spec: shape=(N, T//r, n_mels*r)
        # inp_linear_spec: shape=(N, T//r, (1 + n_fft // 2)*r)
        return self.inp_mel_spec, self.inp_linear_spec

    def encoder(self, inputs):
        with tf.variable_scope('encoder'):
            char_embeddings = tf.get_variable("embedding", [
                self.hparams.vocabulary_size,
                self.hparams.encoder.embedding_size
            ],
                                              dtype=tf.float32)

            # network.shape => (B, T, 256)
            embedded_char_ids = tf.nn.embedding_lookup(char_embeddings, inputs)

            embedded_char_ids = tf.Print(embedded_char_ids,
                                         [tf.shape(embedded_char_ids), embedded_char_ids],
                                         'encoder.embedded_char_ids')

            # network.shape => (B, T, 128)
            network = pre_net(inputs=embedded_char_ids,
                              layers=self.hparams.encoder.pre_net_layers,
                              training=self.training)

            network = tf.Print(network, [tf.shape(network)], 'encoder.network.shape')

            # network.shape => (B, T, 128 * 2)
            network, state = cbhg(inputs=network,
                                  n_banks=self.hparams.encoder.n_banks,
                                  n_filters=self.hparams.encoder.n_filters,
                                  n_highway_layers=self.hparams.encoder.n_highway_layers,
                                  n_highway_units=self.hparams.encoder.n_highway_units,
                                  projections=self.hparams.encoder.projections,
                                  n_gru_units=self.hparams.encoder.n_gru_units,
                                  training=self.training)

            network = tf.Print(network, [tf.shape(network)], 'encoder.cbhg.shape')
            state = tf.Print(state, [tf.shape(state)], 'encoder.cbhg.state.shape')

        return network, state

    def decoder(self, inputs, encoder_state):
        with tf.variable_scope('decoder'):
            # TODO: Experiment with time_major input data to see what the performance gain could be.
            inputs = tf.Print(inputs, [tf.shape(inputs)], 'decoder.inputs.shape')

            encoder_state = tf.Print(encoder_state, [tf.shape(encoder_state)],
                                     'decoder.encoder_state.shape')

            # network.shape => (B, T, 128)
            network = pre_net(inputs=inputs,
                              layers=self.hparams.decoder.pre_net_layers,
                              training=self.training)
            # network = inputs

            network = tf.Print(network, [tf.shape(network)], 'decoder.pre_net.shape')

            n_gru_layers = self.hparams.decoder.n_gru_layers  # 2
            n_gru_units = self.hparams.decoder.n_gru_units  # 256

            # Stack several GRU cells and apply a residual connection after each cell.
            cells = []
            for i in range(n_gru_layers):
                cell = tf.nn.rnn_cell.GRUCell(num_units=n_gru_units, name='gru_cell')
                residual_cell = tf.nn.rnn_cell.ResidualWrapper(cell)
                cells.append(residual_cell)

            stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

            # Project the first cells inputs to the number of decoder units (this way the inputs
            # can be added to the cells outputs using residual connections).
            stacked_cell = contrib_rnn.InputProjectionWrapper(
                cell=stacked_cell,
                num_proj=n_gru_units,
                activation=None
            )

            # Project the final cells output to the decoder target size.
            stacked_cell = contrib_rnn.OutputProjectionWrapper(
                cell=stacked_cell,
                output_size=self.hparams.decoder.target_size,
                activation=tf.nn.sigmoid
            )

            if self.training:
                helper = seq2seq.TrainingHelper(
                    inputs=self.inp_mel_spec,
                    sequence_length=self.inp_time_steps,
                    time_major=False
                )
            else:
                batch_size = tf.shape(network)[0]
                helper = TacotronInferenceHelper(batch_size=batch_size, input_size=self.hparams.decoder.target_size)

            # TODO: I am using encoder_state as the initial state for each cell in the stack.
            # Maybe it would be better to use encoder_state only for the first cell and then
            # continue with zero_state for all other cells.
            decoder = seq2seq.BasicDecoder(cell=stacked_cell,
                                           helper=helper,
                                           initial_state=tuple([encoder_state] * n_gru_layers),
                                           output_layer=None)

            maximum_iterations = None
            if self.training is False:
                maximum_iterations = self.hparams.decoder.maximum_iterations

            final_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=maximum_iterations)

            # final_outputs.type == seq2seq.BasicDecoderOutput
            network = final_outputs.rnn_output
            network = tf.Print(network, [tf.shape(network)], 'decoder.rnn_output.shape')

            # TODO: 1 layer attention GRU (256 cells).

        self.debug_decoder_output = network
        return network

    def post_process(self, inputs):
        """
        Apply the CBHG based post-processing network to the spectrogram.

        Arguments:
            inputs (tf.Tensor):
                The shape is expected to be shape=(B, T, n_mels) with B being the
                batch size and T being the number of time frames.

        Returns:
            tf.Tensor:
                A tensor which shape is expected to be shape=(B, T, n_gru_units * 2) with B
                being the batch size and T being the number of time frames.
        """
        with tf.variable_scope('post_process'):
            network, state = cbhg(inputs=inputs,
                                  n_banks=self.hparams.post.n_banks,
                                  n_filters=self.hparams.post.n_filters,
                                  n_highway_layers=self.hparams.post.n_highway_layers,
                                  n_highway_units=self.hparams.post.n_highway_units,
                                  projections=self.hparams.post.projections,
                                  n_gru_units=self.hparams.post.n_gru_units,
                                  training=self.training)

        return network

    def model(self):
        batch_size = tf.shape(self.inp_sentences)[0]

        # inp_sentences.shape = (B, T_s, decoder.n_gru_units * 2 ) = (B, T_s, 256)
        self.inp_sentences = tf.Print(self.inp_sentences, [
            tf.shape(self.inp_sentences)
        ], message='tf.shape(self.inp_sentences)')
        print('self.inp_sentences', self.inp_sentences)

        # network.shape => (B, T_s, 256)
        # encoder_state.shape => (B, 256)
        network, encoder_state = self.encoder(self.inp_sentences)

        # network.shape => (B, T_w, 80)
        network = self.decoder(network, encoder_state)

        # Note: The Tacotron paper does not explicitly state that the reduction factor r was
        # applied during post-processing. My measurements suggest, that there is no benefit
        # in applying r during post-processing. Therefore the data is reshaped to the
        # original size before processing.

        # network.shape => (B, T, n_mels)
        network = tf.reshape(network, [batch_size, -1, self.hparams.n_mels])

        if self.hparams.apply_post_processing:
            # network.shape => (B, T, n_gru_units * 2)
            network = self.post_process(network)

        # TODO: Should the reduction factor be applied here?
        # network.shape => (B, T, (1 + n_fft // 2))
        network = tf.layers.dense(inputs=network,
                                  units=(1 + self.hparams.n_fft // 2),
                                  activation=tf.nn.sigmoid,
                                  kernel_initializer=tf.glorot_normal_initializer(),
                                  bias_initializer=tf.glorot_normal_initializer())

        self.pred_linear_spec = network

        # linear_spec.shape = > (B, T, (1 + n_fft // 2))
        linear_spec = tf.reshape(self.inp_linear_spec,
                                 [batch_size, -1, (1 + self.hparams.n_fft // 2)])

        if self.training:
            self.loss_op_decoder = tf.reduce_mean(tf.abs(self.inp_mel_spec - self.debug_decoder_output))
            self.loss_op_post_processing = tf.reduce_mean(tf.abs(linear_spec - self.pred_linear_spec))

            self.loss_op = self.loss_op_decoder + self.loss_op_post_processing

    def get_loss_op(self):
        return self.loss_op

    def summary(self):
        tf.summary.scalar('loss', self.loss_op)
        tf.summary.scalar('loss_decoder', self.loss_op_decoder)
        tf.summary.scalar('loss_post_processing', self.loss_op_post_processing)

        with tf.name_scope('normalized_inputs'):
            tf.summary.image('mel_spec',
                             tf.expand_dims(
                                 tf.reshape(self.inp_mel_spec[0],
                                            (1, -1, self.hparams.n_mels)), -1),
                             max_outputs=1)

            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.inp_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1),
                             max_outputs=1)

        with tf.name_scope('normalized_outputs'):
            tf.summary.image('decoder_mel_spec',
                             tf.expand_dims(
                                 tf.reshape(self.debug_decoder_output[0],
                                            (1, -1, self.hparams.n_mels)), -1),
                             max_outputs=1)

            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.pred_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1),
                             max_outputs=1)

        if self.training is True:
            with tf.name_scope('inference_reconstruction'):
                win_len = ms_to_samples(self.hparams.win_len, sampling_rate=self.hparams.sampling_rate)
                win_hop = ms_to_samples(self.hparams.win_hop, sampling_rate=self.hparams.sampling_rate)
                n_fft = self.hparams.n_fft

                def __synthesis(spec):
                    print('synthesis ....', spec.shape)
                    linear_mag_db = inv_normalize_decibel(spec.T, 20, 100)
                    linear_mag = decibel_to_magnitude(linear_mag_db)

                    spec = spectrogram_to_wav(linear_mag,
                                              win_len,
                                              win_hop,
                                              n_fft,
                                              50)

                    save_wav('/tmp/reconstr.wav', spec, 16000, True)
                    return spec

                # reconstruction = tf.py_func(__synthesis,
                #                             [self.pred_linear_spec[0], win_len, win_hop,
                #                              self.hparams.n_fft],
                #                             [tf.float32])

                reconstruction = tf.py_func(__synthesis, [self.pred_linear_spec[0]], [tf.float32])

                tf.summary.audio('synthesized', reconstruction, self.hparams.sampling_rate)

        return tf.summary.merge_all()

    def load(self, checkpoint_dir):
        raise NotImplementedError()
