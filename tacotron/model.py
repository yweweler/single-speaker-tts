import tensorflow as tf
import tensorflow.contrib as tfc
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib import seq2seq

from audio.conversion import inv_normalize_decibel, decibel_to_magnitude, ms_to_samples
from audio.synthesis import spectrogram_to_wav
from tacotron.helpers import TacotronInferenceHelper
from tacotron.layers import cbhg, pre_net
# TODO: Clean up document and comments.
from tacotron.wrappers import PrenetWrapper, ConcatOutputAndAttentionWrapper


class Tacotron:
    def __init__(self, hparams, inputs, training=True):
        self.hparams = hparams

        # Create placeholders for the input data.
        self.inp_sentences, self.seq_lengths, self.inp_mel_spec, self.inp_linear_spec, \
        self.inp_time_steps = inputs

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
        """
        Implementation of the CBHG based Tacotron encoder network.

        Arguments:
            inputs (tf.Tensor):
            The shape is expected to be shape=(B, T_sent, ) with B being the batch size, T_sent
            being the number of tokens in the sentence including the EOS token.

        Returns:
         (outputs, output_states):
            outputs (tf.Tensor): The output states (output_fw, output_bw) of the RNN concatenated
                over time. Its shape is expected to be shape=(B, T_sent, 2 * n_gru_units) with B being
                the batch size, T_sent being the number of tokens in the sentence including the EOS
                token.

            output_states (tf.Tensor): A tensor containing the forward and the backward final states
                (output_state_fw, output_state_bw) of the bidirectional rnn.
                Its shape is expected to be shape=(B, 2, n_gru_units) with B being the batch size.
        """
        with tf.variable_scope('encoder'):
            char_embeddings = tf.get_variable("embedding", [
                                                  self.hparams.vocabulary_size,
                                                  self.hparams.encoder.embedding_size
                                              ],
                                              dtype=tf.float32)

            # shape => (B, T_sent, 256)
            embedded_char_ids = tf.nn.embedding_lookup(char_embeddings, inputs)

            # shape => (B, T_sent, 128)
            network = pre_net(inputs=embedded_char_ids,
                              layers=self.hparams.encoder.pre_net_layers,
                              training=self.training)

            # network.shape => (B, T_sent, 2 * n_gru_units)
            # state.shape   => (2, n_gru_units)
            network, state = cbhg(inputs=network,
                                  n_banks=self.hparams.encoder.n_banks,
                                  n_filters=self.hparams.encoder.n_filters,
                                  n_highway_layers=self.hparams.encoder.n_highway_layers,
                                  n_highway_units=self.hparams.encoder.n_highway_units,
                                  projections=self.hparams.encoder.projections,
                                  n_gru_units=self.hparams.encoder.n_gru_units,
                                  training=self.training)

        return network, state

    def attention_rnn(self, units, inputs, memory):
        attention_mechanism = tfc.seq2seq.BahdanauAttention(
            num_units=units,  # TODO: Unsure how to choose this param.
            memory=memory,
            memory_sequence_length=None,
            dtype=tf.float32
        )

        # TODO: The attention rnn might also be bidirectional I guess?
        cell = tf.nn.rnn_cell.GRUCell(num_units=units, name='attention_gru_cell')

        attention_cell = tfc.seq2seq.AttentionWrapper(
            cell=cell,
            attention_mechanism=attention_mechanism,
            attention_layer_size=units,
            alignment_history=True,
            output_attention=False,  # True for Luong-style att., False for Bhadanau-style.
            initial_cell_state=None
        )

        # TODO: Sequence lengths for the attention rnn?
        outputs, output_state = tf.nn.dynamic_rnn(
            cell=attention_cell,
            inputs=inputs,
            dtype=tf.float32,
            scope='attention_gru'
        )

        return outputs, output_state

    def decoder_rnn(self, n_units, inputs):
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units=n_units)
        cell_bw = tf.contrib.rnn.GRUCell(num_units=n_units)

        # TODO: Sequence lengths for the decoder rnn?
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw,
            cell_bw=cell_bw,
            inputs=inputs,
            dtype=tf.float32,
            scope='decoder_gru'
        )

        return tf.concat(outputs, -1), tf.concat(output_states, -1)

    def decoder(self, inputs, encoder_state):
        with tf.variable_scope('decoder'):
            # TODO: Experiment with time_major input data to see what the performance gain could be.

            # === Attention ========================================================================
            attention_mechanism = tfc.seq2seq.BahdanauAttention(
                num_units=256,
                memory=inputs,
                memory_sequence_length=None,
                dtype=tf.float32
            )

            attention_cell = tf.nn.rnn_cell.GRUCell(num_units=256,
                                                    name='attention_gru_cell')

            wrapped_attention_cell = tfc.seq2seq.AttentionWrapper(
                cell=PrenetWrapper(attention_cell,
                                   self.hparams.decoder.pre_net_layers,
                                   # network.shape => (B, T, 128)
                                   self.training),
                attention_mechanism=attention_mechanism,
                attention_layer_size=None,
                alignment_history=True,
                output_attention=False,  # True for Luong-style att., False for Bhadanau-style.
                initial_cell_state=None
            )  # => (B, T_sent, 256)

            # === Decoder ==========================================================================
            # TODO: Apply the reduction factor.

            n_gru_layers = self.hparams.decoder.n_gru_layers  # 2
            n_gru_units = self.hparams.decoder.n_gru_units  # 256

            concat_cell = ConcatOutputAndAttentionWrapper(wrapped_attention_cell)
            concat_cell = tfc.rnn.OutputProjectionWrapper(concat_cell, 256)

            # Stack several GRU cells and apply a residual connection after each cell.
            cells = [concat_cell]
            for i in range(n_gru_layers):
                cell = tf.nn.rnn_cell.GRUCell(num_units=n_gru_units, name='gru_cell')
                cell = tf.nn.rnn_cell.ResidualWrapper(cell)
                cells.append(cell)

            decoder_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)

            # Project the final cells output to the decoder target size.
            decoder_cell = contrib_rnn.OutputProjectionWrapper(
                cell=decoder_cell,
                output_size=self.hparams.decoder.target_size,
                activation=tf.nn.sigmoid
            )

            # Determine batch size.
            batch_size = tf.shape(inputs)[0]

            # TODO: Init using the encoder state: ".clone(cell_state=encoder_state)"
            # Derived from: https://github.com/tensorflow/nmt/blob/365e7386e6659526f00fa4ad17eefb13d52e3706/nmt/attention_model.py#L131
            decoder_initial_state = decoder_cell.zero_state(
                batch_size=batch_size,
                dtype=tf.float32
            )
            # TODO: I am using encoder_state as the initial state for each cell in the stack.
            # Maybe it would be better to use encoder_state only for the first cell and then
            # continue with zero_state for all other cells.

            if self.training:
                # TODO: Re-implement train data feeding.
                helper = seq2seq.TrainingHelper(
                    inputs=self.inp_mel_spec,
                    sequence_length=self.inp_time_steps,
                    time_major=False
                )
            else:
                helper = TacotronInferenceHelper(batch_size=batch_size,
                                                 input_size=self.hparams.decoder.target_size)

            decoder = seq2seq.BasicDecoder(cell=decoder_cell,
                                           helper=helper,
                                           initial_state=decoder_initial_state)

            maximum_iterations = None
            if self.training is False:
                maximum_iterations = self.hparams.decoder.maximum_iterations

            final_outputs, final_state, final_sequence_lengths = seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,  # Not sure if this is necessary if I pass all
                # sequence length on they way.
                maximum_iterations=maximum_iterations)

            network = final_outputs.rnn_output
            Tacotron._create_attention_summary(final_state)

            # ======================================================================================
            # attention_rnn_outputs, final_state = self.attention_rnn(256, self.inp_mel_spec, network)
            # decoder_rnn_outputs, _ = self.decoder_rnn(128, attention_rnn_outputs)
            # final_outputs = attention_rnn_outputs + decoder_rnn_outputs
            #
            # Tacotron._create_attention_images_summary(final_state)
            #
            # # network = final_outputs.rnn_output
            # network = final_outputs
            #
            # network = tf.layers.dense(inputs=network,
            #                           units=80,
            #                           activation=tf.nn.sigmoid,
            #                           kernel_initializer=tf.glorot_normal_initializer(),
            #                           bias_initializer=tf.glorot_normal_initializer())
            # ======================================================================================

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
                                  training=self.training)

        return network

    def model(self):
        batch_size = tf.shape(self.inp_sentences)[0]

        # inp_sentences.shape = (B, T_sent, decoder.n_gru_units * 2 ) = (B, T_sent, 256)

        # network.shape => (B, T_sent, 256)
        # encoder_state.shape => (B, 2, 256)
        encoder_outputs, encoder_state = self.encoder(self.inp_sentences)

        # shape => (B, T_spec, 80)
        decoder_outputs = self.decoder(encoder_outputs, encoder_state)

        # Note: The Tacotron paper does not explicitly state that the reduction factor r was
        # applied during post-processing. My measurements suggest, that there is no benefit
        # in applying r during post-processing. Therefore the data is reshaped to the
        # original size before processing.

        # shape => (B, T_spec, n_mels)
        network = tf.reshape(decoder_outputs, [batch_size, -1, self.hparams.n_mels])

        if self.hparams.apply_post_processing:
            # shape => (B, T_spec, 256)
            network = self.post_process(network)

        # TODO: Should the reduction factor be applied here?
        # shape => (B, T_spec, (1 + n_fft // 2))
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
            self.loss_op_decoder = tf.reduce_mean(
                tf.abs(self.inp_mel_spec - self.debug_decoder_output))
            self.loss_op_post_processing = tf.reduce_mean(
                tf.abs(linear_spec - self.pred_linear_spec))

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

        # TODO: Turned off since it is only of use fr debugging.
        if self.training is True and False:
            with tf.name_scope('inference_reconstruction'):
                win_len = ms_to_samples(self.hparams.win_len,
                                        sampling_rate=self.hparams.sampling_rate)
                win_hop = ms_to_samples(self.hparams.win_hop,
                                        sampling_rate=self.hparams.sampling_rate)
                n_fft = self.hparams.n_fft

                def __synthesis(spec):
                    print('synthesis ....', spec.shape)
                    linear_mag_db = inv_normalize_decibel(spec.T, 35.7, 100)
                    linear_mag = decibel_to_magnitude(linear_mag_db)

                    _wav = spectrogram_to_wav(linear_mag,
                                              win_len,
                                              win_hop,
                                              n_fft,
                                              50)

                    # save_wav('/tmp/reconstr.wav', _wav, hparams.sampling_rate, True)
                    return _wav

                reconstruction = tf.py_func(__synthesis, [self.pred_linear_spec[0]], [tf.float32])

                tf.summary.audio('synthesized', reconstruction, self.hparams.sampling_rate)

        return tf.summary.merge_all()

    def load(self, checkpoint_dir):
        raise NotImplementedError()

    @staticmethod
    def _create_attention_summary(final_context_state):
        attention_wrapper_state, unkn1, unkn2 = final_context_state

        cell_state, attention, _, alignments, alignment_history, attention_state = \
            attention_wrapper_state
        print('cell_state', cell_state)
        print('attention', attention)
        print('alignments', alignments)
        print('alignment_history', alignment_history)
        print('attention_state', attention_state)

        print('unkn1', unkn1)
        print('unkn2', unkn2)

        # tf.summary.image("cell_state", tf.expand_dims(tf.reshape(cell_state[0], (1, 1, 256)), -1))
        # tf.summary.image("attention", tf.expand_dims(tf.reshape(attention[0], (1, 1, 256)), -1))
        tf.summary.image("alignments", tf.expand_dims(tf.expand_dims(alignments, -1), 0))
        tf.summary.image("attention_state", tf.expand_dims(tf.expand_dims(attention_state, -1), 0))

        # tf.summary.image("unkn1", tf.expand_dims(tf.reshape(unkn1[0], (1, 1, 256)), -1))
        # tf.summary.image("unkn2", tf.expand_dims(tf.reshape(unkn2[0], (1, 1, 256)), -1))

        stacked = alignment_history.stack()
        tf.summary.image("alignments_stacked", tf.expand_dims(stacked, -1))

        alignments = tf.transpose(stacked, [1, 2, 0])
        tf.summary.image("alignments_final", tf.expand_dims(alignments, -1))

        print('alignment_history_stacked', alignments)
