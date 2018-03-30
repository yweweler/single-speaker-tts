import tensorflow as tf

from tacotron.layers import cbhg, pre_net


# TODO: Clean up document and comments.

class Tacotron:
    def __init__(self, hparams, inputs):
        self.hparams = hparams

        # Create placeholders for the input data.
        self.inp_sentences, self.inp_mel_spec, self.inp_linear_spec, self.seq_lengths = inputs
        self.pred_linear_spec = None
        self.loss_op = None

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
            ])

            # network.shape => (B, T, 256)
            embedded_char_ids = tf.nn.embedding_lookup(char_embeddings, inputs)

            embedded_char_ids = tf.Print(embedded_char_ids,
                                         [tf.shape(embedded_char_ids)],
                                         'encoder.embedded_char_ids.shape')

            # network.shape => (B, T, 128)
            network = pre_net(inputs=embedded_char_ids,
                              layers=self.hparams.encoder.pre_net_layers,
                              training=True)

            network = tf.Print(network, [tf.shape(network)], 'encoder.network.shape')

            # network.shape => (B, T, 128 * 2)
            network = cbhg(inputs=network,
                           n_banks=self.hparams.encoder.n_banks,
                           n_filters=self.hparams.encoder.n_filters,
                           n_highway_layers=self.hparams.encoder.n_highway_layers,
                           n_highway_units=self.hparams.encoder.n_highway_units,
                           projections=self.hparams.encoder.projections,
                           n_gru_units=self.hparams.encoder.n_gru_units,
                           training=True)

            network = tf.Print(network, [tf.shape(network)], 'encoder.cbhg.shape')

            # TODO: Encoder timestamps exactly match the number of character inputs.
            #       To my current understanding this so incorrect however since at some point I
            #       have to return a constant size encoded context representation.

        return network

    def decoder(self, inputs):
        with tf.variable_scope('decoder'):
            inputs = tf.Print(inputs, [tf.shape(inputs)], 'decoder.inputs.shape')

            # network.shape => (B, T, 128)
            network = pre_net(inputs=inputs,
                              layers=self.hparams.decoder.pre_net_layers,
                              training=True)

            network = tf.Print(network, [tf.shape(network)], 'decoder.pre_net.shape')

            # TODO: I am not sure how to handle the 128 pre-net to 256 gru conversion (
            # Does the attention con into play here?).

            # TODO: As far as I can see the paper does not use an BI-GRU for the decoder.

            n_gru_layers = self.hparams.decoder.n_gru_layers
            n_gru_units = self.hparams.decoder.n_gru_units
            cells = []
            for i in range(n_gru_layers):
                cell = tf.nn.rnn_cell.GRUCell(num_units=n_gru_units, name='gru_cell')
                residual_cell = tf.nn.rnn_cell.ResidualWrapper(cell)
                cells.append(residual_cell)

            stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)

            # TODO: Experiment with time_major input data to see what the performance gain could be.
            outputs, state = tf.nn.dynamic_rnn(cell=stacked_cells,
                                               inputs=network,
                                               dtype=tf.float32,
                                               # sequence_length=self.seq_lengths,
                                               scope='stacked_gru')

            outputs = tf.Print(outputs, [tf.shape(outputs)], 'decoder.stacked_gru.shape')

            network = tf.layers.dense(inputs=outputs,
                                      units=self.hparams.decoder.target_size,
                                      activation=None,
                                      kernel_initializer=tf.glorot_normal_initializer(),
                                      bias_initializer=tf.zeros_initializer(),
                                      name='target_lifter')

            network = tf.Print(network, [tf.shape(network)], 'decoder.lifter.shape')

            # TODO: 1 layer attention GRU (256 cells).

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
            network = cbhg(inputs=inputs,
                           n_banks=self.hparams.post.n_banks,
                           n_filters=self.hparams.post.n_filters,
                           n_highway_layers=self.hparams.post.n_highway_layers,
                           n_highway_units=self.hparams.post.n_highway_units,
                           projections=self.hparams.post.projections,
                           n_gru_units=self.hparams.post.n_gru_units,
                           training=True)

        return network

    def model(self):
        # Input shape=(B, T, E)

        # network.shape => (B, T//r, n_mels*r)
        network = self.inp_mel_spec
        batch_size = tf.shape(network)[0]

        network = self.encoder(self.inp_sentences)

        network = self.decoder(network)

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

        self.loss_op = tf.reduce_mean(tf.abs(linear_spec - self.pred_linear_spec))

    def get_loss_op(self):
        return self.loss_op

    def summary(self):
        tf.summary.scalar('loss', self.loss_op)

        with tf.name_scope('normalized_inputs'):
            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.inp_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1),
                             max_outputs=1)

        with tf.name_scope('normalized_outputs'):
            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.pred_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1),
                             max_outputs=1)

        return tf.summary.merge_all()

    def load(self, checkpoint_dir):
        raise NotImplementedError()
