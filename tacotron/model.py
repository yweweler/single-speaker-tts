import tensorflow as tf

from tacotron.layers import highway_network, conv_1d_filter_banks


class Tacotron:
    def __init__(self, hparams, inputs):
        self.hparams = hparams

        # Create placeholders for the input data.
        self.inp_mel_spec, self.inp_linear_spec, self.seq_lengths = inputs
        self.pred_linear_spec = None
        self.loss_op = None

        # Construct the network.
        self.model()

    def get_inputs_placeholders(self):
        # inp_mel_spec: shape=(N, T//r, n_mels*r)
        # inp_linear_spec: shape=(N, T//r, (1 + n_fft // 2)*r)
        return self.inp_mel_spec, self.inp_linear_spec

    def postprocess(self, inputs):
        with tf.variable_scope('postprocess'):
            self.pred_linear_spec = tf.layers.dense(inputs=inputs,
                                                    units=(
                                                                  1 + self.hparams.n_fft // 2) * self.hparams.reduction,
                                                    activation=tf.nn.sigmoid,
                                                    kernel_initializer=tf.glorot_normal_initializer(),
                                                    bias_initializer=tf.glorot_normal_initializer())

            self.pred_linear_spec = \
                highway_network(self.pred_linear_spec,
                                units=(1 + self.hparams.n_fft // 2) * self.hparams.reduction,
                                layers=4,
                                # activation=prelu,
                                scope='highway_network')

        return self.pred_linear_spec

    def model(self):
        # Input shape=(B, T, E)

        # Produces shape=(B, T, E//2 * K)
        network = conv_1d_filter_banks(self.inp_mel_spec, 2, 4)

        # TODO: Check if this strides over the correct axis.
        # Produces shape=(B, T, E//2 * K)
        network = tf.layers.max_pooling1d(inputs=network,
                                          pool_size=2,
                                          strides=1,
                                          padding='SAME')

        network = tf.layers.conv1d(inputs=network,
                                   filters=128,
                                   kernel_size=3,
                                   strides=1,
                                   activation=tf.nn.relu,
                                   padding='SAME')

        # TODO: Since they state that they use a batch_norm layer after each con1 layer I guess
        # here would be a good place for another one.

        network = tf.layers.conv1d(inputs=network,
                                   filters=128,
                                   kernel_size=3,
                                   strides=1,
                                   activation=None,
                                   padding='SAME')

        # TODO: Add a residual connection.
        # TODO: The residual connection dimensions do not add up currently.
        # TODO: I Need to rework all of this to support a Tacotron reduction factor > 1.
        network = tf.add(network, self.inp_mel_spec)

        # self.postprocess(self.inp_mel_spec)
        self.postprocess(network)

        # tf.losses.absolute_difference could be used either (in case reduction=Reduction.MEAN is used).
        self.loss_op = tf.reduce_mean(tf.abs(self.inp_linear_spec - self.pred_linear_spec))

    def get_loss_op(self):
        return self.loss_op

    def summary(self):
        tf.summary.scalar('loss', self.loss_op)

        # with tf.name_scope('reduced_inputs'):
        #     tf.summary.image('mel_spec', tf.expand_dims(self.inp_mel_spec, -1), max_outputs=1)
        #     tf.summary.image('linear_spec', tf.expand_dims(self.inp_linear_spec, -1), max_outputs=1)

        with tf.name_scope('normalized_inputs'):
            # tf.summary.image('mel_spec',
            #                  tf.expand_dims(
            #                      tf.reshape(self.inp_mel_spec[0],
            #                                 (1, -1, self.hparams.n_mels)), -1), max_outputs=1)

            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.inp_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1),
                             max_outputs=1)

        # with tf.name_scope('reduced_outputs'):
        #     # tf.summary.image('inp_linear_spec', tf.expand_dims(self.inp_linear_spec, -1), max_outputs=1)
        #     tf.summary.image('linear_spec', tf.expand_dims(self.pred_linear_spec, -1), max_outputs=1)

        with tf.name_scope('normalized_outputs'):
            tf.summary.image('linear_spec',
                             tf.expand_dims(
                                 tf.reshape(self.pred_linear_spec[0],
                                            (1, -1, (1 + self.hparams.n_fft // 2))), -1),
                             max_outputs=1)

        return tf.summary.merge_all()

    def load(self, checkpoint_dir):
        raise NotImplementedError()
